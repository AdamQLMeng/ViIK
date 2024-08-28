import os.path
import random
from typing import Tuple, Dict, List
from time import time

from PIL import Image
from einops import repeat
from jrl.robots import Fetch
import numpy as np
import torch
from pytorch_lightning.core.module import LightningModule

from model.VisionIKFlow import VisionIKFlow, draw_latent
from torch.cuda.amp import GradScaler
from utils.utils import grad_stats, img_transform


class IKLitModel(LightningModule):
    def __init__(
        self,
        model: VisionIKFlow,
        dim_dist,
        dim_cond,
        idx_list: list,
        env_name_list: list,
        learning_rate: float,
        gamma: float = 0.975,
        samples_per_pose: int = 1000,
        gradient_clip: float = float("inf"),
        step_lr_every_n_epoch: int = 4,
        weight_decay: float = 1.8e-05,
        log_every: int = 1e10,
        resume: str = "./model_latest.pth",
        logger=None
    ):
        super().__init__()
        self.validation_step_outputs = []
        self.model = model
        self.dim_tot = model.dim_dist
        self.ndof = self.model.robot.n_dofs
        self.log_every = log_every
        self.resume = resume
        self.save_dir = os.path.join(os.getcwd(), self.resume[:-len("model_latest.pth")])
        self.alpha = 0.0
        self.beta = 0.0
        self.loss = []
        self.no_improvement = 0
        self.best_loss = torch.inf
        self.epoch = 0
        self.gradient_clip = gradient_clip
        self.samples_per_pose = samples_per_pose
        self.dim_dist = dim_dist
        self.dim_cond = dim_cond
        self.noise_type = "padding"
        self.padding_scale = 2
        self.scaler = GradScaler(init_scale=0.1)
        self.scaled_loss = 0
        self.config_opt(learning_rate, gamma, weight_decay, step_lr_every_n_epoch)
        self.log = logger
        self.idx_list = idx_list
        self.env_name_list = env_name_list

    def config_opt(self, learning_rate, gamma, weight_decay, step_lr_every):
        """Configure the optimizer and learning rate scheduler"""
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_lr_every,
            gamma=gamma, verbose=False)

        self.auto_resume(optimizer, lr_scheduler)

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    def auto_resume(self, optimizer, lr_scheduler):
        if os.path.exists(self.resume):
            checkpoint = torch.load(self.resume, map_location='cpu')
            print("Checkpoint keys:", [i for i in checkpoint.keys()])
            self.model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_states'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            if len(checkpoint["scaler"]):
                self.scaler.load_state_dict(checkpoint["scaler"])
            else:
                self.scaler = GradScaler(enabled=False)
            self.epoch = checkpoint['epoch'] + 1
            self.best_loss = checkpoint["best_loss"]
            self.no_improvement = checkpoint["no_improvement"]
            if "idx_list" in checkpoint.keys():
                self.idx_list = checkpoint["idx_list"]
            print(f"Resume from: {self.resume}; "
                  f"Epochs: {self.epoch}; "
                  f"Best Loss: {self.best_loss}; "
                  f"No Improvement: {self.no_improvement};"
                  f"Index list: {self.idx_list[:2]}.")

    def safe_log_metrics(self, vals: Dict):
        assert isinstance(vals, dict)
        try:
            self.logger.log_metrics(vals)
        except AttributeError:
            pass

    def get_lr(self) -> float:
        return self.lr_scheduler.get_last_lr()[0]

    def training_step(self, batch, batch_idx):
        configs, poses, imgs = batch
        x = self.add_noise(configs, 7, self.dim_dist-7, 1)
        poses = self.add_noise(poses, 7, self.dim_cond-7, 0)
        t0 = time()
        rst = self.model(imgs, poses, x)
        t = time() - t0
        loss = rst["value_div"] - self.alpha * rst["log_n_z0"] + self.beta * rst["cond_div"]

        if torch.isnan(loss):
            raise ValueError(f"loss is nan! ({loss}, {rst})")

        scaled_loss = self.scaler.scale(loss)
        scaled_loss.backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        torch.nn.utils.clip_grad_value_(self.parameters(), self.gradient_clip)

        if self.scaler.get_scale() >= 1 and self.scaler.is_enabled():
            self.scaler = GradScaler(enabled=False)

        self.loss.append(loss.item())
        self.scaled_loss = scaled_loss.item()
        return rst

    def on_epoch_end(self):
        loss = np.mean(self.loss)
        self.loss = []
        if loss < self.best_loss:
            self.best_loss = loss
            torch.save({
                'epoch': self.epoch,
                "global_step": self.global_step,
                'state_dict': self.model.state_dict(),
                "best_loss": self.best_loss,
                "env_name_list": self.env_name_list,
            }, os.path.join(self.save_dir, 'checkpt_best.pth'))
            self.no_improvement = 0
            print("--> Best model saved!")
        else:
            self.no_improvement += 1

        latest_checkpt = {
            'epoch': self.epoch,
            "global_step": self.global_step,
            'state_dict': self.model.state_dict(),
            "optimizer_states": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "best_loss": self.best_loss,
            "no_improvement": self.no_improvement,
            "idx_list": self.idx_list,
        }
        current_model_f = os.path.join(self.save_dir, f'model_{self.epoch%2}.pth')
        current_model_f = os.path.expanduser(current_model_f)
        torch.save(latest_checkpt, current_model_f)
        print(f"--> The model of Epoch {self.epoch} saved! ({current_model_f})")
        link_f = os.path.expanduser(os.path.join(self.save_dir, "model_latest.pth"))
        if os.path.exists(link_f):
            os.remove(link_f)
        os.symlink(os.path.relpath(current_model_f, start=os.path.dirname(link_f)), link_f)
        print("--> Latest model link created!")
        self.safe_log_metrics({"tr/loss_avg": loss, "tr/best_loss": self.best_loss})
        print(f"Alpha: {self.alpha}, Beta: {self.beta}, Loss: {loss}, Best Loss: {self.best_loss}, "
              f"No Improvement: {self.no_improvement}/200")

    def add_noise(self, configs, dim_continue, dim_discrete, dim_pad):
        return self.add_paddingflow_noise(configs, dim_continue, dim_discrete, dim_pad)

    # for PaddingFlow
    def add_paddingflow_noise(self, x, dim_continue, dim_discrete, dim_pad):
        B, D = x.shape
        noise_pad = torch.zeros([B, dim_continue])

        if dim_discrete:
            noise = 0.01 * torch.randn([B, dim_discrete])
            noise = torch.cat([noise_pad, noise], dim=1)
        else:
            noise = noise_pad

        if dim_pad:
            x_pad = self.padding_scale * torch.randn([x.shape[0], dim_pad]).to(x).float()  # padding
            xx = torch.cat([x + noise, x_pad], dim=1)
        else:
            xx = x + noise
        return xx
