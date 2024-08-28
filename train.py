import argparse
import os
import random
import sys
from time import time
from typing import Union

import cv2
import numpy as np
import tqdm
from PIL import Image
from jrl.robots import get_robot

from pytorch_lightning import Trainer, seed_everything
import torch
from torch.utils.tensorboard import SummaryWriter

from utils.utils import img_transform
from model.VisionIKFlow import VisionIKFlow
from model.lit_model import IKLitModel
from model.lit_data import IKLitDataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="cinn w/ softflow CLI")

    parser.add_argument("--robot_name", type=str, default='panda')
    parser.add_argument("--env_name_list", type=tuple,
                        default=["env6", "env7", "env8", "env9", "env10"])
    # "env1", "env2", "env3", "env4", "env5", "env6", "env7", "env8", "env9", "env10"

    # Model parameters
    parser.add_argument("--num_nodes_img", type=int, default=6)
    parser.add_argument("--num_nodes_value", type=int, default=36)
    parser.add_argument("--dim_dist", type=int, default=9)
    parser.add_argument("--dim_cond", type=int, default=7)
    parser.add_argument("--img_ch", type=int, default=4)
    parser.add_argument("--dim_img", type=int, default=3)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--dim_latent", type=int, default=128)
    parser.add_argument("--with_rfilm", type=bool, default=False)

    # Training parameters
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--num_used_imgs", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--gamma", type=float,
                        default=0.994260074,
                        help="0.995405417-1500 (1500-1e-5) 0.994260074-1200 (1300-1.5e-5)"
                             "0.992354096-900 (1000-2e-5) 0.988553095-600 (700-3e-5)")
    parser.add_argument("--learning_rate", type=float, default=1.5e-05)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--step_lr_every_n_epoch", type=int, default=1)
    parser.add_argument("--gradient_clip_val", type=float, default=1)
    parser.add_argument("--lambd", type=float, default=1)
    parser.add_argument("--weight_decay", type=float, default=1.8e-05)

    # Logging options
    parser.add_argument("--val_set_size", type=int, default=10)
    parser.add_argument("--log_every", type=int, default=10)

    args = parser.parse_args()
    print("\nArgparse arguments:\n", args)

    exp_name = (f'exp_{args.num_nodes_value}_{args.num_nodes_img}_{args.env_name_list[0]}{len(args.env_name_list)}'
                f'_{args.batch_size}_{args.gamma}_{args.learning_rate}_{args.step_lr_every_n_epoch}'
                + ('_with_rfilm' if args.with_rfilm else ''))
    # exp_name = (f'exp_{args.num_nodes_value}_{args.num_nodes_img}_env{len(args.env_name_list)}'
    #             f'_{args.batch_size}_{args.gamma}_{args.learning_rate}_{args.step_lr_every_n_epoch}'
    #             + ('_with_rfilm' if args.with_rfilm else ''))

    # sets seeds for numpy, torch, torch.cuda, and random.
    seed_everything(args.seed, workers=True)

    # logger
    log_dir = "./runs/" + exp_name + '_' + str(time())
    log_tag = ""  # "train/"
    logger = SummaryWriter(log_dir=log_dir)
    print("log directory: ", log_dir)

    # Load robot
    robot = get_robot(args.robot_name)

    # device
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    print(f"Using device: {device}")

    torch.autograd.set_detect_anomaly(False)
    data_module = IKLitDataset(robot.name, args.env_name_list, args.batch_size, args.val_set_size)
    train_dataloader = data_module.train_dataloader()
    idx_list = []
    for i, env_name in enumerate(args.env_name_list):
        env_imgs = data_module.env_imgs[env_name]
        idx = env_imgs["Cameras"][0]["idx_list"][:min(len(train_dataloader), args.num_used_imgs)]
        for j in idx:
            idx_list.append(np.array([j, i]))
    random.shuffle(idx_list)

    filedir = f'experiments/{exp_name}'  # should be relative path
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    model_latest = os.path.join(filedir, "model_latest.pth")

    model = VisionIKFlow(robot, args.dim_dist+1, args.img_ch,
                         args.dim_img, args.img_size, dim_latent=args.dim_latent,
                         num_nodes_img=args.num_nodes_img,
                         num_nodes_value=args.num_nodes_value,
                         with_rfilm=args.with_rfilm,
                         device=device)

    # print(model.nn_model.encoder)
    print(f"{len(idx_list)} images are used in training! ({idx_list[:5]})")
    print(f"latest model: {model_latest} ({os.path.exists(model_latest)})")

    LitModel = IKLitModel(
        model=model,
        dim_dist=args.dim_dist,
        dim_cond=args.dim_cond,
        idx_list=idx_list,
        env_name_list=args.env_name_list,
        learning_rate=args.learning_rate,
        log_every=args.log_every,
        gradient_clip=args.gradient_clip_val,
        gamma=args.gamma,
        step_lr_every_n_epoch=args.step_lr_every_n_epoch,
        weight_decay=args.weight_decay,
        resume=model_latest,
        logger=logger)

    # idx_list_tqdm = tqdm.tqdm(LitModel.idx_list, desc=f"[Scan images]: ", file=sys.stdout)
    # for i in idx_list_tqdm:
    #     env_imgs = data_module.env_imgs[args.env_name_list[i[-1]]]
    #     imgs = [os.path.join(cam["cam_dir"], cam["imgs"][i[0]]) for cam in env_imgs["Cameras"]]
    #     # imgs = [Image.open(i) for i in imgs]
    #     for p in imgs:
    #         rst = cv2.imread(p)
    #         if rst is None:
    #             os.remove(p)
    #             print(p, " removed!")

    start_epoch = LitModel.epoch
    LitModel.scaler.set_growth_interval(len(train_dataloader))
    LitModel.scaler.set_growth_factor(1.023292992)  # 100 epochs for growing to 1
    for i in range(start_epoch, args.epochs):
        LitModel.epoch = i
        print(f"Learning rate: {LitModel.get_lr()}; Step every n epoch: {data_module.batch_size}.")
        data_loader = tqdm.tqdm(train_dataloader, desc=f"[Train][Epoch {i}]: ", file=sys.stdout)
        LitModel.model.train()
        if (LitModel.epoch * len(data_loader)) % len(idx_list) ==0:
            random.shuffle(LitModel.idx_list)
            print("Shuffled!", LitModel.idx_list[:2])
        for j, batch in enumerate(data_loader):
            index = LitModel.epoch * len(data_loader) + j
            index = LitModel.idx_list[index % len(idx_list)]

            configs, pose = batch
            configs = torch.cat([configs[:, :robot.n_dofs+1], configs[:, robot.n_dofs+index[-1]+1:robot.n_dofs+index[-1]+2]], dim=1)

            env_imgs = data_module.env_imgs[args.env_name_list[index[-1]]]
            imgs = [os.path.join(cam["cam_dir"], cam["imgs"][index[0]]) for cam in env_imgs["Cameras"]]
            imgs = [img_transform(Image.open(i)) for i in imgs]
            imgs = torch.cat(imgs, dim=0).unsqueeze(dim=0).float().cuda()

            rst = LitModel.training_step((configs, pose, imgs), j)
            data_loader.desc = (f"[Train][Epoch {i}]"
                                f"[Loss {np.round(LitModel.loss[-1], 2)}]"
                                f"[Entropy {np.round(rst['log_n_z0'].cpu().detach().numpy(), 2)}]"
                                f"[Cond {np.round(rst['cond_div'].cpu().detach().numpy(), 2)}]"
                                f"[Value {np.round(rst['value_div'].cpu().detach().numpy(), 2)}]: ")
            LitModel.log.add_scalar(log_tag + 'Entropy', rst['log_n_z0'].cpu().detach().numpy(), LitModel.epoch * len(data_loader) + j)
            LitModel.log.add_scalar(log_tag + 'cond_div', rst['cond_div'].cpu().detach().numpy(), LitModel.epoch * len(data_loader) + j)
            LitModel.log.add_scalar(log_tag + 'value_div', rst['value_div'].cpu().detach().numpy(), LitModel.epoch * len(data_loader) + j)
            LitModel.log.add_scalar(log_tag + 'loss', LitModel.loss[-1], LitModel.epoch * len(data_loader) + j)
            LitModel.log.add_scalar(log_tag + "scaled_loss", LitModel.scaled_loss, LitModel.epoch * len(data_loader) + j)
            LitModel.log.add_scalar(log_tag + "scale", LitModel.scaler.get_scale(), LitModel.epoch * len(data_loader) + j)

        LitModel.lr_scheduler.step()
        LitModel.on_epoch_end()
