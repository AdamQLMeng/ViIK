import os
import random
from typing import List, Dict

from torch.utils.data import DataLoader
from pytorch_lightning.core.datamodule import LightningDataModule
import torch

from jrl.config import DEVICE as device
from model.config import DATASET_DIR
from utils.utils import get_sum_joint_limit_range, get_dataset_filepaths


class IKLitDataset(LightningDataModule):
    def __init__(self, robot_name: str,
                 env_name_list: list,
                 batch_size: int,
                 val_set_size: int,
                 subset_list: list=["train", "val", "test"],
                 prepare_data_per_node=True):
        self._robot_name = robot_name
        self.batch_size = batch_size
        self._val_set_size = val_set_size

        # If set to True will call prepare_data() on LOCAL_RANK=0 for every node. If set to False will only call from NODE_RANK=0, LOCAL_RANK=0.
        self.prepare_data_per_node = prepare_data_per_node
        self._log_hyperparams = True

        dataset_directory = os.path.join(DATASET_DIR, self._robot_name)
        env_imgs_root = os.path.join(DATASET_DIR, "env")
        assert os.path.isdir(dataset_directory), f"Directory '{dataset_directory}' doesn't exist"
        assert os.path.isdir(env_imgs_root), f"Directory '{env_imgs_root}' doesn't exist"

        self.samples = dict()
        self.poses = dict()
        self.env_imgs = {"root": env_imgs_root}
        for subset_name in subset_list:
            (samples_file_path, poses_file_path, is_self_collides_file_path, _, _) = get_dataset_filepaths(dataset_directory, subset_name, None)
            samples = torch.load(samples_file_path).to(device)
            poses = torch.load(poses_file_path).to(device)
            is_self_collides = torch.load(is_self_collides_file_path).unsqueeze(dim=1).to(device).int() * 2 - 1

            is_collision_list = []
            for env_name in env_name_list:
                (_, _, _, is_collision_file_path, _) = get_dataset_filepaths(dataset_directory, subset_name, env_name)

                is_collision = torch.load(is_collision_file_path).unsqueeze(dim=1).to(device).int() * 2 - 1
                is_collision_list.append(is_collision)

            self.samples[subset_name] = torch.cat([samples, is_self_collides] + is_collision_list, dim=1)
            self.poses[subset_name] = poses
            print(f"Subset '{subset_name}' loaded ! "
                  f"( Samples shape: {self.samples[subset_name].shape}, Poses shape: {self.poses[subset_name].shape})")

        for env_name in env_name_list:
            env_imgs_dir = os.path.join(env_imgs_root, env_name, "imgs")
            cam_folders = os.listdir(env_imgs_dir)
            self.env_imgs[env_name] = {"root": env_imgs_dir, "Cameras": []}
            for f in cam_folders:
                dir = os.path.join(env_imgs_dir, f)
                if os.path.isdir(dir):
                    imgs = os.listdir(dir)
                    imgs.sort()
                    idx_list = [i for i in range(0, len(imgs) - 1, 1)]
                    random.shuffle(idx_list)
                    self.env_imgs[env_name]["Cameras"].append(
                        {"cam_dir": dir, "num_imgs": len(imgs), "imgs": imgs, "idx_list": idx_list})
            print(f"Env Name: {env_name} (Load env images from {env_imgs_dir}, {[cam['num_imgs'] for cam in self.env_imgs[env_name]['Cameras']]})")

        self._sum_joint_limit_range = get_sum_joint_limit_range(self.samples["train"])
        self.allow_zero_length_dataloader_with_multiple_devices = False

    def add_dataset_hashes_to_cfg(self, cfg: Dict):
        cfg.update({
            "dataset_hashes": str([
                self.samples["train"].sum().item(),
                self.poses["train"].sum().item(),
                self.samples["val"].sum().item(),
                self.poses["val"].sum().item(),
                self.samples["test"].sum().item(),
                self.poses["test"].sum().item(),
            ])
        })

    def log_dataset_sizes(self, epoch=0, batch_nb=0):
        """Log the training and testset size to wandb"""
        assert self.samples["train"].shape[0] == self.poses["train"].shape[0]
        assert self.samples["val"].shape[0] == self.poses["val"].shape[0]
        assert self.samples["test"].shape[0] == self.poses["test"].shape[0]

    def train_dataloader(self):
        return DataLoader(
            torch.utils.data.TensorDataset(self.samples["train"], self.poses["train"]),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            # see https://github.com/dbolya/yolact/issues/664#issuecomment-975051339
            generator=torch.Generator(device=device),
        )

    def val_dataloader(self):
        return DataLoader(
            torch.utils.data.TensorDataset(
                self.samples["val"][0 : self._val_set_size],
                self.poses["val"][0 : self._val_set_size]),
            batch_size=1,
            shuffle=False,
            drop_last=True,
        )

    def test_dataloader(self):
        return DataLoader(
            torch.utils.data.TensorDataset(self.samples["test"], self.poses["test"]),
            batch_size=1,
            shuffle=False,
            drop_last=True,
        )
