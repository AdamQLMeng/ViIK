from typing import List, Tuple, Optional, Union
import pickle
from time import time

import jrl.robots
from einops import repeat
from jrl.robots import Robot
import numpy as np
import torch
from torch import nn

from model.VisionFlow import VisionFlow


def draw_latent(latent_distribution: str, latent_scale: float, shape: Tuple[int, int]):
    """Draw a sample from the latent noise distribution for running inference"""
    assert latent_distribution in ["gaussian", "uniform"]
    assert latent_scale > 0
    assert len(shape) == 2
    if latent_distribution == "gaussian":
        return latent_scale * torch.randn(shape)
    if latent_distribution == "uniform":
        return 2 * latent_scale * torch.rand(shape) - latent_scale


class VisionIKFlow(nn.Module):
    def __init__(self, robot: Robot, dim_dist, img_ch, dim_img, img_size, dim_cond=7, num_nodes_img=2, num_nodes_value=2, dim_latent=128, with_rfilm=True, device="cpu"):
        super(VisionIKFlow, self).__init__()

        assert isinstance(robot, Robot), f"Error - robot should be Robot type, is {type(robot)}"

        self.robot = robot
        self.dim_pose = dim_cond
        self.dim_dist = dim_dist
        self.dim_latent = dim_latent
        self.num_nodes_img = num_nodes_img
        self.num_nodes_value = num_nodes_value

        # Note: Changing `nn_model` to `_nn_model` may break the logic in 'download_model_from_wandb_checkpoint.py'
        # Transform Node to map x_i from joint space to [-1, 1]
        x_invSig = torch.eye(self.dim_dist)
        x_Mu = torch.zeros(self.dim_dist)
        for i in range(robot.n_dofs):
            x_invSig[i, i] = 1.0 / max(abs(robot.actuated_joints_limits[i][0]), abs(robot.actuated_joints_limits[i][1]))

        self.nn_model = VisionFlow(self.dim_dist, self.dim_pose, self.num_nodes_img, self.num_nodes_value, img_ch*dim_img, img_size, dim_latent, with_rfilm=with_rfilm, x_invSig=x_invSig, x_Mu=x_Mu)
        self.ndof = self.robot.n_dofs
        self.device = device

    def forward(self, img, pose, config):
        y, log_det_py, z_mu, z_logvar, z0, z1, log_det_pz = self.nn_model(config, img, pose)
        loss, log_n_z0, cond_div, log_n_z1, value_div = self.nn_model.loss(y, log_det_py, z_mu, z_logvar, z0, z1, log_det_pz)
        return {
            'loss': loss,
            'log_n_z0': log_n_z0,
            'log_n_z1': log_n_z1,
            'cond_div': cond_div,
            'value_div': value_div
        }

    def solve(
        self,
        poses: np.ndarray,
        img,
        num_samples=100,
        clamp_to_joint_limits: bool = False,
        refine_solutions: bool = False,
        return_detailed: bool = False
    ):
        assert poses.shape[1] == self.dim_pose

        with torch.inference_mode():
            if not isinstance(poses, torch.Tensor):
                poses = torch.tensor(poses, dtype=torch.float32)

            rst = []
            for p in poses:
                rst.append(
                    self.solve_single_pose(p.unsqueeze(dim=0), img, num_samples, clamp_to_joint_limits,
                                           refine_solutions, return_detailed))
        return rst

    def solve_single_pose(
        self,
        pose,
        img,
        num_samples,
        clamp_to_joint_limits: bool = False,
        refine_solutions: bool = False
    ):
        """Internal function to call IKFlow, and format and return the output"""
        B, D = pose.shape
        assert D == self.dim_pose

        # Run model
        pose_repeat = repeat(pose, "1 d -> n d", n=num_samples)
        y = torch.randn([num_samples, self.dim_dist])
        # print(y.shape, pose_repeat.shape, img.shape)
        output_rev = self.nn_model(y, img, pose_repeat, rev=True)
        solutions = output_rev[:, :self.ndof]

        if clamp_to_joint_limits:
            solutions = self.robot.clamp_to_joint_limits(solutions.cpu().detach().numpy())

        # Refine if requested
        if refine_solutions:
            target_pose_s = pose_repeat.detach().cpu().numpy()[:, 0:7]
            solutions = self.refine_solutions(solutions, target_pose_s)

        return solutions, output_rev

    def refine_solutions(
        self,
        ikflow_solutions: torch.Tensor,
        target_pose: Union[List[float], np.ndarray],
        positional_tolerance: float = 1e-3,
    ) -> Tuple[torch.Tensor, float]:
        """Refine a batch of IK solutions using the klampt IK solver
        Args:
            ikflow_solutions (torch.Tensor): A batch of IK solutions of the form [batch x n_dofs]
            target_pose (Union[List[float], np.ndarray]): The target endpose(s). Must either be of the form
                                                            [x, y, z, q0, q1, q2, q3] or be a [batch x 7] numpy array
        Returns:
            torch.Tensor: A batch of IK refined solutions [batch x n_dofs]
        """
        t0 = time()
        b = ikflow_solutions.shape[0]
        if isinstance(target_pose, list):
            target_pose = np.array(target_pose)
        if isinstance(target_pose, np.ndarray) and len(target_pose.shape) == 2:
            assert target_pose.shape[0] == b, f"target_pose.shape ({target_pose.shape[0]}) != [{b} x {self.ndof}]"

        ikflow_solutions_np = ikflow_solutions.detach().cpu().numpy()
        refined = ikflow_solutions_np.copy()
        is_single_pose = (len(target_pose.shape) == 1) or (target_pose.shape[0] == 1)
        pose = target_pose

        for i in range(b):
            if not is_single_pose:
                pose = target_pose[i]
            ik_sol = self._robot.inverse_kinematics_klampt(
                pose, seed=ikflow_solutions_np[i], positional_tolerance=positional_tolerance
            )
            if ik_sol is not None:
                refined[i] = ik_sol

        return torch.tensor(refined, device=self.device)
