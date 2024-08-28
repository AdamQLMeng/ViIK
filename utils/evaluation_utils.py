import math
from typing import Tuple, List, Optional

import numpy as np
import torch
from einops import repeat

from jrl.robot import Robot
from jrl.conversions import geodesic_distance_between_quaternions
from jrl.config import DEVICE, PT_NP_TYPE
from tqdm import tqdm

""" Description of 'SOLUTION_EVALUATION_RESULT_TYPE':
- torch.Tensor: [n] tensor of positional errors of the IK solutions. The error is the L2 norm of the realized poses of
                    the solutions and the target pose.
- torch.Tensor: [n] tensor of angular errors of the IK solutions. The error is the angular geodesic distance between the
                    realized orientation of the robot's end effector from the IK solutions and the targe orientation.
- torch.Tensor: [n] tensor of bools indicating whether each IK solutions has exceeded the robots joint limits.
- torch.Tensor: [n] tensor of bools indicating whether each IK solutions is self colliding.
- float: Runtime
"""


def pose_errors(
    poses_1: PT_NP_TYPE, poses_2: PT_NP_TYPE, acos_epsilon: Optional[float] = None
) -> Tuple[PT_NP_TYPE, PT_NP_TYPE]:
    """Return the positional and rotational angular error between two batch of poses."""
    assert poses_1.shape == poses_2.shape, f"Poses are of different shape: {poses_1.shape} != {poses_2.shape}"

    if isinstance(poses_1, torch.Tensor):
        l2_errors = torch.norm(poses_1[:, 0:3] - poses_2[:, 0:3], dim=1)
    else:
        l2_errors = np.linalg.norm(poses_1[:, 0:3] - poses_2[:, 0:3], axis=1)
    angular_errors = geodesic_distance_between_quaternions(
        poses_1[:, 3 : 3 + 4], poses_2[:, 3 : 3 + 4], acos_epsilon=acos_epsilon
    )
    assert l2_errors.shape == angular_errors.shape
    return l2_errors, angular_errors


def pose_errors_cm_deg(
    poses_1: PT_NP_TYPE, poses_2: PT_NP_TYPE, acos_epsilon: Optional[float] = None
) -> Tuple[PT_NP_TYPE, PT_NP_TYPE]:
    """Return the positional and rotational angular error between two batch of poses in cm and degrees"""
    assert poses_1.shape == poses_2.shape, f"Poses are of different shape: {poses_1.shape} != {poses_2.shape}"
    l2_errors, angular_errors = pose_errors(poses_1, poses_2, acos_epsilon=acos_epsilon)
    if isinstance(poses_1, torch.Tensor):
        return 100 * l2_errors, torch.rad2deg(angular_errors)
    return 100 * l2_errors, np.rad2deg(angular_errors)


def solution_pose_errors(
    robot: Robot, solutions: torch.Tensor, target_poses: PT_NP_TYPE
) -> Tuple[np.ndarray, np.ndarray]:
    """Return the L2 and angular errors of calculated ik solutions for a given target_pose. Note: this function expects
    multiple solutions but only a single target_pose. All of the solutions are assumed to be for the given target_pose

    Args:
        robot (Robot): The Robot which contains the FK function we will use
        solutions (Union[torch.Tensor, np.ndarray]): [n x 7] IK solutions for the given target pose
        target_pose (np.ndarray): [7] the target pose the IK solutions were generated for

    Returns:
        Tuple[np.ndarray, np.ndarray]: The L2, and angular (rad) errors of IK solutions for the given target_pose
    """
    assert isinstance(
        target_poses, (np.ndarray, torch.Tensor)
    ), f"target_poses must be a torch.Tensor or np.ndarray (got {type(target_poses)})"
    assert isinstance(solutions, torch.Tensor), f"solutions must be a torch.Tensor (got {type(solutions)})"
    n_solutions = solutions.shape[0]
    if n_solutions >= 1000:
        print("Heads up: It may be faster to run solution_pose_errors() with pytorch directly on the cpu/gpu")

    if isinstance(target_poses, torch.Tensor):
        target_poses = target_poses.detach().cpu().numpy()

    ee_pose_ikflow = robot.forward_kinematics(solutions.detach().cpu().numpy())
    rot_output = ee_pose_ikflow[:, 3:]

    # Positional Error
    l2_errors = np.linalg.norm(ee_pose_ikflow[:, 0:3] - target_poses[:, 0:3], axis=1)
    rot_target = target_poses[:, 3:]
    assert rot_target.shape == rot_output.shape, f"rot_target: {rot_target.shape}; rot_output: {rot_output.shape}"

    # Surprisingly, this is almost always faster to calculate on the gpu than on the cpu. I would expect the opposite
    # for low number of solutions (< 200).
    q_target_pt = torch.tensor(rot_target, device=DEVICE, dtype=torch.float32)
    q_current_pt = torch.tensor(rot_output, device=DEVICE, dtype=torch.float32)
    ang_errors = geodesic_distance_between_quaternions(q_target_pt, q_current_pt).detach().cpu().numpy()
    return l2_errors, ang_errors


def calculate_joint_limits_exceeded(configs: torch.Tensor, joint_limits: List[Tuple[float, float]]) -> torch.Tensor:
    """Calculate if the given configs have exceeded the specified joint limits

    Args:
        configs (torch.Tensor): [batch x ndof] tensor of robot configurations
        joint_limits (List[Tuple[float, float]]): The joint limits for the robot. Should be a list of tuples, where each
                                                    tuple contains (lower, upper).
    Returns:
        torch.Tensor: [batch] tensor of bools indicating if the given configs have exceeded the specified joint limits
    """
    toolarge = configs > torch.tensor([x[1] for x in joint_limits], dtype=torch.float32, device=configs.device)
    toosmall = configs < torch.tensor([x[0] for x in joint_limits], dtype=torch.float32, device=configs.device)
    return torch.logical_or(toolarge, toosmall).any(dim=1)


def self_collision_check(robot: Robot, configs, desc=None):
    if desc:
        configs = tqdm(configs, desc=desc)
    is_self_colliding = [robot.config_self_collides(config) for config in configs]
    return torch.tensor(is_self_colliding).int()


def collision_check(robot: Robot, configs, desc=None):
    is_collision = []
    if desc:
        configs = tqdm(configs, desc=desc)
    for s in configs:
        rst = False
        robot.set_klampt_robot_config(s)
        for i in range(robot.klampt_world_model.numTerrains()):
            pairs = robot._klampt_collision_checker.robotTerrainCollisions(robot._klampt_robot, i)
            for link, terrain in pairs:
                assert not link.getName().split("_")[-1] == "link0", \
                    ("Link 0 should ignored in collision check between robot and floor! "
                     f"(Collision occurs between {link.getName()} and {terrain.getName()}!)")
                rst = True
        if not rst:
            for i in range(robot.klampt_world_model.numRigidObjects()):
                pairs = robot._klampt_collision_checker.robotObjectCollisions(robot._klampt_robot, i)
                for link, obs in pairs:
                    rst = True
        is_collision.append(rst)
    return torch.tensor(is_collision).int()


def calculate_l2_angular_jlimt_single_pose(robot: Robot, target_pose: PT_NP_TYPE, solutions: torch.Tensor):
    assert target_pose.shape[0] == 1 and len(target_pose.shape) == 2, "This function is only for single target pose!"

    joint_limits_exceeded = calculate_joint_limits_exceeded(solutions[:, :robot.n_dofs], robot.actuated_joints_limits)

    s_non_exceeded = []
    for s, is_exceeded in zip(solutions, joint_limits_exceeded):
        if not is_exceeded:
            s_non_exceeded.append(s.unsqueeze(dim=0))
    s_non_exceeded = torch.cat(s_non_exceeded, dim=0) if len(s_non_exceeded) else []

    if len(s_non_exceeded):
        target_poses_repeat = repeat(target_pose, "1 d -> n d", n=s_non_exceeded.shape[0])
        l2_errors, angular_errors = solution_pose_errors(robot, s_non_exceeded[:, :robot.n_dofs], target_poses_repeat)
    else:
        return 0, 0, joint_limits_exceeded, s_non_exceeded

    return l2_errors, angular_errors, joint_limits_exceeded, s_non_exceeded
