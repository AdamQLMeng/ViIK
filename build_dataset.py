import argparse
from typing import Optional, List
import os
from time import time

import numpy as np
from tqdm import tqdm

from utils.evaluation_utils import self_collision_check, collision_check
from utils.visualization_utils import set_environment, show_world
from utils.utils import (
    get_dataset_directory,
    safe_mkdir,
    print_tensor_stats,
    get_sum_joint_limit_range,
    get_dataset_filepaths,
    assert_joint_angle_tensor_in_joint_limits,
)
from jrl.robots import get_robot, Robot

import torch

TRAINING_SET_SIZE_SMALL = int(1e5)
TEST_SET_SIZE = int(2.5 * 1e4)


def print_saved_datasets_stats(filepath, robots):
    """Printout summary statistics for each dataset. Optionaly print out the default joint limits of all robots in
    `robots`
    """

    print("\tSummary info on all saved datasets:")

    def print_joint_limits(limits, robot=""):
        print(f"robot joint limits {robot}")
        print(f"-------------------------------")
        sum_range = 0
        for idx, (l, u) in enumerate(limits):
            sum_range += u - l
            print(f"  joint_{idx}: ({np.rad2deg(l)},\t{np.rad2deg(u)})")
        print(f"  sum_range: {np.rad2deg(sum_range)}")

    for robot in robots:
        print_joint_limits(robot.actuated_joints_limits, robot=robot)

    sp = "\t"
    print(f"\nrobot {sp} dataset_name {sp} sum_joint_range")
    print(f"----- {sp} ------------ {sp} ---------------")

    # For each child directory (for each robot) in config.DATASET_DIR:
    for dataset_directory, dirs, files in os.walk(filepath):

        dataset_name = dataset_directory.split("/")[-1]
        robot_name = dataset_name.split("_")[0]
        subset_name = "train"

        samples_file_path, _, _, _, _ = get_dataset_filepaths(dataset_directory, subset_name, None)
        samples = torch.load(samples_file_path)
        sum_joint_range = get_sum_joint_limit_range(samples)
        print(f"{robot_name} {sp} {subset_name} {sp}{sp} {torch.rad2deg(sum_joint_range)}")
        break


def self_collision_check_dataset(robot: Robot, samples, subset_name):
    return self_collision_check(robot, samples, f"[{subset_name}][check Self-colliding]: ")


def collision_check_dataset(robot: Robot, samples, subset_name, env_name):
    return self_collision_check(robot, samples, f"[{subset_name}][{env_name}][check collision with env]: ")


def sample_randomly(robot: Robot, dataset_size: int, joint_limit_eps: float = 1e-6):
    # sample randomly
    samples, poses = robot.sample_joint_angles_and_poses(
        dataset_size,
        joint_limit_eps=joint_limit_eps,
        only_non_self_colliding=False,
        tqdm_enabled=True,
    )
    return samples, poses


def load_samples(
        robot: Robot,
        dataset_directory: str,
        subset_name
):
    (
        samples_file_path, poses_file_path, is_self_collides_file_path, _, _,
    ) = get_dataset_filepaths(dataset_directory, subset_name, None)

    samples = torch.load(samples_file_path)
    poses = torch.load(poses_file_path)
    is_self_collides = torch.load(is_self_collides_file_path)
    # Sanity check
    for arr in [samples, poses]:
        for i in range(arr.shape[1]):
            assert torch.std(arr[:, i]) > 0.001, f"Error: Column {i} in samples has zero stdev"
    assert_joint_angle_tensor_in_joint_limits(robot.actuated_joints_limits, samples, f"samples_{subset_name}", 0.0)
    return samples.cpu().numpy(), poses.cpu().numpy(), is_self_collides.cpu().numpy()


def save_dataset_to_disk(
    robot: Robot,
    dataset_directory: str,
    data,
    env_name,
    subset_name,
    collision_check_for_exist_configs
):
    safe_mkdir(dataset_directory)
    safe_mkdir(os.path.join(dataset_directory, env_name))

    samples, poses, is_self_collides, is_collision = data

    samples = torch.tensor(samples, dtype=torch.float32)
    poses = torch.tensor(poses, dtype=torch.float32)
    is_self_collides = torch.tensor(is_self_collides, dtype=torch.float32)
    is_collision = torch.tensor(is_collision, dtype=torch.float32)

    # Sanity check
    for arr in [samples, poses]:
        for i in range(arr.shape[1]):
            assert torch.std(arr[:, i]) > 0.001, f"Error: Column {i} in samples has zero stdev"
    assert_joint_angle_tensor_in_joint_limits(robot.actuated_joints_limits, samples, f"samples_{subset_name}", 0.0)

    # Save training set
    (
        samples_file_path,
        poses_file_path,
        is_self_collides_file_path,
        is_collision_file_path,
        info_filepath,
    ) = get_dataset_filepaths(dataset_directory, subset_name, env_name)

    with open(info_filepath, "w") as f:
        f.write("Dataset info")
        f.write(f"  robot:             {robot.name}\n")
        f.write(f"  dataset_directory: {dataset_directory}\n")
        f.write(f"  dataset_name: {subset_name}\n")
        f.write(f"  dataset_size:     {len(samples)}\n")

    if not collision_check_for_exist_configs:
        torch.save(samples, samples_file_path)
        torch.save(poses, poses_file_path)
        torch.save(is_self_collides, is_self_collides_file_path)
    torch.save(is_collision, is_collision_file_path)


"""
# Build dataset

python scripts/build_dataset.py --robot_name=fetch --training_set_size=25000000 --only_non_self_colliding
python scripts/build_dataset.py --robot_name=panda --training_set_size=25000000 --only_non_self_colliding
python scripts/build_dataset.py --robot_name=fetch_arm --training_set_size=25000000 --only_non_self_colliding
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot_name", type=str, default="panda")
    parser.add_argument("--env_name_list", type=str, default=["env4", "env5", "env6", "env7", "env8", "env9", "env10"])#
    parser.add_argument("--subset_size", type=list, default=[int(5e3), int(1.5e3), int(2.5e3)])
    parser.add_argument("--subset_list", type=list, default=["train", "val", "test"])
    parser.add_argument("--obs_mesh_root", type=str, default="data/env/")
    parser.add_argument("--collision_check_for_exist_configs", type=bool, default=True)
    args = parser.parse_args()

    robot = get_robot(args.robot_name)

    # Build dataset
    dset_directory = get_dataset_directory(robot.name)
    print(f"Building dataset for robot: {robot} (Will save in {dset_directory})")
    t0 = time()
    for env_name in args.env_name_list:
        robot = get_robot(args.robot_name)
        obs_mesh_dir = os.path.join(args.obs_mesh_root, env_name, "mesh")
        obstacles = [os.path.join(obs_mesh_dir, i) for i in os.listdir(obs_mesh_dir)]
        print(f"obstacles in '{env_name}': {obstacles}")
        set_environment(robot, obstacles)
        for subset_name, size in zip(args.subset_list, args.subset_size):
            if not args.collision_check_for_exist_configs:
                print(f"Sample randomly for subset '{subset_name}'!")
                samples, poses = sample_randomly(robot, size, joint_limit_eps=0.004363323129985824)
                is_self_collides = self_collision_check_dataset(robot, samples, subset_name)
            else:
                print(f"Load from exist samples files for subset '{subset_name}'!")
                samples, poses, is_self_collides = load_samples(robot, dset_directory, subset_name)
            print(f"Has {len(samples)} configs for subset '{subset_name}' with "
                  f"{int(sum(is_self_collides))} self-colliding configs.")

            is_collision = collision_check_dataset(robot, samples, subset_name, env_name)
            print(f"{int(sum(is_collision))} configs colliding with environment!")
            save_dataset_to_disk(robot, dset_directory, (samples, poses, is_self_collides, is_collision), env_name, subset_name, args.collision_check_for_exist_configs)

    print(f"Saved dataset (subsets: {args.subset_list}) with {args.subset_size} samples in {time() - t0:.2f} seconds")
    print_saved_datasets_stats(dset_directory, [robot])
