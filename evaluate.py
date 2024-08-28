import os
import argparse
from collections import namedtuple

import klampt
import torch
import numpy as np
import tqdm
from PIL import Image
from jrl.robots import Robot, get_robot

from model.VisionIKFlow import VisionIKFlow
from model.lit_data import IKLitDataset

from utils.utils import set_seed, img_transform, time_synchronized
from utils.evaluation_utils import collision_check, self_collision_check, calculate_l2_angular_jlimt_single_pose
from utils.visualization_utils import set_environment

set_seed()

_DEFAULT_LATENT_DISTRIBUTION = "gaussian"
_DEFAULT_LATENT_SCALE = 1.
ErrorStats = namedtuple(
    "ErrorStats", "env_name num_samples_per_pose "
                  "mean_l2_error_mm mean_angular_error_deg pct_joint_limits_exceeded "
                  "pct_self_collision iou_self_collision pct_success_self_collision "
                  "pct_collision iou_collision pct_success_collision"
)


def calculate_collision_by_fn(
        robot: Robot,
        solutions,
        rst_collision,
        collision_fn,
        desc
):
    assert solutions[0].shape[-1] == robot.n_dofs
    assert len(solutions) == len(rst_collision)

    num_configs = 0
    num_s_tp = 0
    num_s_fp = 0
    num_s_fn = 0
    rst_success = []

    data = tqdm.tqdm(zip(solutions, rst_collision), desc=desc, total=len(solutions))
    for s, r in data:
        gt_collision = collision_fn(robot, s.detach().cpu().numpy())
        rst_c = (r > 0).int().cuda()

        gt = []
        pred = []
        for g, p in zip(gt_collision, rst_c):
            if p == 0:
                num_configs += 1
                gt.append(g)
                pred.append(p)
                if g == 0:
                    num_s_tp += 1
                else:
                    num_s_fp += 1
            else:
                if g == 0:
                    num_s_fn += 1
        assert sum(pred) == 0
        gt = torch.tensor(gt)
        pred = torch.tensor(pred)

        if torch.sum((pred == gt).int()):
            rst_success.append(1)
        else:
            rst_success.append(0)

    collision_states = {"Collision Rate": 100 * (num_s_fp / num_configs),
                        "IoU of Collision": 100 * (num_s_tp / (num_configs + num_s_fn)),
                        "Success rate": 100 * (sum(rst_success) / len(rst_success))
                        }
    return collision_states


def calculate_collision_with_env(
        robot: Robot,
        solutions,
        rst_collision,
        desc
):
    return calculate_collision_by_fn(robot, solutions, rst_collision, collision_check, desc)


def calculate_collision_with_self(
        robot: Robot,
        solutions,
        rst_collision,
        desc
):
    return calculate_collision_by_fn(robot, solutions, rst_collision, self_collision_check, desc)


def calculate_l2_angular_jlimt(robot: Robot, target_poses, solutions: list):
    assert target_poses.shape[0] == len(solutions), "The number of target poses and solitions' should match!"

    l2_errors = []
    angular_errors = []
    joint_limits_exceeded = []
    s_non_exceeded = []
    data = tqdm.tqdm(zip(solutions, target_poses), desc="[Calculate L2 Angular JLimt_exceeded]: ", total=len(solutions))
    for (s, p) in data:
        l2_e, angular_e, jlimt_exceeded, s_n_exceeded = calculate_l2_angular_jlimt_single_pose(robot, p.unsqueeze(dim=0), torch.from_numpy(s))
        joint_limits_exceeded += jlimt_exceeded.tolist()
        s_non_exceeded.append(s_n_exceeded)
        if (isinstance(l2_e, int) and l2_e == 0) or (isinstance(angular_e, int) and angular_e == 0):
            print("Warning: all solutions exceeded the joints limits!")
            continue

        l2_errors += l2_e.tolist()
        angular_errors += angular_e.tolist()

    l2_angular_jlimt = {"L2 Error": np.mean(l2_errors),
                        "Angular Error": np.mean(angular_errors),
                        "JLimt Rate": 100*np.mean(joint_limits_exceeded)}
    return l2_angular_jlimt, s_non_exceeded


def solve_all_ik(
        ik_solver: VisionIKFlow,
        testset,
        samples_per_pose: int,
        clamp_to_joint_limits: bool = False,
        desc=None
):
    poses, env_imgs, idx = testset
    solutions = []
    runtime = []
    if desc is not None:
        poses = tqdm.tqdm(poses, desc=desc)
    for i, p in enumerate(poses):
        index = i % len(idx)
        index = idx[index]
        imgs = [os.path.join(cam["cam_dir"], cam["imgs"][index]) for cam in env_imgs["Cameras"]]
        imgs = [img_transform(Image.open(i)) for i in imgs]
        imgs = torch.cat(imgs, dim=0).unsqueeze(dim=0).float().cuda()

        t0 = time_synchronized()
        _, output = ik_solver.solve_single_pose(
            p.unsqueeze(dim=0),
            imgs,
            samples_per_pose,
            clamp_to_joint_limits=clamp_to_joint_limits,
            refine_solutions=False,
        )
        t1 = time_synchronized()
        solutions.append(output.cpu().detach().numpy())
        runtime.append(t1-t0)
    return solutions, runtime


def solve_ik_classical(robot: Robot, poses, num_samples_per_pose, desc=None):
    runtime = []
    solutions = []
    rst_sc = []
    rst_c = []
    if desc is not None:
        poses = tqdm.tqdm(poses, desc=desc)
    for p in poses:
        t0 = time_synchronized()
        s = []
        cnt = 0
        while len(s) < num_samples_per_pose and cnt < 1000:
            cnt += 1
            rst = robot.inverse_kinematics_klampt(p, seed=None, positional_tolerance=1e-3, n_tries=1000)
            if rst is not None and rst.size == robot.n_dofs:
                s.append(rst)
        s = np.concatenate(s, axis=0)
        rst_sc.append(self_collision_check(robot, s).float().mean().item())
        rst_c.append(collision_check(robot, s).float().mean().item())
        t1 = time_synchronized()
        solutions.append(s)
        runtime.append(t1-t0)
    return solutions, runtime, np.mean(rst_sc)*100, np.mean(rst_c)*100


def evaluate_runtime_classical(ik_solver, poses, num_sulutions_list, desc):
    runtime = []
    num_sulutions_list = tqdm.tqdm(num_sulutions_list, desc=desc)
    for num in num_sulutions_list:
        _, runtime_classical, _, _ = solve_ik_classical(ik_solver.robot, poses[:100].detach().cpu().numpy(), num)
        runtime.append([np.mean(runtime_classical), np.var(runtime_classical)])
    return runtime


def evaluate_runtime_viik(ik_solver, testset, num_sulutions_list, desc):
    (poses, env_imgs, idx) = testset
    runtime = []
    num_sulutions_list = tqdm.tqdm(num_sulutions_list, desc=desc)
    for num in num_sulutions_list:
        _, runtime_viik = solve_all_ik(ik_solver, (poses[:150], env_imgs, idx), num, clamp_to_joint_limits=False)
        runtime.append([np.mean(runtime_viik[50:]), np.var(runtime_viik[50:])])
    return runtime


def pp_results(args: argparse.Namespace, error_stats: ErrorStats):
    text4print = (f"\n----------------------------------------\n"
                  f"> Results in {error_stats.env_name} while number of samples is {error_stats.num_samples_per_pose}\n"
                  f"\n  Average positional error:              {round(1000*error_stats.mean_l2_error_mm, 4)} mm\n"
                  f"  Average rotational error:         {round(np.rad2deg(error_stats.mean_angular_error_deg), 4)} deg\n"
                  f"  Percent joint limits exceeded: {round(error_stats.pct_joint_limits_exceeded, 4)} %\n"
                  f"  Percent self-collision:        {round(error_stats.pct_self_collision, 4)} %\n"
                  f"  IoU of self-collision:        {round(error_stats.iou_self_collision, 4)} %\n"
                  f"  Percent success of self-collision:        {round(error_stats.pct_success_self_collision, 4)} %\n"
                  f"  Percent collision:        {round(error_stats.pct_collision, 4)} %\n"
                  f"  IoU of collision:        {round(error_stats.iou_collision, 4)} %\n"
                  f"  Percent success of collision:        {round(error_stats.pct_success_collision, 4)} %\n")
    print(text4print)
    return text4print


def evaluate_model(args):
    assert (os.path.exists(args.model_file) if args.model_file else False) or args.eval_runtime_only, \
        f"Checkpoint is not found! ({args.model_file})"

    # device
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    print(f"Using device: {device}")

    robot = get_robot(args.robot_name)
    ik_solver = VisionIKFlow(robot, args.dim_dist+1, args.img_ch,
                             args.dim_img, args.img_size, dim_cond=7, dim_latent=args.dim_latent,
                             num_nodes_img=args.num_nodes_img,
                             num_nodes_value=args.num_nodes_value,
                             with_rfilm=args.with_rfilm,
                             device=device)

    # resume
    env_name_list = args.env_name_list
    if args.model_file:
        print("Resume from: ", args.model_file)
        checkpoint = torch.load(args.model_file, map_location='cpu')
        env_name_list = (checkpoint['env_name_list'] if "env_name_list" in checkpoint.keys() else args.env_name_list)
        print(f"Epochs: {checkpoint['epoch']}")
        print(f"Global steps: {checkpoint['global_step']}")
        print(f"Best Loss: {checkpoint['best_loss']}")
        print(f"Env Name List: {env_name_list}")
        ik_solver.load_state_dict(checkpoint['state_dict'])

    # Evaluate
    data_module = IKLitDataset(robot.name, env_name_list, 1, 0)
    poses = torch.tensor(data_module.poses['test'][:args.testset_size])
    print(f"evaluate on {len(poses)} ik problems!")
    rst_file = f"./results_{env_name_list[0]}_{len(env_name_list)}env.txt"
    print("Results are recoded in ", rst_file)
    text4print = (f"\n--------------------------------------------------------------------------------\n"
                  f"> Results for {args.model_file}\n\n")

    # Test set
    o_data_sc = ((torch.tensor(data_module.samples['test'][:, 7]) + 1) / 2).bool()
    data_sc = o_data_sc.float().mean().item() * 100
    data_c = dict()
    data_c_a = dict()
    for i, env_name in enumerate(env_name_list):
        o_data_c = ((torch.tensor(data_module.samples['test'][:, 8 + i]) + 1) / 2).bool()
        data_c[env_name] = o_data_c.float().mean().item() * 100
        data_c_a[env_name] = torch.logical_or(o_data_c, o_data_sc).float().mean().item() * 100
    text4print += (f"Self-collision Rate: {data_sc:.2f}%;\n"
                  f"Mean of collision with env rate: {np.mean([v for _, v in data_c.items()]):.2f};\n"
                  f"Mean of collision rate: {np.mean([v for _, v in data_c_a.items()]):.2f};\n\n")
    for k, v in data_c.items():
        text4print += f"Collision with env Rate in '{k}': {v:.2f}%;\n"
    text4print += "\n"
    for k, v in data_c_a.items():
        text4print += f"Collision Rate in '{k}': {v:.2f}%;\n"
    text4print += "\n"
    print(text4print)
    with open(rst_file, 'a') as f:
        f.write(text4print)

    num_sulutions_list = [10 * (i + 1) for i in range(int(100 / 10))] + [100*(i+1) for i in range(int(5000/100))]
    # num_sulutions_list.reverse()
    for i, env_name in enumerate(env_name_list):
        # Load the environment
        robot = get_robot(args.robot_name)
        ik_solver.robot = robot
        obs_mesh_dir = os.path.join(args.obs_mesh_root, env_name, "mesh")
        obstacles = [os.path.join(obs_mesh_dir, i) for i in os.listdir(obs_mesh_dir)]
        set_environment(robot, obstacles)
        print(f"obstacles in '{env_name}' ({i+1}/{len(env_name_list)}): {obstacles}")

        env_imgs = data_module.env_imgs[env_name]
        idx = env_imgs["Cameras"][0]["idx_list"][:4882]

        # Test runtime of viik
        if not i and args.eval_runtime:
            runtime_viik = evaluate_runtime_viik(ik_solver, (poses, env_imgs, idx), num_sulutions_list,
                                                 desc=f"[Test runtime of ViIK']: ")
            text4print = f"List of solution numbers: {num_sulutions_list} \n"
            text4print += f"Runtime of ViIK: {runtime_viik} \n\n"
            print(text4print)
            with open(rst_file, 'a') as f:
                f.write(text4print)
        # Test runtime of classical
        runtime_classical = []
        if args.eval_runtime:
            runtime_classical = evaluate_runtime_classical(ik_solver, poses, num_sulutions_list,
                                                           desc=f"[Test runtime of the classical method in '{env_name}']: ")
            text4print = (f"\n----------------------------------------\n"
                          f"> Results in {env_name}\n"
                          f"  Runtime (Classical):        {runtime_classical}\n")
            print(text4print)
            with open(rst_file, 'a') as f:
                f.write(text4print)

        # evaluate runtime only, so jump the rest of test
        if args.eval_runtime_only and args.eval_runtime:
            continue

        if len(args.num_samples_per_pose) == 1:
            testset = (poses, env_imgs, idx)
        else:
            testset = (poses[:10000], env_imgs, idx)

        for num_samples in args.num_samples_per_pose:
            print(f"Number of samples: {num_samples}; clamp_to_joint_limits: {args.clamp_to_joint_limits}")
            print()
            # Solve IK
            solutions, _ = solve_all_ik(ik_solver, testset, num_samples,
                                        clamp_to_joint_limits=args.clamp_to_joint_limits,
                                        desc=f"[Solve all IK problems in '{env_name}']: ")

            # Calculate all metrics
            rst = [env_name, num_samples]
            rst_error, s_non_exceeded = calculate_l2_angular_jlimt(ik_solver.robot, testset[0], solutions)
            solutions_filtered = []
            rst_sc = []
            rst_c = []
            for s in s_non_exceeded:
                if len(s) == 0:
                    continue
                solutions_filtered.append(s[:, :robot.n_dofs])
                rst_sc.append(s[:, 7])
                rst_c.append(s[:, 8])
            rst_sc = calculate_collision_with_self(ik_solver.robot, solutions_filtered[:10000], rst_sc[:10000],
                                                   f"[Calculate Self-collision Rate in '{env_name}']")
            rst_c = calculate_collision_with_env(ik_solver.robot, solutions_filtered[:10000], rst_c[:10000],
                                                 f"[Calculate Collision Rate in '{env_name}']")

            # Summary results
            for _, v in rst_error.items():
                rst.append(v)
            for _, v in rst_sc.items():
                rst.append(v)
            for _, v in rst_c.items():
                rst.append(v)
            error_stats = ErrorStats(*rst)
            text4print = pp_results(args, error_stats)
            with open(rst_file, 'a') as f:
                f.write(text4print)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="evaluate.py - evaluates IK models")
    parser.add_argument("--testset_size", default=150000, type=int)
    parser.add_argument("--eval_runtime", default=False, type=bool)
    parser.add_argument("--eval_runtime_only", default=False, type=bool)
    parser.add_argument("--robot_name", type=str, default="panda")
    parser.add_argument("--env_name_list", type=list,
                        default=["env1", "env2"])#, "env3", "env4", "env5"])#, "env6", "env7", "env8", "env9", "env10",])  #
    parser.add_argument("--num_samples_per_pose", default=[1000], type=list)
    parser.add_argument("--model_file", type=str)
    parser.add_argument("--obs_mesh_root", type=str, default="data/env/")
    parser.add_argument("--device", type=str, default='cuda')

    # Model parameters
    parser.add_argument("--num_nodes_img", type=int, default=6)
    parser.add_argument("--num_nodes_value", type=int, default=36)
    parser.add_argument("--dim_dist", type=int, default=9)
    parser.add_argument("--img_ch", type=int, default=4)
    parser.add_argument("--dim_img", type=int, default=3)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--dim_latent", type=int, default=128)
    parser.add_argument("--with_rfilm", type=bool, default=False)
    parser.add_argument("--clamp_to_joint_limits", type=bool, default=False)
    args = parser.parse_args()

    # Build IKFlowSolver and set weights
    print("\n-------------")
    print(f"Evaluating model '{args.model_file}'")
    evaluate_model(args)
