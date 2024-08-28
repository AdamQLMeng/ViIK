from time import sleep

import numpy as np
from klampt import vis, WorldModel
from klampt.model import coordinates, trajectory
from jrl.robot import Robot
from klampt.model.collide import WorldCollider

from utils.evaluation_utils import collision_check


def set_environment(robot: Robot, obstacles):
    world = robot.klampt_world_model
    world.loadTerrain(r"data/terrains/block.off")
    assert world.numTerrains(), "Terrain hasn't been loaded correctly."
    robot._ignored_collision_pairs_formatted += [
        (robot._klampt_robot.link(robot._base_link), world.terrain(i))
        for i in range(world.numTerrains())]
    for obs in obstacles:
        world.loadRigidObject(obs)
    print("ignored_collision_pairs_formatted: ", len(robot._ignored_collision_pairs_formatted))
    for i in robot._ignored_collision_pairs_formatted:
        print(i)
    robot._klampt_collision_checker = WorldCollider(
        robot._klampt_world_model,
        ignore=robot._ignored_collision_pairs_formatted
    )
    print("reinitialize the world collider", robot.klampt_world_model.numRigidObjects())
    return world


def show_world(world: WorldModel, window_title):
    vis.init()
    vis.add("world", world)
    vis.add("coordinates", coordinates.manager())
    vis.setWindowTitle(window_title)
    vis.show()
    while vis.shown():
        vis.lock()
        vis.unlock()
        sleep(1 / 30)  # note: don't put sleep inside the lock()
    vis.kill()


def _init_klampt_vis(robot: Robot, window_title: str, show_collision_capsules: bool = True):
    vis.init()

    background_color = (0.8, 0.8, 0.9, 0.7)
    vis.setBackgroundColor(background_color[0], background_color[1], background_color[2], background_color[3])
    vis.add("world", robot.klampt_world_model)
    vis.add("coordinates", coordinates.manager())
    print(vis.listItems())
    for i in range(robot.klampt_world_model.numRigidObjects()):
        vis.add(f"obs{i}", robot.klampt_world_model.rigidObject(i))
        vis.setColor(f"obs{i}", 0.85, 0.85, 1, 1)
    vis.setWindowTitle(window_title)
    vis.show()


def show_ik_solutions(robot: Robot, solutions, show_collision=False, title="IK redundancy"):
    _init_klampt_vis(robot, title)
    qs = robot._x_to_qs(np.array(np.remainder(np.array(solutions) + np.pi, 2 * np.pi) - np.pi))
    collision = collision_check(robot, solutions).tolist()
    print(sum(collision), type(solutions), len(qs))
    for i, (q, rst) in enumerate(zip(qs, collision)):
        print("add config", i)
        vis.add(f"robot_{i}", q)
        if rst and show_collision:
            print("collision!")
            vis.setColor(f"robot_{i}", 0.7, 0.1, 0.1, 1.0)
        else:
            vis.setColor(f"robot_{i}", 0.7, 0.7, 0.7, 1.0)

    while vis.shown():
        vis.lock()
        vis.unlock()
        sleep(1 / 30)  # note: don't put sleep inside the lock()
    # vis.kill()


def ik_solutions_classical(robot: Robot, target_pose, n_qs=15):
    """Fixed end pose with n=n_qs different solutions"""

    xs = []
    for i in range(n_qs*3):
        if len(xs) == n_qs:
            break
        x_new = robot.inverse_kinematics_klampt(target_pose)
        if x_new is None:
            continue
        fk_old = robot.forward_kinematics_klampt(x_new)
        x_new = np.remainder(x_new + np.pi, 2 * np.pi) - np.pi
        assert np.linalg.norm((robot.forward_kinematics_klampt(x_new)[0] - fk_old[0])[0:3]) < 1e-3
        is_duplicate = False
        # for x in xs:
        #     if np.linalg.norm(x_new - x) < 0.01:
        #         is_duplicate = True
        if not is_duplicate:
            xs.append(x_new[0])

    return xs


def show_path(robot: Robot, path):
    _init_klampt_vis(robot, f"{robot.formal_robot_name} - IK redundancy")
    print(len(path), "configs in the path")

    qs = robot._x_to_qs(np.array(path))
    collision = []
    for i, (q, x) in enumerate(zip(qs, path)):
        print("add config", i)
        vis.add(f"robot_{i}", q)
        robot.set_klampt_robot_config(np.array(x))
        rst = False
        for j in range(robot.klampt_world_model.numRigidObjects()):
            pairs = robot._klampt_collision_checker.robotObjectCollisions(robot._klampt_robot, j)
            for link, obs in pairs:
                # print(f"Collision occurs between {link.getName()} and {obs.getName()}!")
                rst = True
        collision.append(rst)

        if not rst:
            vis.setColor(f"robot_{i}", 0.7, 0.7, 0.7, 1.0)

    while vis.shown():
        vis.lock()
        vis.unlock()
        sleep(1 / 30)  # note: don't put sleep inside the lock()
    vis.kill()
