from mani_skill.envs.tasks.empty_env import EmptyEnv
from mani_skill.envs.sapien_env import BaseEnv

from mani_skill.agents.robots.fetch import FETCH_WHEELS_COLLISION_BIT
from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.registration import register_env
from mani_skill.utils import common, sapien_utils
import sapien
from mani_skill.sensors.camera import CameraConfig
import numpy as np
from mani_skill.utils.wrappers import RecordEpisode
from transforms3d import quaternions
from .robocasaroom import RoomFromRobocasa
from .darkstore_env import DarkstoreEnv
from mani_skill.utils.structs.actor import Actor
from mani_skill.examples.motionplanning.panda.motionplanner import \
    PandaArmMotionPlanningSolver
from mani_skill.examples.motionplanning.panda.utils import (
    compute_grasp_info_by_obb, get_actor_obb)
import torch 

def solve_by_coords(env: DarkstoreEnv, target: Actor, goal_pose: sapien.Pose, seed=None, debug=False, vis=False):
    planner = PandaArmMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
    )

    FINGER_LENGTH = 0.025
    env = env.unwrapped

    # retrieves the object oriented bounding box (trimesh box object)
    if target.get_collision_meshes():  # Ensure it has collision meshes
        obb = get_actor_obb(target)  # Should now work correctly
    else:
        print("Error: Target has no collision meshes.")

    approaching = np.array([0, 0, -1])
    # get transformation matrix of the tcp pose, is default batched and on torch
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    # we can build a simple grasp pose using this information for Panda
    agent_pose = env.agent.robot.get_pose()
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, target.pose.sp.p)

    # -------------------------------------------------------------------------- #
    # Reach
    # -------------------------------------------------------------------------- #

    agent_p_np = np.array(agent_pose.p[0], dtype=np.float32)
    grasp_p_np = np.array(grasp_pose.p, dtype=np.float32)

    agent_q_np = np.array(agent_pose.q[0], dtype=np.float32)

    z_reach_pose = sapien.Pose(
        p=np.array([agent_p_np[0], agent_p_np[1], grasp_p_np[2]], dtype=np.float32),
        q=agent_q_np
    )

    planner.move_to_pose_with_screw(z_reach_pose)

    # -------------------------------------------------------------------------- #
    # Grasp
    # -------------------------------------------------------------------------- #

    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])

    planner.move_to_pose_with_screw(reach_pose)
    planner.close_gripper()

    # -------------------------------------------------------------------------- #
    # Move to goal pose
    # -------------------------------------------------------------------------- #


    z_goal_pose = sapien.Pose(
        p=[goal_pose.p[0], goal_pose.p[1], reach_pose.p[2]],  
        q=goal_pose.q  
    )

    planner.move_to_pose_with_screw(z_goal_pose)
    res = planner.move_to_pose_with_screw(goal_pose)

    planner.close()
    return res