from mani_skill.envs.tasks.empty_env import EmptyEnv
from mani_skill.envs.sapien_env import BaseEnv

from typing import Dict

from mani_skill.agents.robots.fetch import FETCH_WHEELS_COLLISION_BIT
from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.registration import register_env
from mani_skill.utils import common, sapien_utils
import sapien
from mani_skill.sensors.camera import CameraConfig
import sys
import os
import json
import numpy as np
import gymnasium as gym
import torch
import mani_skill.envs
from tqdm import tqdm
from mani_skill.utils.wrappers import RecordEpisode
from transforms3d import quaternions
import random
import string
from robocasa_scene_builder import EmptyRoomFromRobocasa
import argparse
def solve_by_coords(env: PickCubeEnv, target: Actor, goal_pose: sapien.Pose, vis=False):
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
    obb = get_actor_obb(target)

    approaching = np.array([0, 0, -1])
    # get transformation matrix of the tcp pose, is default batched and on torch
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    # we can build a simple grasp pose using this information for Panda
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, target.pose.sp.p)
    agent_z_reach_pose = env.agent.get_pose()
    agent_z_reach_pose[2] = grasp_pose[2]

    # -------------------------------------------------------------------------- #
    # Reach
    # -------------------------------------------------------------------------- #

    # reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
    planner.move_to_pose_with_screw(agent_z_reach_pose)

    # -------------------------------------------------------------------------- #
    # Grasp
    # -------------------------------------------------------------------------- #
    planner.move_to_pose_with_screw(grasp_pose)

    # reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
    planner.close_gripper()

    # -------------------------------------------------------------------------- #
    # Move to goal pose
    # -------------------------------------------------------------------------- #
    grasp_pose[:2] = goal_pose[:2]
    res = planner.move_to_pose_with_screw(grasp_pose)
    res = planner.move_to_pose_with_screw(goal_pose)

    planner.close()
    return res