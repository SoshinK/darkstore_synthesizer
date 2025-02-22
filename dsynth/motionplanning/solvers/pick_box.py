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
from dsynth.scenes.darkstore_env import DarkstoreEnv
from mani_skill.utils.structs.actor import Actor
from mani_skill.examples.motionplanning.panda.motionplanner import \
    PandaArmMotionPlanningSolver
from mani_skill.examples.motionplanning.panda.utils import (
    compute_grasp_info_by_obb, get_actor_obb)
import torch 
import trimesh
from dsynth.scenes.robocasaroom import RoomFromRobocasa
from dsynth.motionplanning.utils import compute_box_grasp_thin_side_info
from torch import Tensor

def made_mv_seq(planner, reach_pose, cur_pose, inv: bool = False):
    res = None
    if (inv is False):
        np_cur_pose = np.array(cur_pose.p, dtype = np.float32)
        z_reach_pose = sapien.Pose(
            p=np.array([np_cur_pose[0], np_cur_pose[1], reach_pose.p[2]], dtype=np.float32),
            q=reach_pose.q
        )
        y_reach_pose = sapien.Pose(
            p=np.array([reach_pose.p[0], np_cur_pose[1], reach_pose.p[2]], dtype=np.float32),
            q=reach_pose.q
        )
        x_reach_pose = sapien.Pose(
            p=np.array([reach_pose.p[0], reach_pose.p[1], reach_pose.p[2]], dtype=np.float32),
            q=reach_pose.q
        )

        res = planner.move_to_pose_with_screw(z_reach_pose)
        res = planner.move_to_pose_with_screw(y_reach_pose)
        res = planner.move_to_pose_with_screw(x_reach_pose)
    else:
        np_reach_pose = np.array(reach_pose.p, dtype = np.float32)
        np_reach_pose_q = np.array(reach_pose.q, dtype = np.float32)
        z_reach_pose = sapien.Pose(
            p=np.array([cur_pose.p[0], np_reach_pose[1], cur_pose.p[2]], dtype=np.float32),
            q=cur_pose.q
        )
        y_reach_pose = sapien.Pose(
            p=np.array([np_reach_pose[0], np_reach_pose[1], cur_pose.p[2]], dtype=np.float32),
            q=cur_pose.q
        )
        x_reach_pose = sapien.Pose(
            p=np.array([np_reach_pose[0], np_reach_pose[1], np_reach_pose[2]], dtype=np.float32),
            q=cur_pose.q
        )
        res = planner.move_to_pose_with_screw(z_reach_pose)
        res = planner.move_to_pose_with_screw(y_reach_pose)
        res = planner.move_to_pose_with_screw(x_reach_pose)
    return res
    

def solve(env: DarkstoreEnv, target: Actor, goal_pose: sapien.Pose, seed=None, debug=False, vis=False):
    env.reset(seed=seed)
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
        obb = get_actor_obb(target, vis=False)  # Should now work correctly
    else:
        print("Error: Target has no collision meshes.")

    
    # approaching = np.array([0, 1, 0])
    # get transformation matrix of the tcp pose, is default batched and on torch
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    target_approaching = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 2].cpu().numpy()
    ee_direction = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 2].cpu().numpy()
    tcp_center = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 3].cpu().numpy()

    goal_closing = goal_pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    goal_approaching = goal_pose.to_transformation_matrix()[0, :3, 2].cpu().numpy()
    goal_center = goal_pose.to_transformation_matrix()[0, :3, 3].cpu().numpy() - np.array([0.1, 0., 0.3])

    init_pose = env.agent.build_grasp_pose(target_approaching, target_closing, tcp_center)
    goal_pose = env.agent.build_grasp_pose(-goal_approaching, -goal_closing, goal_center)


    # we can build a simple grasp pose using this information for Panda
    agent_pose = env.agent.robot.get_pose()
    grasp_info = compute_box_grasp_thin_side_info(
        obb,
        target_closing=target_closing,
        ee_direction=ee_direction,
        depth=FINGER_LENGTH,
    )
    height = obb.primitive.extents[2]
    closing, center, approaching = grasp_info["closing"], grasp_info["center"], grasp_info["approaching"]
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, target.pose.sp.p + np.array([0., 0., height / 2]))

    # -------------------------------------------------------------------------- #
    # Reach
    # -------------------------------------------------------------------------- #
    print(agent_pose)
    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.1])
    res = made_mv_seq(planner, reach_pose, init_pose)

    # -------------------------------------------------------------------------- #
    # Grasp
    # -------------------------------------------------------------------------- #

    # reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
    res = planner.move_to_pose_with_screw(grasp_pose)
    res = planner.close_gripper()

    # -------------------------------------------------------------------------- #
    # Lift
    # -------------------------------------------------------------------------- #

    lift_pose = grasp_pose * sapien.Pose([0.02, 0., 0.])
    res = planner.move_to_pose_with_screw(lift_pose)

    # -------------------------------------------------------------------------- #
    # Return 
    # -------------------------------------------------------------------------- #
    res = made_mv_seq(planner, init_pose, reach_pose, inv=True)
    # planner.close()
    # return res

    # -------------------------------------------------------------------------- #
    # Move to goal pose
    # -------------------------------------------------------------------------- #

    res = move_to_goal_pose(planner, init_pose, goal_pose)
    
    planner.close()
    return res

def move_to_goal_pose(planner: PandaArmMotionPlanningSolver, init_pose, goal_pose: sapien.Pose):
    # res = made_mv_seq(planner, init_pose, goal_pose)
    res = planner.move_to_pose_with_screw(goal_pose)
    res = planner.open_gripper()
    return res

def solve_smooth(env: DarkstoreEnv, target: Actor, goal_pose: sapien.Pose, seed=None, debug=False, vis=False):
    env.reset(seed=seed)
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
        obb = get_actor_obb(target, vis=False)  # Should now work correctly
    else:
        print("Error: Target has no collision meshes.")

    
    # approaching = np.array([0, 1, 0])
    # get transformation matrix of the tcp pose, is default batched and on torch
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    target_approaching = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 2].cpu().numpy()
    ee_direction = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 2].cpu().numpy()
    tcp_center = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 3].cpu().numpy()

    goal_closing = goal_pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    goal_approaching = goal_pose.to_transformation_matrix()[0, :3, 2].cpu().numpy()

    pre_goal_center = goal_pose.to_transformation_matrix()[0, :3, 3].cpu().numpy() - np.array([0.1, -0.2, 0.4])
    goal_center = goal_pose.to_transformation_matrix()[0, :3, 3].cpu().numpy() - np.array([0.1, 0., 0.3])

    init_pose = env.agent.build_grasp_pose(target_approaching, target_closing, tcp_center)
    pre_goal_pose = env.agent.build_grasp_pose(-goal_approaching, -goal_closing, pre_goal_center)
    goal_pose = env.agent.build_grasp_pose(-goal_approaching, -goal_closing, goal_center)


    # we can build a simple grasp pose using this information for Panda
    agent_pose = env.agent.robot.get_pose()
    grasp_info = compute_box_grasp_thin_side_info(
        obb,
        target_closing=target_closing,
        ee_direction=ee_direction,
        depth=FINGER_LENGTH,
    )
    height = obb.primitive.extents[2]
    closing, center, approaching = grasp_info["closing"], grasp_info["center"], grasp_info["approaching"]
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, target.pose.sp.p + np.array([0., 0., height / 2]))

    # -------------------------------------------------------------------------- #
    # Reach
    # -------------------------------------------------------------------------- #
    print(agent_pose)
    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.1])
    res = planner.move_to_pose_with_screw(reach_pose)

    # -------------------------------------------------------------------------- #
    # Grasp
    # -------------------------------------------------------------------------- #

    # reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
    res = planner.move_to_pose_with_screw(grasp_pose)
    res = planner.close_gripper()

    # -------------------------------------------------------------------------- #
    # Lift
    # -------------------------------------------------------------------------- #

    lift_pose = grasp_pose * sapien.Pose([0.02, 0., 0.])
    res = planner.move_to_pose_with_screw(lift_pose)

    # -------------------------------------------------------------------------- #
    # Return 
    # -------------------------------------------------------------------------- #
    # res = planner.move_to_pose_with_screw(init_pose)
    res = planner.move_to_pose_with_screw(pre_goal_pose)
    
    # planner.close()
    # return res

    # -------------------------------------------------------------------------- #
    # Move to goal pose
    # -------------------------------------------------------------------------- #

    res = planner.move_to_pose_with_screw(goal_pose)
    res = planner.open_gripper()
    planner.close()
    return res
