from mani_skill.envs.tasks.empty_env import EmptyEnv
from mani_skill.envs.sapien_env import BaseEnv

from typing import Dict
import time
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
import argparse
sys.path.append(".")
from dsynth.scenes.darkstore_env import get_arena_data, DarkstoreEnv
from dsynth.envs.pick_to_cart import PickToCart
from dsynth.motionplanning.solvers.pick_box import solve, solve_smooth
#lubin/darkstore_synthesizer/scenes/myscene_5_5.json

NUM_PROC = 4

ENV_NAME = 'DarkstoreEnv'
# cnt_good = 0
from multiprocessing import Pool


IDLE_STEPS_MAX = 90


def routine(j):
    json_file_path = "myscene_2_2.json"
    json_file_path = "./gen_scenes/myscene_2_2" + str(j)+ ".json"
    # json_file_path = "darkstore_synthesizer/scenes/myscene_5_5.json"
    assets_dir = "models"
    style_id = 0
    mapping_file = 'models/connect.json'
    with open(json_file_path, "r") as f: # big_scene , one_shelf_many_milk_scene , customize
        data = json.load(f)
    for el in data['meta']:
        print(el)

    n = data['meta']['n']
    m = data['meta']['m']
    arena_data = get_arena_data(x_cells=n, y_cells=m, height=4)

    # env = PickToCart(scene_json = json_file_path, assets_dir = "models", style_ids = [0], robot_uids = "panda", render_mode="rgb_array", control_mode="pd_joint_pos", **arena_data)
    env = gym.make('PickToCart', 
            robot_uids='panda_wristcam', 
            scene_json = json_file_path,
            assets_dir = assets_dir,
            mapping_file = mapping_file,
            style_ids = [style_id], 
            num_envs=1, 
            viewer_camera_configs={'shader_pack': 'default'}, 
            control_mode="pd_joint_pos",
            render_mode="rgb_array", 
            obs_mode="sensor_data",
            enable_shadow=True, 
            **arena_data)

    # new_traj_name = str(cnt_good)
    
    env = RecordEpisode(
        env,
        output_dir=f"./arm_checker_out/videos_style_{style_id}_motionplanning_{j}",
        trajectory_name=str(j), save_video=False,
        source_type="motionplanning",
        source_desc="AI360 winter school",
        video_fps=30,
        save_on_reset=True
    )

    obs, _ = env.reset()
    viewer = env.render()
    if isinstance(viewer, sapien.utils.Viewer):
        viewer.paused = False
    env.render()

    # target = env.actors['objects']['milk'][0]

    target = env.actors["objects"]["milk_1_1_0"][0]['actor']
    goal_pose = env.target_volume.pose * sapien.Pose([0, 0, 0.6])

    for i in range(1):
    # while True:
        res = solve_smooth(env, target, goal_pose, vis=False)

        # do-nothin action for pd_joint_pos_control
        action_zero = env.agent.robot.qpos.numpy()[0, :-1]

        for _ in range(IDLE_STEPS_MAX):
            obs, reward, terminated, truncated, info = env.step(action_zero)
            env.render()
            if env.evaluate()['success']:
                break

    result = env.evaluate()
    print(result)
    # is_success = False
    # if result['success']:
    #     is_success = True
        # cnt_good += 1
    viewer.paused = True
    env.render()
    env.close()
    return {"videos_style_{style_id}_motionplanning_{j}": result['success'].item()}
    # print(cnt_good)
    
# for j in tqdm(range(0, 10)):
#     print(routine(j))
    
    
with Pool(NUM_PROC) as p:
    # print(p.map(routine, range(0, 10)))
    r = list(tqdm(p.imap(routine, range(0, 200)), total=200))


with open('./arm_checker_out/successes.json', 'w') as f:
    json.dump(r, f)
    
