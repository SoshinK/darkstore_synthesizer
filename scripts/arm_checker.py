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
import argparse
sys.path.append(".")
from dsynth.scenes.mv_panda_arm import solve_by_coords
from dsynth.scenes.darkstore_env import get_arena_data, DarkstoreEnv
from dsynth.envs.pick_to_cart import PickToCart
#lubin/darkstore_synthesizer/scenes/myscene_5_5.json
ENV_NAME = 'DarkstoreEnv'
json_file_path = "scenes/myscene_2_2.json"
# json_file_path = "darkstore_synthesizer/scenes/myscene_5_5.json"
assets_dir = "models"
style_id = 0
mapping_file = None
with open(json_file_path, "r") as f: # big_scene , one_shelf_many_milk_scene , customize
    data = json.load(f)
for el in data['meta']:
    print(el)

n = data['meta']['n']
m = data['meta']['m']
arena_data = get_arena_data(x_cells=n, y_cells=m, height=4)

env = PickToCart(scene_json = json_file_path, assets_dir = "models", style_ids = [0], robot_uids = "panda", render_mode="rgb_array", control_mode="pd_joint_pos", **arena_data)
env = RecordEpisode(
        env,
        f"./videos__style{style_id}", # the directory to save replay videos and trajectories to
        # on GPU sim we record intervals, not by single episodes as there are multiple envs
        # each 100 steps a new video is saved
        max_steps_per_video=100
    )

obs, _ = env.reset()
viewer = env.render()
if isinstance(viewer, sapien.utils.Viewer):
    viewer.paused = False
env.render()

# target = env.actors['objects']['milk'][0]

target = env.actors["objects"]["milk_1_1_0"][0]['actor']
goal_pose = sapien.Pose([0.5, 0, 0.3])

for i in tqdm(range(100)):
    # if target is not None and goal_pose is not None:
    #     action = solve_by_coords(env, target, goal_pose)
    # else:
    #     action = env.action_space.sample()
    action = env.action_space.sample()

    obs, reward, terminated, truncated, info = env.step(torch.tensor(action, dtype=torch.float32))
    env.render()

env.close()