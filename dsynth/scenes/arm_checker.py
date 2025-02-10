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


json_file_path = "myscene_4_3.json"
assets_dir = "darkstore_synthesizer/models"
style_id = 0
mapping_file = None

with open(json_file_path, "r") as f: # big_scene , one_shelf_many_milk_scene , customize
    data = json.load(f)

n = data['meta']['n']
m = data['meta']['m']
arena_data = get_arena_data(x_cells=n, y_cells=m, height=4)

env = gym.make(ENV_NAME, robot_uids='fetch', style_ids = [style_id], num_envs=1, render_mode="rgb_array", enable_shadow=True, **arena_data)
