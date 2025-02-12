import sys 
sys.path.append('.')
from dsynth.scenes.darkstore_env import DarkstoreEnv, get_arena_data
from dsynth.envs.pick_to_cart import PickToCart
from typing import Dict

import sapien
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

def parse_args():
    parser = argparse.ArgumentParser(
        description="Использование: python script.py <путь_к_JSON_файлу> <путь_к_assets> <style id (0-11)> [mapping_file]"
    )
    parser.add_argument("json_file_path", help="Путь к JSON файлу")
    parser.add_argument("assets_dir", help="Путь к assets")
    parser.add_argument("--style_id", type=int, default=0, help="Style id (0-11)")
    parser.add_argument("--mapping_file", default=None, help="Путь к mapping_file (опционально)")
    # parser.add_argument("--shader", default=None, help="Путь к mapping_file (опционально)")
    parser.add_argument('--shader',
                        default='default',
                        const='default',
                        nargs='?',
                        choices=['rt', 'rt-fast', 'default', 'minimal'],)
    parser.add_argument('--gui',
                        action='store_true',
                        default=False)

    args = parser.parse_args()

    return args

def main(args):

    json_file_path = args.json_file_path
    assets_dir = args.assets_dir
    style_id = args.style_id
    mapping_file = args.mapping_file
    gui = args.gui

    with open(json_file_path, "r") as f: # big_scene , one_shelf_many_milk_scene , customize
        data = json.load(f)

    n = data['meta']['n']
    m = data['meta']['m']
    arena_data = get_arena_data(x_cells=n, y_cells=m, height=4)

    env = gym.make('PickToCart', 
                   robot_uids='panda', 
                   scene_json = json_file_path,
                   assets_dir = assets_dir,
                   mapping_file = mapping_file,
                   style_ids = [style_id], 
                   num_envs=1, 
                   viewer_camera_configs={'shader_pack': args.shader}, 
                   render_mode="human" if gui else "rgb_array", 
                   enable_shadow=True, 
                   **arena_data)

    if not gui:
        env = RecordEpisode(
            env,
            f"./videos_{n}_{m}_style{style_id}", # the directory to save replay videos and trajectories to
            # on GPU sim we record intervals, not by single episodes as there are multiple envs
            # each 100 steps a new video is saved
            max_steps_per_video=100
        )

    # step through the environment with random actions
    obs, _ = env.reset()


    viewer = env.render()
    if isinstance(viewer, sapien.utils.Viewer):
        viewer.paused = False
    env.render()


    for i in tqdm(range(100000)):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(torch.zeros_like(torch.from_numpy(action)))

        env.render()
        # env.render_human() # will render with a window if possible
    env.close()


if __name__ == '__main__':
    args = parse_args()
    main(args)