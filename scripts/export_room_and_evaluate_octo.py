import sys 
sys.path.append('.')
from dsynth.scenes.darkstore_env import DarkstoreEnv, get_arena_data
from dsynth.envs.pick_to_cart import PickToCart
from typing import Dict
from octo.model.octo_model_pt import OctoModelPt
from mani_skill.sensors.camera import CameraConfig

import sapien
import sys
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
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
        description="Использование: python script.py <путь_к_папке_с_TEST_JSONS> <путь_к_assets> [style id (0-11)] [mapping_file] [shader] [gui] [instruction] [device]"
    )
    parser.add_argument("json_file_path", help="Путь к папке с JSON файлами для теста")
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
    parser.add_argument('--instruction',
                        default='pick a milk from the shelf and put it on the cart')
    parser.add_argument('--device',
                        default="cuda:0")

    args = parser.parse_args()

    return args

def main(args):

    json_folder_path = args.json_file_path
    assets_dir = args.assets_dir
    style_id = args.style_id
    mapping_file = args.mapping_file
    gui = args.gui
    language_instruction = args.instruction
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")



    cnt_envs = 0
    cnt_success = 0 
    
    # CHANGE model = ... Kostya FineTuned model from jax !!!!!
    # model = OctoModelPt.load_pretrained("hf://rail-berkeley/octo-small-1.5")['octo_model']
    # model = OctoModelPt.load_pretrained('../octo-pytorch//finetune_darkstore/octo_finetune/experiment_20250214_010238/')['octo_model']
    # model = OctoModelPt.load_pretrained('../octo-pytorch/finetune_darkstore/octo_finetune/finetune_darkstore_20250214_064911/')['octo_model']
    model = OctoModelPt.load_pretrained('../octo-pytorch/finetune_darkstore/octo_finetune/finetune_darkstore_50k_iters_20250214_201643/')['octo_model']
    stats = model.dataset_statistics["action"]
    model.to(device)

    MAX_ENVS = 50
    
    def change_gripper_value(action):
        action[:, -1] = (action[:, -1] - 0.5) * 2
        return action
    
    for n_env, file_name in enumerate(os.listdir(json_folder_path)):
        if n_env > MAX_ENVS:
            break
        json_file_path = os.path.join(json_folder_path, file_name)
        if os.path.isfile(json_file_path) and file_name.endswith(".json"):  
            
            with open(json_file_path, "r") as f:
                data = json.load(f)

            n = data['meta']['n']
            m = data['meta']['m']
            arena_data = get_arena_data(x_cells=n, y_cells=m, height=4)

            env = gym.make('PickToCart', 
                        robot_uids='panda_wristcam', 
                        scene_json = json_file_path,
                        assets_dir = assets_dir,
                        mapping_file = mapping_file,
                        style_ids = [style_id], 
                        num_envs=1, 
                        # extra_camera_configs=camera_configs,
                        viewer_camera_configs={'shader_pack': args.shader}, 
                        render_mode="human" if gui else "rgb_array", 
                        enable_shadow=True,
                        control_mode="pd_ee_delta_pose",
                        obs_mode='rgbd',
                        **arena_data)

            if not gui:
                env = RecordEpisode(
                    env,
                    output_dir=f"./octo_videos_50k_iters/{n_env}",
                    trajectory_name=str(n_env), 
                    save_video=True,
                    source_desc="Octo Model",
                    video_fps=30,
                    save_on_reset=True
                )

            obs, _ = env.reset()
            
            rgb_image_primary = torch.tensor(obs["sensor_data"]["base_camera"]["rgb"], device=device).permute(0, 3, 1, 2)
            obs_primary = torch.zeros((1, 2, 3, 256, 256), dtype=rgb_image_primary.dtype, device=device)
            obs_primary[:, 1] = rgb_image_primary
            
            rgb_image_wrist = torch.tensor(obs["sensor_data"]["hand_camera"]["rgb"], device=device).permute(0, 3, 1, 2)
            obs_wrist = torch.zeros((1, 2, 3, 128, 128), dtype=rgb_image_wrist.dtype, device=device)
            obs_wrist[:, 1] = rgb_image_wrist
            
            observation = {'image_primary': obs_primary, 'image_wrist': obs_wrist}

            viewer = env.render()
            if isinstance(viewer, sapien.utils.Viewer):
                viewer.paused = False
            env.render()
            

            task = model.create_tasks(texts=[language_instruction], device=device)
            timestep_pad_mask = torch.tensor([False, True], dtype=torch.bool).to(device)
            for i in tqdm(range(400)):

                obs = observation
                
                # dummy_action = torch.zeros((1, 8), dtype=torch.float32, device=device)
                
                action = model.sample_actions(
                    obs,
                    task, 
                    timestep_pad_mask = timestep_pad_mask,
                    unnormalization_statistics=model.dataset_statistics["action"], 
                    generator=torch.Generator(device).manual_seed(0),
                )
                action = action[:, 0, :]
                # print(f"Action: {action}")
                action = change_gripper_value(action)
                obs, reward, terminated, truncated, info = env.step(action)
                
                if (env.evaluate()['success']):
                    break
                
                rgb_image_primary = torch.tensor(obs["sensor_data"]["base_camera"]["rgb"], device=device).permute(0, 3, 1, 2)
                # obs_primary = torch.zeros((1, 2, 3, 256, 256), dtype=rgb_image_primary.dtype, device=device)
                obs_primary[:, 0] = obs_primary[:, 1].clone() 
                obs_primary[:, 1] = rgb_image_primary
                
                rgb_image_wrist = torch.tensor(obs["sensor_data"]["hand_camera"]["rgb"], device=device).permute(0, 3, 1, 2)
                # obs_wrist = torch.zeros((1, 2, 3, 128, 128), dtype=rgb_image_wrist.dtype, device=device)
                obs_wrist[:, 0] = obs_wrist[:, 1].clone() 
                obs_wrist[:, 1] = rgb_image_wrist
                
                observation = {'image_primary': obs_primary, 'image_wrist': obs_wrist}                
                # obs_primary[:, 1] = rgb_image_primary
                # obs_wrist[:, 1] = rgb_image_wrist
                timestep_pad_mask = torch.tensor([True, True], dtype=torch.bool).to(device)
                observation = {'image_primary': obs_primary, 'image_wrist': obs_wrist}



                env.render()
                # env.render_human() # will render with a window if possible
                
            cnt_envs += 1
            print
            if (env.evaluate()['success']):
                cnt_success += 1
                
            env.close()
            
    success_rate = cnt_success / cnt_envs
    print(f"Success rate: {success_rate}")
    


if __name__ == '__main__':
    args = parse_args()
    main(args)