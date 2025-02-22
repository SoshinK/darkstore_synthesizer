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
from dsynth.scenes.robocasaroom import RoomFromRobocasa
from mani_skill.utils.building import actors

CELL_SIZE = 1.55
DEFAULT_ASSETS_DIR = 'models'

def get_arena_data(x_cells=4, y_cells=5, height = 3):
    x_size = x_cells * CELL_SIZE
    y_size = y_cells * CELL_SIZE
    return {
        'meta': {
            'x_size': x_size,
            'y_size': y_size,
            'height': height
        },
        'arena_config': {
            'room': {
                'walls': [
                    {'name': 'wall', 'type': 'wall', 'size': [x_size / 2, height / 2, 0.02], 'pos': [x_size / 2, y_size, height / 2]}, 
                    {'name': 'wall_backing', 'type': 'wall', 'backing': True, 'backing_extended': [True, False], 'size': [x_size / 2, height / 2, 0.1], 'pos': [x_size / 2, y_size, height / 2]}, 
                    
                    {'name': 'wall_front', 'type': 'wall', 'wall_side' : 'front', 'size': [x_size / 2, height / 2, 0.02], 'pos': [x_size / 2, 0, height / 2]}, 
                    {'name': 'wall_front_backing', 'type': 'wall', 'wall_side' : 'front', 'backing': True, 'size': [x_size / 2, height / 2, 0.1], 'pos': [x_size / 2, 0, height / 2]}, 
                    
                    {'name': 'wall_left', 'type': 'wall', 'wall_side': 'left', 'size': [y_size / 2, height / 2, 0.02], 'pos': [0, y_size / 2, height / 2]}, 
                    {'name': 'wall_left_backing', 'type': 'wall', 'wall_side': 'left', 'backing': True, 'size': [y_size / 2, height / 2, 0.1], 'pos': [0, y_size / 2, height / 2]}, 
                    
                    {'name': 'wall_right', 'type': 'wall', 'wall_side': 'right', 'size': [y_size / 2, height / 2, 0.02], 'pos': [x_size, y_size / 2, height / 2]}, 
                    {'name': 'wall_right_backing', 'type': 'wall', 'wall_side': 'right', 'backing': True, 'size': [y_size / 2, height / 2, 0.1], 'pos': [x_size, y_size / 2, height / 2]}
                ], 
                'floor': [
                    {'name': 'floor', 'type': 'floor', 'size': [x_size / 2, y_size / 2, 0.02], 'pos': [x_size / 2, y_size / 2, 0.0]}, 
                    {'name': 'floor_backing', 'type': 'floor', 'backing': True, 'size': [x_size / 2, y_size / 2, 0.1], 'pos': [x_size / 2, y_size / 2, 0.0]}
                ]
            }
        }
    }


@register_env('DarkstoreEnv', max_episode_steps=200000)
class DarkstoreEnv(BaseEnv):
    SUPPORTED_REWARD_MODES = ["none"]
    """
    This is just a very smart environment for goida transformation from ss
    """
    IMPORTED_SS_SCENE_SHIFT = np.array([CELL_SIZE / 2, CELL_SIZE / 2, 0])

    def __init__(self, *args, 
                 robot_uids="panda_wristcam", 
                 scene_json=None, 
                 arena_config = None, 
                 meta = None, 
                 style_ids = 0, 
                 mapping_file=None,
                 assets_dir = DEFAULT_ASSETS_DIR,
                 **kwargs):
        self.style_ids = style_ids
        self.arena_config = arena_config
        self.json_file_path = scene_json
        self.mapping_file = mapping_file
        self.assets_dir = assets_dir
        self.x_size = meta['x_size']
        self.y_size = meta['y_size']
        self.height = meta['height']
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at([0.7, 1.8, 1.15], [1.2, 2.2, 1.2])
        return [CameraConfig("base_camera", pose, 256, 256, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        # pose = sapien_utils.look_at([0.2, 0.2, 4], [5, 5, 2])
        pose = sapien_utils.look_at([3, 3, 3], [0, 0, 0])
        return CameraConfig(
            "render_camera", pose=pose, width=512, height=512, fov=1, near=0.01, far=100
        )

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.scene_builder = RoomFromRobocasa(self, arena_config=self.arena_config)
        self.scene_builder.build(self.style_ids)
        self._load_scene_from_json(options)

    def _process_string(self, s):
        if '_' in s:
            return s.split('_',1)[0] + '.obj'
        if '.' in s:
            return s.split('.',1)[0] + '.obj'
        return s + '.obj'

    def _temp_process_string(self, s):
        for i, char in enumerate(s):
            if char in "_." or char.isdigit():
                return s[:i] + ".obj"
        return s + ".obj"

    def _get_absolute_matrix(self, node, nodes_dict):
        current_matrix = np.array(node[2]["matrix"])
        parent_name = node[0]
        while parent_name != "world":
            parent_node = nodes_dict[parent_name]
            parent_matrix = np.array(parent_node[2]["matrix"])
            current_matrix = parent_matrix @ current_matrix
            parent_name = parent_node[0]
        return current_matrix

    def _get_pq(self, matrix, origin):
        matrix = np.array(matrix)
        q = quaternions.mat2quat(matrix[:3,:3])
        p = matrix[:-1, 3] - origin
        return p, q
    
    def _add_noise(self, p, max_noise = 1e-4):
        new_p = [0] * len(p)
        for i in range(len(p)):
            new_p[i] = p[i] + random.randrange(-max_noise, max_noise)
        return new_p
    
    def _load_shopping_cart(self, options: dict):
        # recommended to use shift = (0,0.5,0)
        # print(self.unwrapped.agent.robot.get_pose())
        if not hasattr(self, 'shopping_cart'):
            shopping_cart_asset = os.path.join(self.assets_dir, "smallShoppingCart2.glb")
            self.cube_half_size = 0.2
            
            if not os.path.exists(shopping_cart_asset):
                print(f"Shopping cart asset not found: {shopping_cart_asset}")
            else:
                builder = self.scene.create_actor_builder()
                builder.add_visual_from_file(filename=shopping_cart_asset, scale=np.array([1.0, 1.0, 1.0]))
                builder.add_nonconvex_collision_from_file(filename=shopping_cart_asset, scale=np.array([1.0, 1.0, 1.0]))
                shopping_cart_pose = sapien.Pose(p=[11.0, 10.0, 0.0], q=np.array([1, 0, 0, 0]))
                builder.set_initial_pose(shopping_cart_pose)
                self.shopping_cart = builder.build_static(name="shopping_cart")
                # self.actors.append(self.shopping_cart)
                self.target_volume = actors.build_cube(
                    self.scene,
                    half_size=self.cube_half_size,
                    color=[0, 0, 0, 0],
                    name="cube",
                    body_type="static",
                    add_collision=False,
                    initial_pose=shopping_cart_pose,
                )

    
    def _load_scene_from_json(self, options: dict):
        super()._load_scene(options)
        self.actors = {
            "fixtures": {
                "shelf" : [

                ]
            },
            "objects": {

            }
        }

        scale = np.array(options.get("scale", [1.0, 1.0, 1.0]))
        origin = - self.IMPORTED_SS_SCENE_SHIFT#np.array(options.get("origin", [0.0, 1.0, 0.0]))

        with open(self.json_file_path, "r") as f:
            data = json.load(f)

        nodes_dict = {}
        for node in data["graph"]:
            nodes_dict[node[1]] = node

        asset_mapping = {}
        if self.mapping_file is not None:
            with open(self.mapping_file, "r") as f:
                asset_mapping = json.load(f)

        for node in data["graph"]:
            parent_name, obj_name, props = node
            if ('/' not in obj_name):
                abs_matrix = self._get_absolute_matrix(node, nodes_dict)

                p, q = self._get_pq(abs_matrix, origin)

                obj_name_to_check = self._temp_process_string(obj_name)[:-4]

                if obj_name_to_check in asset_mapping:
                    asset_file = os.path.join(self.assets_dir, asset_mapping[obj_name_to_check])
                else:
                    asset_file = ""


                if not os.path.exists(asset_file):
                    asset_file = os.path.join(self.assets_dir, self._temp_process_string(obj_name))

                if not os.path.exists(asset_file):
                    asset_file = os.path.splitext(asset_file)[0] + ".glb"

                if not os.path.exists(asset_file):
                    print("Not found file for " + obj_name + " =(" + " ( " + asset_file + " )")
                else:
                    # print("Found file for " + obj_name + " =)" + " ( " + asset_file + " )")
                    builder = self.scene.create_actor_builder()
                    builder.add_visual_from_file(filename=asset_file, scale=scale)
                    builder.set_initial_pose(sapien.Pose(p=p, q=q))



                    if obj_name.startswith('shelf'):
                        builder.add_nonconvex_collision_from_file(filename=asset_file, scale=scale)
                        actor = builder.build_static(name=obj_name)
                        self.actors["fixtures"]["shelf"].append({"actor" : actor, "p" : p, "q" : q})
                    else:
                        builder.add_convex_collision_from_file(filename=asset_file, scale=scale)
                        actor = builder.build(name=obj_name)
                        if self.actors["objects"].get(obj_name, -1) == -1:
                            self.actors["objects"][obj_name] = []
                        self.actors["objects"][obj_name].append({"actor" : actor, "p" : p, "q" : q})

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        
        if self.robot_uids == "fetch":
            qpos = np.array(
                [
                    0,
                    0,
                    0,
                    0.386,
                    0,
                    0,
                    0,
                    -np.pi / 4,
                    0,
                    np.pi / 4,
                    0,
                    np.pi / 3,
                    0,
                    0.015,
                    0.015,
                ]
            )
            self.agent.reset(qpos)
            self.agent.robot.set_pose(sapien.Pose([0.5, 0.5, 0.0]))
            # self._load_shopping_cart(options)
        elif self.robot_uids == "panda_wristcam":
            qpos = np.array(
                [
                    0.0,        
                    -np.pi / 6, 
                    0.0,        
                    -np.pi / 3, 
                    0.0,        
                    np.pi / 2,  
                    np.pi / 4,  
                    0.04,       
                    0.04,       
                ]
            )
            self.agent.reset(qpos)
            self.agent.robot.set_pose(sapien.Pose([0.0, 0.0, 0.0]))
            
        else:
            raise NotImplementedError

    def _get_obs_extra(self, info: Dict):
        return dict()