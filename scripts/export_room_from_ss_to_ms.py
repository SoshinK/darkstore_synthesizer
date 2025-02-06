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

if len(sys.argv) < 3:
    print("Использование: python script.py <путь_к_JSON_файлу> <путь_к_assets>")
    sys.exit(1)

json_file_path = sys.argv[1]
assets_dir = sys.argv[2]

def generate_random_string(length=10):
    characters = string.ascii_letters + string.digits
    return ''.join(random.choice(characters) for _ in range(length))


ENV_NAME = 'Darkstore'


CELL_SIZE = 1.55

def get_arena_config(x_cells=4, y_cells=5, height = 3):
    x_xize = x_cells * CELL_SIZE
    y_size = y_cells * CELL_SIZE
    return {
        'room': {
            'walls': [
                {'name': 'wall', 'type': 'wall', 'size': [x_xize / 2, height / 2, 0.02], 'pos': [x_xize / 2, y_size, height / 2]}, 
                {'name': 'wall_backing', 'type': 'wall', 'backing': True, 'backing_extended': [True, False], 'size': [x_xize / 2, height / 2, 0.1], 'pos': [x_xize / 2, y_size, height / 2]}, 
                {'name': 'wall_left', 'type': 'wall', 'wall_side': 'left', 'size': [y_size / 2, height / 2, 0.02], 'pos': [0, y_size / 2, height / 2]}, 
                {'name': 'wall_left_backing', 'type': 'wall', 'wall_side': 'left', 'backing': True, 'size': [y_size / 2, height / 2, 0.1], 'pos': [0, y_size / 2, height / 2]}, 
                {'name': 'wall_right', 'type': 'wall', 'wall_side': 'right', 'size': [y_size / 2, height / 2, 0.02], 'pos': [x_xize, y_size / 2, height / 2]}, 
                {'name': 'wall_right_backing', 'type': 'wall', 'wall_side': 'right', 'backing': True, 'size': [y_size / 2, height / 2, 0.1], 'pos': [x_xize, y_size / 2, height / 2]}
            ], 
            'floor': [
                {'name': 'floor', 'type': 'floor', 'size': [x_xize / 2, y_size / 2, 0.02], 'pos': [x_xize / 2, y_size / 2, 0.0]}, 
                {'name': 'floor_backing', 'type': 'floor', 'backing': True, 'size': [x_xize / 2, y_size / 2, 0.1], 'pos': [x_xize / 2, y_size / 2, 0.0]}
            ]
        }
    }

@register_env(ENV_NAME, max_episode_steps=200000)
class OurEnv(BaseEnv):
    SUPPORTED_REWARD_MODES = ["none"]
    """
    This is just a very smart environment for goida transformation from ss
    """
    IMPORTED_SS_SCENE_SHIFT = np.array([CELL_SIZE / 2, CELL_SIZE / 2, 0])

    def __init__(self, *args, robot_uids="panda", arena_config = None, **kwargs):
        self.arena_config = arena_config
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at([1.25, -1.25, 1.5], [0.0, 0.0, 0.2])
        return [CameraConfig("base_camera", pose, 256, 256, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.2, 0.2, 4], [5, 5, 2])
        return CameraConfig(
            "render_camera", pose=pose, width=512, height=512, fov=1, near=0.01, far=100
        )

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.scene_builder = EmptyRoomFromRobocasa(self, arena_config=self.arena_config)
        self.scene_builder.build()
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
            # print(f"Doing GOIDA IN PROCESS for name {node[1]} with parent {node[0]}")
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
    
    def _load_scene_from_json(self, options: dict):
        super()._load_scene(options)
        self.actors = []

        scale = np.array(options.get("scale", [1.0, 1.0, 1.0]))
        origin = np.array(options.get("origin", [0.0, 0.0, 0.0]))

        with open(json_file_path, "r") as f: # big_scene , one_shelf_many_milk_scene , customize
            data = json.load(f)

        nodes_dict = {}
        for node in data["graph"]:
            nodes_dict[node[1]] = node

        for node in data["graph"]:
            parent_name, obj_name, props = node
            if ('/' not in obj_name):
                abs_matrix = self._get_absolute_matrix(node, nodes_dict)

                p, q = self._get_pq(abs_matrix, origin)
                p += self.IMPORTED_SS_SCENE_SHIFT

                asset_file = os.path.join(assets_dir, self._temp_process_string(obj_name))
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
                    else:
                        builder.add_convex_collision_from_file(filename=asset_file, scale=scale)
                        actor = builder.build(name=obj_name)

                    self.actors.append(actor)





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
            self.agent.robot.set_pose(sapien.Pose([1.0, 0, 0.0]))

            # self.ground.set_collision_group_bit(
            #     group=2, bit_idx=FETCH_WHEELS_COLLISION_BIT, bit=1
            # )
        else:
            raise NotImplementedError


    def evaluate(self):
        return {}

    def _get_obs_extra(self, info: Dict):
        return dict()
    
arena_config = get_arena_config(x_cells=10, y_cells=5)

env = gym.make(ENV_NAME, robot_uids='fetch', arena_config=arena_config, num_envs=1, render_mode="human", enable_shadow=True)


# env = RecordEpisode(
#     env,
#     "./videos", # the directory to save replay videos and trajectories to
#     # on GPU sim we record intervals, not by single episodes as there are multiple envs
#     # each 100 steps a new video is saved
#     max_steps_per_video=100
# )

# step through the environment with random actions
obs, _ = env.reset()


viewer = env.render()
if isinstance(viewer, sapien.utils.Viewer):
    viewer.paused = False
env.render()


for i in tqdm(range(10000)):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(torch.zeros_like(torch.from_numpy(action)))

    env.render()
    # env.render_human() # will render with a window if possible
env.close()
