
from mani_skill.envs.tasks.empty_env import EmptyEnv
from mani_skill.envs.sapien_env import BaseEnv

from typing import Dict

from mani_skill.agents.robots.fetch import FETCH_WHEELS_COLLISION_BIT
from mani_skill.utils.building.ground import build_ground
# from mani_skill.utils.scene_builder import SceneBuilder
from mani_skill.utils.registration import register_env
from mani_skill.utils import common, sapien_utils
import sapien
from mani_skill.sensors.camera import CameraConfig

import numpy as np
import gymnasium as gym
import torch
import mani_skill.envs
from tqdm import tqdm
from mani_skill.utils.wrappers import RecordEpisode
import json
from robocasa_scene_builder import EmptyRoomFromRobocasa

ENV_NAME ="EmptyDarkstore"

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
class EmptyDarkstore(BaseEnv):
    SUPPORTED_REWARD_MODES = ["none"]
    """
    This is just a dummy environment for showcasing robots in a empty scene
    """

    def __init__(self, *args, robot_uids="panda", arena_config = None, **kwargs):
        self.arena_config = arena_config            
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at([1.25, -1.25, 1.5], [0.0, 0.0, 0.2])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([1.25, -1.25, 1.5], [0.0, 0.0, 0.2])
        return CameraConfig("render_camera", pose, 2048, 2048, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.scene_builder = EmptyRoomFromRobocasa(self, arena_config=self.arena_config)
        self.scene_builder.build()
        # self.fixtures = data["fixtures"]
        # self.actors = data["actors"]
        # self.fixture_configs = data["fixture_configs"]
        self.fixture_refs = []
        self.objects = []
        self.object_cfgs = []
        self.object_actors = []
        for _ in range(self.num_envs):
            self.fixture_refs.append({})
            self.objects.append({})
            self.object_cfgs.append({})
            self.object_actors.append({})




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

arena_config = get_arena_config(x_cells=4, y_cells=5)

env = gym.make(ENV_NAME, robot_uids='fetch', num_envs=1, arena_config=arena_config, render_mode="human", enable_shadow=True)

pose = sapien_utils.look_at([3.25, -3.25, 1.5], [0.0, 0.0, 0.2])
env._default_human_render_camera_configs = CameraConfig("render_camera", pose, 2048, 2048, 1, 0.01, 100)

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
from IPython.display import Video
Video("./videos/0.mp4", embed=True, width=640)