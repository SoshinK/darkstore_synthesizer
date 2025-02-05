
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

ENV_NAME ="MyEmptyEnv"

@register_env(ENV_NAME, max_episode_steps=200000)
class MyEmptyEnv(BaseEnv):
    SUPPORTED_REWARD_MODES = ["none"]
    """
    This is just a dummy environment for showcasing robots in a empty scene
    """

    def __init__(self, *args, robot_uids="panda", **kwargs):
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
        self.ground = build_ground(self.scene)
        self.ground.set_collision_group_bit(group=2, bit_idx=30, bit=1)


        #===============================
        # Load URDF articuation
        #===============================
        # loader = self.scene.create_urdf_loader()
        # articulation_builders = loader.parse(str('/home/kvsoshin/Work/AIRI/ManiSkill/scene.urdf'))["articulation_builders"]
        # builder = articulation_builders[0]

        # builder.initial_pose = sapien.Pose(p=[0, 0, 0.5])
        # builder.build(name="my_articulation")


        #===============================
        # Load actor from glb
        #===============================
        scale=np.array([1.0, 1.0, 1.0])
        builder = self.scene.create_actor_builder()
        builder.add_convex_collision_from_file(
            filename="/home/kvsoshin/Work/AIRI/ManiSkill/textures/milk_carton.glb",
            scale=scale
        )
        builder.add_visual_from_file(filename="/home/kvsoshin/Work/AIRI/ManiSkill/textures/milk_carton.glb", scale=scale)
        builder.set_initial_pose(sapien.Pose(p=[0.0, 0.0, 0.0]))
        self.mesh = builder.build_static(name="mesh")



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
            self.agent.robot.set_pose(sapien.Pose([10.0, 10, 0.0]))

            self.ground.set_collision_group_bit(
                group=2, bit_idx=FETCH_WHEELS_COLLISION_BIT, bit=1
            )
        else:
            raise NotImplementedError


    def evaluate(self):
        return {}

    def _get_obs_extra(self, info: Dict):
        return dict()
    

env = gym.make(ENV_NAME, robot_uids='fetch', num_envs=1, render_mode="human", enable_shadow=True)

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