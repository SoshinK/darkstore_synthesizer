import torch

import gymnasium as gym
import numpy as np
import sapien

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers import RecordEpisode


import tyro
from dataclasses import dataclass
from typing import List, Optional, Annotated, Union




from mani_skill.utils.registration import register_env
from mani_skill.envs.tasks.tabletop.push_cube import PushCubeEnv
from mani_skill.utils.building import actors

from mani_skill.utils.structs.pose import Pose

import json
from transforms3d import quaternions

@register_env("MyEnv", max_episode_steps=50)
class MyEnv(PushCubeEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _load_scene1(self, options: dict):
        super()._load_scene(options)

        self.obj2 = actors.build_cube(
            self.scene,   
            half_size=self.cube_half_size,
            color=np.array([42, 42, 160, 255]) / 255,
            name="cube2",
            body_type="dynamic",
            initial_pose=sapien.Pose(p=[0.1, 0.1, self.cube_half_size + 0.2]),
        )

        # scale=(1.0, 1.0, 1.0,)
        # builder = self.scene.create_actor_builder()
        # builder.add_convex_collision_from_file(
        #     filename="untitled.glb",
        #     scale=scale
        # )
        # builder.add_visual_from_file(filename="untitled.glb", scale=scale)
        # builder.set_initial_pose(sapien.Pose(p=[-0.0, 0.0, 1.0 + 0.05]))
        # self.mesh = builder.build(name="mesh")

        #=======
        # p = [[0.3, 0.3, 1.0 + 0.05],
        #      [-0.2, -0.3, 1.0 + 0.05]]
        # q = [1, 0, 0, 0]
        # obj_pose = Pose.create_from_pq(p=p, q=q)
        # self.mesh.set_pose(obj_pose)

    # def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
    #     super()._initialize_episode(env_idx, options)
    #     # p = [[-0.3, 0.3, 1.0 + 0.05],
    #     #      [-0.4, -0.6, 1.0 + 0.05]]
    #     p = [-0.3, -0.3, 1.0 + 0.05]
    #     # p = [-0.4, -0.6, 1.0 + 0.05]
    #     q = [0.1, 0, 0, 0]
    #     obj_pose = Pose.create_from_pq(p=p, q=q)
    #     self.mesh.set_pose(obj_pose)



    def _load_scene3(self, options: dict):
        super()._load_scene(options)
        builder = self.scene.create_actor_builder()
        builder.add_convex_collision_from_file(
            filename="/home/kvsoshin/Work/AIRI/scene_synthesizer/shit_urdf/shit.urdf",
            scale=(0.4, 0.4, 0.4,)
        )
        builder = self.scene.create_actor_builder()
        builder.add_visual_from_file(filename="/home/kvsoshin/Work/AIRI/scene_synthesizer/shit_urdf/shit.urdf", scale=(0.4, 0.4, 0.4,))
        builder.set_initial_pose(sapien.Pose(p=[-0.3, 0.3, 1.0 + 0.05]))
        self.mesh = builder.build_static(name="mesh")

    def _load_scene2(self, options: dict):
        super()._load_scene(options)
        loader = self.scene.create_urdf_loader()

        articulation_builders = loader.parse(str('/home/kvsoshin/Work/AIRI/scene_synthesizer/shit_urdf/shit.urdf'))["articulation_builders"]
        builder = articulation_builders[0]
        # choose a reasonable initial pose that doesn't intersect other objects
        # this matters a lot for articulations in GPU sim or else simulation bugs can occur
        builder.initial_pose = sapien.Pose(p=[0, 0, 0.5])
        builder.build(name="my_articulation")

    def _load_scene4(self, options: dict):
        super()._load_scene(options)
        builder = self.scene.create_actor_builder()
        builder.add_multiple_convex_collisions_from_file(
            filename="cans.obj",
            scale=(0.4, 0.4, 0.4,),
            decomposition='coacd'
        )
        # builder = self.scene.create_actor_builder()
        builder.add_visual_from_file(filename="cans.obj", scale=(0.4, 0.4, 0.4,))
        builder.set_initial_pose(sapien.Pose(p=[-0.3, 0.3, 1.0 + 0.05]))
        self.mesh = builder.build_static(name="mesh")

    def _load_scene5(self, options: dict):
        super()._load_scene(options)

        # self.obj2 = actors.build_cube(
        #     self.scene,   
        #     half_size=self.cube_half_size,
        #     color=np.array([42, 42, 160, 255]) / 255,
        #     name="cube2",
        #     body_type="dynamic",
        #     initial_pose=sapien.Pose(p=[0.3, 0.3, self.cube_half_size + 0.5]),
        # )

        self.meshes = []

        with open('/home/kvsoshin/Work/AIRI/scene_synthesizer/apples.json', 'r') as f:
            d = json.load(f)
        d_processed = [d['graph'][i] for i in range(len(d['graph'])) if not 'geometry' in d['graph'][i][2]]
        d_processed = [obj for obj in d_processed if 'can' in obj[1]]
        hmatrices = [np.array(obj[2]['matrix']) for obj in d_processed]

        shift = np.array([0.1, 0.2, 0.3])
        transform = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ])
        for i, matrix in enumerate(hmatrices):
            # matrix = matrix @ transform
            q = quaternions.mat2quat(matrix[:3,:3])

            p = matrix[:-1, 3] - shift

            builder = self.scene.create_actor_builder()

            scale = (0.002, 0.002, 0.002)
            builder.add_convex_collision_from_file(
                filename="apple.glb",
                scale=scale
            )
            builder.add_visual_from_file(filename="apple.glb", scale=scale)
            builder.set_initial_pose(sapien.Pose(p=p, q=q))
            self.meshes.append(builder.build(name=f"mesh_{i}"))
        # self.mesh = builder.build(name="mesh")

    def _load_scene6(self, options: dict):
        super()._load_scene(options)

        self.meshes = []

        with open('/home/kvsoshin/Work/AIRI/scene_synthesizer/milks.json', 'r') as f:
            d = json.load(f)
        d_processed = [d['graph'][i] for i in range(len(d['graph'])) if not 'geometry' in d['graph'][i][2]]
        d_processed = [obj for obj in d_processed if 'milk' in obj[1]]
        hmatrices = [np.array(obj[2]['matrix']) for obj in d_processed]
        print("WTF!!!", hmatrices)
        shift = np.array([0.1, 0.2, 0.3])
        
        for i, matrix in enumerate(hmatrices):
            # matrix = matrix @ transform
            q = quaternions.mat2quat(matrix[:3,:3])

            p = matrix[:-1, 3] - shift

            builder = self.scene.create_actor_builder()

            scale = (0.1, 0.1, 0.1)
            builder.add_convex_collision_from_file(
                filename="untitled.glb",
                scale=scale
            )
            builder.add_visual_from_file(filename="untitled.glb", scale=scale)
            builder.set_initial_pose(sapien.Pose(p=p, q=q))
            self.meshes.append(builder.build(name=f"mesh_{i}"))
        # self.mesh = builder.build(name="mesh")

    def _load_scene_shelf_milk(self, options: dict):
        super()._load_scene(options)

        self.obj2 = actors.build_cube(
            self.scene,   
            half_size=self.cube_half_size,
            color=np.array([42, 42, 160, 255]) / 255,
            name="cube2",
            body_type="dynamic",
            initial_pose=sapien.Pose(p=[0.3, 0.3, self.cube_half_size + 0.5]),
        )

        scale=(1.0, 1.0, 1.0,)
        builder = self.scene.create_actor_builder()
        builder.add_nonconvex_collision_from_file(
            filename="/home/kvsoshin/Work/AIRI/3d_assets/metal_shelf_-_5mb_processed.glb",
            scale=scale
        )
        builder.add_visual_from_file(filename="/home/kvsoshin/Work/AIRI/3d_assets/metal_shelf_-_5mb_processed.glb", scale=scale)
        builder.set_initial_pose(sapien.Pose(p=[0.0, 0.0, 1.5 -0.92]))
        self.mesh = builder.build_static(name="mesh")

        scale=(1.0, 1.0, 1.0,)
        builder = self.scene.create_actor_builder()
        builder.add_convex_collision_from_file(
            filename="/home/kvsoshin/Work/AIRI/3d_assets/milk_processed.glb",
            scale=scale
        )
        builder.add_visual_from_file(filename="/home/kvsoshin/Work/AIRI/3d_assets/milk_processed.glb", scale=scale)
        half_milk = 0.18
        floor_to_low_clearance = 0.184
        shelf_clearance = 0.40

        builder.set_initial_pose(sapien.Pose(p=[0.0, 0.0, 1.5  -0.92+ half_milk + floor_to_low_clearance + 0.01]))
        self.mesh2 = builder.build(name="mesh2")


    def _load_shelf_urdf(self, options: dict):
        super()._load_scene(options)

        loader = self.scene.create_urdf_loader()
        articulation_builders = loader.parse(str('/home/kvsoshin/Work/AIRI/scene_synthesizer/shelf.urdf'))["articulation_builders"]
        builder = articulation_builders[0]
        builder.initial_pose = sapien.Pose(p=[0, 0, 0.5])
        builder.build(name="shelf")

    def _load_shelf(self, options: dict):
        super()._load_scene(options)

        builder = self.scene.create_actor_builder()
        builder.add_nonconvex_collision_from_file(
            filename="/home/kvsoshin/Work/AIRI/scene_synthesizer/shelf.obj",
            scale=(1.0, 1.0, 1.0,)
        )
        builder.add_visual_from_file(filename="/home/kvsoshin/Work/AIRI/3d_assets/metal_shelf_-_5mb_processed.glb", scale=(1.0, 1.0, 1.0,))
        builder.set_initial_pose(sapien.Pose(p=[0.0, 0.0, 1.5 -0.92]))
        self.mesh = builder.build_static(name="mesh")


    def _get_pq(self, matrix, origin):
        matrix = np.array(matrix)
        q = quaternions.mat2quat(matrix[:3,:3])
        p = matrix[:-1, 3] - origin
        return p, q

    def _load_shelf_arrangement_from_json(self, options: dict):
        super()._load_scene(options)

        self.things = []

        with open('/home/kvsoshin/Work/AIRI/scene_synthesizer/shelf.json', 'r') as f:
            d = json.load(f)

        d_processed = [d['graph'][i] for i in range(len(d['graph'])) if not 'geometry' in d['graph'][i][2]]
        things = [obj for obj in d_processed if 'milk' in obj[1]]
        shelf = [obj for obj in d_processed if 'shelf' in obj[1]]
        
        scale = np.array([1.0, 1.0, 1.0])
        origin = np.array([0.0, 1.0, -0.0])

        # make shelf:
        p_shelf, q_shelf = self._get_pq(shelf[0][2]['matrix'], origin)
        builder = self.scene.create_actor_builder()
        builder.add_nonconvex_collision_from_file(
            filename="/home/kvsoshin/Work/AIRI/3d_assets/metal_shelf_-_5mb_processed.glb",
            scale=scale
        )
        builder.add_visual_from_file(filename="/home/kvsoshin/Work/AIRI/3d_assets/metal_shelf_-_5mb_processed.glb", scale=scale)
        builder.set_initial_pose(sapien.Pose(p=p_shelf, q=q_shelf))
        self.mesh = builder.build_static(name="shelf")


        # place things:
        for i, thing_placement in enumerate(things):
            p_thing, q_thing = self._get_pq(thing_placement[2]['matrix'], origin)

            if 'milk_' in thing_placement[1]:
                vis_name = '/home/kvsoshin/Work/AIRI/3d_assets/milk_karton_processed.glb'
                collision_fname = vis_name
                shift_z= np.array([0., 0., 0.0285])
            elif 'milk2' in thing_placement[1]:
                vis_name = '/home/kvsoshin/Work/AIRI/3d_assets/milk_processed.glb'
                collision_fname = vis_name
            else:
                raise NotImplementedError
            
            builder = self.scene.create_actor_builder()
            builder.add_convex_collision_from_file(
                filename=collision_fname,
                scale=scale
            )
            builder.add_visual_from_file(filename=vis_name, scale=scale)
            builder.set_initial_pose(sapien.Pose(p=p_thing + shift_z, q=q_thing))
            self.things.append(builder.build(name=f"{i}_{thing_placement[1]}"))



    def _load_scene(self, options: dict):
        # self._load_shelf_arrangement_from_json(options)
        self._load_scene6(options)

@dataclass
class Args:
    # env_id: Annotated[str, tyro.conf.arg(aliases=["-e"])] = "PushCube-v1"
    # """The environment ID of the task you want to simulate"""

    obs_mode: Annotated[str, tyro.conf.arg(aliases=["-o"])] = "none"
    """Observation mode"""

    robot_uids: Annotated[Optional[str], tyro.conf.arg(aliases=["-r"])] = None
    """Robot UID(s) to use. Can be a comma separated list of UIDs or empty string to have no agents. If not given then defaults to the environments default robot"""

    sim_backend: Annotated[str, tyro.conf.arg(aliases=["-b"])] = "auto"
    """Which simulation backend to use. Can be 'auto', 'cpu', 'gpu'"""

    reward_mode: Optional[str] = None
    """Reward mode"""

    num_envs: Annotated[int, tyro.conf.arg(aliases=["-n"])] = 1
    """Number of environments to run."""

    control_mode: Annotated[Optional[str], tyro.conf.arg(aliases=["-c"])] = None
    """Control mode"""

    render_mode: str = "rgb_array"
    """Render mode"""

    shader: str = "default"
    """Change shader used for all cameras in the environment for rendering. Default is 'minimal' which is very fast. Can also be 'rt' for ray tracing and generating photo-realistic renders. Can also be 'rt-fast' for a faster but lower quality ray-traced renderer"""

    record_dir: Optional[str] = None
    """Directory to save recordings"""

    pause: Annotated[bool, tyro.conf.arg(aliases=["-p"])] = False
    """If using human render mode, auto pauses the simulation upon loading"""

    quiet: bool = False
    """Disable verbose output."""

    seed: Annotated[Optional[Union[int, List[int]]], tyro.conf.arg(aliases=["-s"])] = None
    """Seed(s) for random actions and simulator. Can be a single integer or a list of integers. Default is None (no seeds)"""

def main(args: Args):
    np.set_printoptions(suppress=True, precision=3)
    verbose = not args.quiet
    if isinstance(args.seed, int):
        args.seed = [args.seed]
    if args.seed is not None:
        np.random.seed(args.seed[0])
    parallel_in_single_scene = args.render_mode == "human"
    if args.render_mode == "human" and args.obs_mode in ["sensor_data", "rgb", "rgbd", "depth", "point_cloud"]:
        print("Disabling parallel single scene/GUI render as observation mode is a visual one. Change observation mode to state or state_dict to see a parallel env render")
        parallel_in_single_scene = False
    if args.render_mode == "human" and args.num_envs == 1:
        parallel_in_single_scene = False
    env_kwargs = dict(
        obs_mode=args.obs_mode,
        reward_mode=args.reward_mode,
        control_mode=args.control_mode,
        render_mode=args.render_mode,
        sensor_configs=dict(shader_pack=args.shader),
        human_render_camera_configs=dict(shader_pack=args.shader),
        viewer_camera_configs=dict(shader_pack=args.shader),
        num_envs=args.num_envs,
        sim_backend=args.sim_backend,
        enable_shadow=True,
        parallel_in_single_scene=parallel_in_single_scene,
        render_backend='cpu',
    )
    if args.robot_uids is not None:
        env_kwargs["robot_uids"] = tuple(args.robot_uids.split(","))
    env: BaseEnv = gym.make(
        'MyEnv',
        **env_kwargs
    )
    record_dir = args.record_dir
    if record_dir:
        record_dir = record_dir.format(env_id='MyEnv')
        env = RecordEpisode(env, record_dir, info_on_video=False, save_trajectory=False, max_steps_per_video=gym_utils.find_max_episode_steps_value(env))

    if verbose:
        print("Observation space", env.observation_space)
        print("Action space", env.action_space)
        if env.unwrapped.agent is not None:
            print("Control mode", env.unwrapped.control_mode)
        print("Reward mode", env.unwrapped.reward_mode)

    obs, _ = env.reset(seed=args.seed, options=dict(reconfigure=True))
    if args.seed is not None and env.action_space is not None:
            env.action_space.seed(args.seed[0])
    if args.render_mode is not None:
        viewer = env.render()
        if isinstance(viewer, sapien.utils.Viewer):
            viewer.paused = args.pause
        env.render()
    while True:
        action = env.action_space.sample() if env.action_space is not None else None
        obs, reward, terminated, truncated, info = env.step(action)
        if verbose:
            print("reward", reward)
            print("terminated", terminated)
            print("truncated", truncated)
            print("info", info)
        if args.render_mode is not None:
            env.render()
        if args.render_mode is None or args.render_mode != "human":
            if (terminated | truncated).any():
                break
    env.close()

    if record_dir:
        print(f"Saving video to {record_dir}")


if __name__ == "__main__":
    parsed_args = tyro.cli(Args)
    main(parsed_args)
