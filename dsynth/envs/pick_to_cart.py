from dsynth.scenes.darkstore_env import DarkstoreEnv
from mani_skill.utils.registration import register_env
import torch
from mani_skill.utils import common, sapien_utils
from mani_skill.sensors.camera import CameraConfig
import sapien
import numpy as np

@register_env('PickToCart', max_episode_steps=200000)
class PickToCart(DarkstoreEnv):


    # def evaluate(self):
    #     is_obj_placed = (
    #         torch.linalg.norm(self.agent.robot.get_pose().p - self.actors["objects"]["milk_1_1_0"][0]['p'], axis=1)
    #         <= 0.001
    #     )
    #     is_robot_static = self.agent.is_static(0.2)
    #     return {
    #         "success": is_obj_placed & is_robot_static,
    #         "is_obj_placed": is_obj_placed,
    #         "is_robot_static": is_robot_static,
    #     }
        
    def _load_scene(self, options: dict):
        super()._load_scene(options)
        self._load_shopping_cart(options)
        
        
        
    @property
    def _default_human_render_camera_configs(self):
        # pose = sapien_utils.look_at([0.2, 0.2, 4], [5, 5, 2])
        pose = sapien_utils.look_at([-2, -2, 2], [1, 1, 1])
        return CameraConfig(
            "render_camera", pose=pose, width=512, height=512, fov=1, near=0.01, far=100
        )
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        robot_pose = self.agent.robot.get_pose()

        cart_shift = np.array([1.2, 0.5, 0.])
        new_cart_pose_p = robot_pose.p[0].numpy() + cart_shift 
        self.shopping_cart.set_pose(sapien.Pose(p=new_cart_pose_p, q=robot_pose.q[0].numpy()))
        
        if self.robot_uids == "panda":
            qpos = np.array(
                [
                    np.pi / 2,        
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
            self.agent.robot.set_pose(sapien.Pose([0.5, 1.7, 0.0]))
            


