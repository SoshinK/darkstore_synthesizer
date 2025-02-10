from dsynth.scenes.darkstore_env import DarkstoreEnv


class PickToCart(DarkstoreEnv):


    def evaluate(self):
        is_obj_placed = (
            torch.linalg.norm(self.goal_site.pose.p - self.actors["objects"]["milk"][0], axis=1)
            <= 0.001
        )
        is_robot_static = self.agent.is_static(0.2)
        return {
            "success": is_obj_placed & is_robot_static,
            "is_obj_placed": is_obj_placed,
            "is_robot_static": is_robot_static,
        }

