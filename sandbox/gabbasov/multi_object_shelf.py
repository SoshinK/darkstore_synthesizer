from store import (
    MyShelfAsset,
    COUNT_OF_PRODUCT_ON_SHELF,
    BOARDS,
    COUNT_OF_PRODUCT_ON_BOARD,
    ASSETS_PATH,
)
import numpy as np
import scene_synthesizer as synth
from scene_synthesizer import procedural_assets as pa
from scene_synthesizer import procedural_scenes as ps
from scene_synthesizer.assets import TrimeshSceneAsset
from scene_generator import add_many_products, get_orientation
from scene_synthesizer import utils
import trimesh.transformations as tra
import json
import trimesh
import os

shelf = MyShelfAsset(
    width=1.517,
    depth=0.5172,
    height=2.0,
    board_thickness=0.05135,
    num_boards=BOARDS,
    num_side_columns=2,
    bottom_board=True,
    cylindrical_columns=False,
    num_vertical_boards=0,
    shift_bottom=0.131952 - 0.05135 / 2,
    shift_top=0.2288 + 0.05135 / 2,
)

def set_shelf(scene, x:float, y:float, rotation:bool, name:str, support_name:str)->None:
    if rotation:
        scene.add_object(shelf, name, transform=np.dot(tra.translation_matrix((x, y, 0.0)),
                                                                        tra.rotation_matrix(np.radians(90), [0, 0, 1])))                                                              tra.rotation_matrix(np.radians(90), [0, 0, 1])))
    else:
        scene.add_object(shelf, name, transform=tra.translation_matrix((x, y, 0.0)))
    support_data = scene.label_support(
        label=support_name,
        obj_ids=[name],
        min_area=0.05,
        gravity=np.array([0, 0, -1]),
    )
        
    