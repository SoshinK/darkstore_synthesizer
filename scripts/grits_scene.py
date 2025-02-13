import numpy as np
import scene_synthesizer as synth
from scene_synthesizer import procedural_assets as pa
from scene_synthesizer import procedural_scenes as ps
from scene_synthesizer.assets import TrimeshSceneAsset
from scene_synthesizer import utils
import trimesh.transformations as tra
import json
import sys
import argparse
import trimesh
import os
sys.path.append('.')
from dsynth.ss_scene.store import try_shelf_placement


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Запуск генерации сцены")
    parser.add_argument(
        "--input",
        default="models/input.json",
        help="Путь к JSON-файлу с входными данными (по умолчанию: models/input.json)"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Показать сцену после обработки"
    )
    parser.add_argument(
        "--its",
        default=1,
        help="Сколько сцен генерировать"
    )

    args = parser.parse_args()

    with open(args.input, "r") as f:
        data = json.load(f)
    
    for j in range(int(args.its)):
        n, m = data["room_size"]
        x, y = data["door_coords"]

        room = [[0, "milk"], [0, 0]]
        is_rotate = [[0, 1], [0 , 0]]

        try_shelf_placement(room, is_rotate, data['random_shelfs'], args.show, suf = str(j))