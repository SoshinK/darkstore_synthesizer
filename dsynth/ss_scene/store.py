import numpy as np
import scene_synthesizer as synth
from scene_synthesizer import procedural_assets as pa
from scene_synthesizer import procedural_scenes as ps
from scene_synthesizer.assets import TrimeshSceneAsset
from scene_generator import add_many_products, get_orientation
from scene_synthesizer import utils
import trimesh.transformations as tra
import json
import sys
import argparse
import trimesh
import os

# CONST
COUNT_OF_PRODUCT_ON_SHELF = 2
BOARDS = 5
COUNT_OF_PRODUCT_ON_BOARD = 1

ASSETS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../models")

with open(f"{ASSETS_PATH}/assets.json", "r") as f:
    assets_config = json.load(f)

asset_type_mapping = {
    "MeshAsset": synth.assets.MeshAsset,
    "USDAsset": synth.assets.USDAsset,
}

NAMES_OF_PRODUCTS = {}

for name, params in assets_config.items():
    asset_type_str = params.pop("asset_type")
    asset_constructor = asset_type_mapping.get(asset_type_str)
    if asset_constructor is None:
        raise ValueError(f"Unknown asset type: {asset_type_str}")

    file_path = os.path.join(ASSETS_PATH, params.pop("filename"))

    asset_obj = asset_constructor(file_path, **params)

    NAMES_OF_PRODUCTS[name] = asset_obj


class UserError(Exception):
    pass


class MyShelfAsset(TrimeshSceneAsset):
    """A shelf asset."""

    def __init__(
        self,
        width,
        depth,
        height,
        num_boards,
        board_thickness=0.03,
        backboard_thickness=0.0,
        num_vertical_boards=0,
        num_side_columns=2,
        column_thickness=0.03,
        bottom_board=True,
        cylindrical_columns=True,
        shift_bottom=0.0,
        shift_top=0.0,
        **kwargs,
    ):
        boards = []
        board_names = []
        if backboard_thickness > 0:
            back = trimesh.primitives.Box(
                extents=[width, backboard_thickness, height],
                transform=tra.translation_matrix(
                    [0, depth / 2.0 + backboard_thickness / 2.0, height / 2.0]
                ),
            )
            boards.append(back)
            board_names.append("back")

        min_z = +float("inf")
        max_z = -float("inf")
        cnt = 0
        for h in np.linspace(
            shift_bottom + board_thickness / 2.0,
            height - board_thickness / 2.0 - shift_top,
            num_boards,
        ):
            if h == shift_bottom + board_thickness / 2.0 and not bottom_board:
                continue

            boards.append(
                trimesh.primitives.Box(
                    extents=[width, depth, board_thickness],
                    transform=tra.translation_matrix([0, 0, h]),
                )
            )
            board_names.append(f"board_{cnt}")
            cnt += 1

            min_z = min(min_z, h)
            max_z = max(max_z, h)

        cnt = 0
        for v in np.linspace(-width / 2.0, width / 2.0, num_vertical_boards + 2)[1:-1]:
            boards.append(
                trimesh.primitives.Box(
                    extents=[board_thickness, depth, max_z - min_z],
                    transform=tra.translation_matrix(
                        [v, 0, min_z + (max_z - min_z) / 2.0]
                    ),
                )
            )
            board_names.append(f"separator_{cnt}")
            cnt += 1

        int_num_side_columns = 1 if np.isinf(num_side_columns) else num_side_columns
        offset = depth / 2.0 if int_num_side_columns == 1 else 0.0
        for j in range(2):
            cnt = 0
            for c in np.linspace(-depth / 2.0, depth / 2.0, int_num_side_columns):
                if cylindrical_columns:
                    column = trimesh.primitives.Cylinder(
                        radius=column_thickness,
                        height=height,
                        transform=tra.translation_matrix(
                            [-width / 2.0 + j * width, c + offset, height / 2.0]
                        ),
                    )
                else:
                    column = trimesh.primitives.Box(
                        extents=[
                            column_thickness,
                            depth if np.isinf(num_side_columns) else column_thickness,
                            height,
                        ],
                        transform=tra.translation_matrix(
                            [-width / 2.0 + j * width, c + offset, height / 2.0]
                        ),
                    )
                boards.append(column)
                board_names.append(f"post_{j}_{cnt}")
                cnt += 1

        scene = trimesh.Scene()
        for mesh, name in zip(boards, board_names):
            scene.add_geometry(
                geometry=mesh,
                geom_name=name,
                node_name=name,
            )

        super().__init__(scene=scene, **kwargs)


DefaultShelf = MyShelfAsset(
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


def set_shelf(
    scene, shelf, x: float, y: float, rotation: bool, name: str, support_name: str
):
    if not (rotation):
        scene.add_object(
            shelf,
            name,
            transform=np.dot(
                tra.translation_matrix((x, y, 0.0)),
                tra.rotation_matrix(np.radians(90), [0, 0, 1]),
            ),
        )
    else:
        scene.add_object(shelf, name, transform=tra.translation_matrix((x, y, 0.0)))
    support_data = scene.label_support(
        label=support_name,
        obj_ids=[name],
        min_area=0.05,
        gravity=np.array([0, 0, -1]),
    )
    return support_data


def add_objects_to_shelf(
    scene,
    cnt_boards: int,
    product_on_board: list[list],
    suf: str,
    cnt_prod_on_board: int,
    support_data,
):
    for num_board in range(cnt_boards):
        for elem in product_on_board[num_board]:
            scene.place_objects(
                obj_id_iterator=utils.object_id_generator(
                    f"{elem}_" + suf + f"_{num_board}_"
                ),
                obj_asset_iterator=tuple(NAMES_OF_PRODUCTS[elem] for _ in range(cnt_prod_on_board)),
                # obj_support_id_iterator=scene.support_generator(f'support{cnt}'),
                obj_support_id_iterator=utils.cycle_list(support_data, [num_board]),
                obj_position_iterator=utils.PositionIteratorGrid(
                    step_x=0.2,
                    step_y=0.1,
                    noise_std_x=0.01,
                    noise_std_y=0.01,
                    direction="x",
                ),
                obj_orientation_iterator=utils.orientation_generator_uniform_around_z(),
            )



def try_shelf_placement(
        darkstore: list[list],
        is_rotate: list[list],
        random_shelfs: list[list[list]],
        is_showed: bool = False):
    n, m = len(darkstore), len(darkstore[0])
    cells = []
    for i in range(n):
        for j in range(m):
            if darkstore[i][j]:
                cells.append(i * m + j)
    scene = synth.Scene()
    shelf = DefaultShelf
    cnt = 0
    it = 0
    for x in range(n):
        for y in range(m):
            if not darkstore[x][y]:
                cnt += 1
                continue
            support_data = set_shelf(
                scene,
                shelf,
                x * 1.55,
                y * 1.55,
                is_rotate[x][y],
                f"shelf{cnt}",
                f"support{cnt}",
            )
            add_objects_to_shelf(
                scene,
                BOARDS,
                random_shelfs[it],
                str(cnt),
                COUNT_OF_PRODUCT_ON_BOARD,
                support_data,
            )
            cnt += 1
            it += 1

    if is_showed:
        scene.colorize()
        scene.colorize(specific_objects={f"shelf{i}": [123, 123, 123] for i in cells})
        scene.show()
    json_str = synth.exchange.export.export_json(scene, include_metadata=False)

    data = json.loads(json_str)
    del data["geometry"]
    data["meta"] = {"n": n, "m": m}

    with open(f"myscene_{n}_{m}.json", "w") as f:
        # f.write(json_str)
        json.dump(data, f, indent=4)


def try_one_shelf_placement_with(products_on_boards: list[list]):
    scene = synth.Scene()
    shelf = DefaultShelf
    support_data = set_shelf(
        scene,
        shelf,
        0,
        0,
        False,
        f"shelf",
        f"support",
    )
    add_objects_to_shelf(
        scene,
        BOARDS,
        products_on_boards,
        'try',
        COUNT_OF_PRODUCT_ON_BOARD,
        support_data,
    )
    scene.colorize()
    scene.show()

def try_one_shelf_placement_with_diff_of_one_board(
    set_of_products_on_each_boards: list[tuple],
    suf: str = 'diff'
    ):
    scene = synth.Scene()
    shelf = DefaultShelf
    support_data = set_shelf(
        scene,
        shelf,
        0,
        0,
        False,
        f"shelf",
        f"support",
    )
    for num_board in range(BOARDS):
        scene.place_objects(
            obj_id_iterator=utils.object_id_generator(
                f"products_" + suf + f"_{num_board}_"
            ),
            obj_asset_iterator=set_of_products_on_each_boards[num_board],
            # obj_support_id_iterator=scene.support_generator(f'support{cnt}'),
            obj_support_id_iterator=utils.cycle_list(support_data, [num_board]),
            obj_position_iterator=utils.PositionIteratorGrid(
                step_x=0.2,
                step_y=0.1,
                noise_std_x=0.01,
                noise_std_y=0.01,
                direction="x",
            ),
            obj_orientation_iterator=utils.orientation_generator_uniform_around_z(),
        )
    scene.colorize()
    scene.show()

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

    args = parser.parse_args()

    with open(args.input, "r") as f:
        data = json.load(f)

    n, m = data["room_size"]
    x, y = data["door_coords"]

    mat = data.get("blocked_matrix", [[0] * m for _ in range(n)])
    name_to_cnt = {'milk': 5, 'baby': 3, 'cereal': 2}

    is_gen, room = add_many_products((x, y), mat, name_to_cnt)
    is_rotate = get_orientation((x, y), room)

    if not is_gen:
        raise UserError("retry to generate a scene")

    try_shelf_placement(room, is_rotate, data['random_shelfs'], args.show)
