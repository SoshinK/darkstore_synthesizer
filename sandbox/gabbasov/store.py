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

#CONST
COUNT_OF_PRODUCT_ON_SHELF = 20
BOARDS = 5
ASSETS_PATH = f'{os.path.dirname(os.path.abspath(__file__))}/../../models'
NAMES_OF_PRODUCTS = {'milk' : synth.assets.MeshAsset(
        f'{ASSETS_PATH}/milk.glb',
        scale=1.1, origin=("com", "com", "bottom"),),
                    'cereal' : synth.assets.MeshAsset(
        f'{ASSETS_PATH}/cereals.glb',
        scale=1, origin=("com", "com", "bottom"),),
                    'coke' : synth.assets.USDAsset(
        f'{ASSETS_PATH}/coke.usdc',
        scale=1,
        origin=("com", "com", "bottom"),),
                    'baby' : synth.assets.USDAsset(
        f'{ASSETS_PATH}/baby.usdc',
        scale=1, origin=("com", "com", "bottom"),),
                    'banana' : synth.assets.USDAsset(
        f'{ASSETS_PATH}/banana.usdc',
        scale=1.5, origin=("com", "com", "bottom"),),
                    'CokeBottle' : synth.assets.USDAsset(
        f'{ASSETS_PATH}/CokeBottle.usdc',
        scale=0.9, origin=("com", "com", "bottom"),),
        }

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
            board_names.append('back')

        min_z = +float("inf")
        max_z = -float("inf")
        cnt = 0
        for h in np.linspace(
                shift_bottom + board_thickness / 2.0, height - board_thickness / 2.0 - shift_top, num_boards
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
                    transform=tra.translation_matrix([v, 0, min_z + (max_z - min_z) / 2.0]),
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


def try_shelf_placement(darkstore, is_rotate):
    n, m = len(darkstore), len(darkstore[0])
    cells = []
    for i in range(n):
        for j in range(m):
            if darkstore[i][j]:
                cells.append(i * m + j)
    scene = synth.Scene()
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
        shift_top=0.2288 + 0.05135 / 2
    )

    obj_position_iterator = [
        utils.PositionIteratorUniform(),
        utils.PositionIteratorGaussian(params=[0, 0, 0.08, 0.08]),
        utils.PositionIteratorPoissonDisk(k=30, r=0.1),
        utils.PositionIteratorGrid(step_x=0.02, step_y=0.02, noise_std_x=0.04, noise_std_y=0.04),
        utils.PositionIteratorGrid(step_x=0.05, step_y=0.005, noise_std_x=0.0, noise_std_y=0.0),
        utils.PositionIteratorFarthestPoint(sample_count=1000),
    ]

    cnt = 0
    for x in range(n):
        for y in range(m):
            if not darkstore[x][y]:
                cnt += 1
                continue
            if is_rotate[x][y]:
                scene.add_object(shelf, f'shelf{cnt}', transform=tra.translation_matrix((x * 1.55, y * 1.55, 0.0)))
            else:
                scene.add_object(shelf, f'shelf{cnt}', transform=np.dot(tra.translation_matrix((x * 1.55, y * 1.55, 0.0)),
                                                                        tra.rotation_matrix(np.radians(90), [0, 0, 1])))
            scene.label_support(f'support{cnt}', obj_ids=[f'shelf{cnt}'])
            scene.place_objects(
                obj_id_iterator=utils.object_id_generator(f"{darkstore[x][y]}{cnt}_"),
                obj_asset_iterator=(NAMES_OF_PRODUCTS[darkstore[x][y]] for _ in range(COUNT_OF_PRODUCT_ON_SHELF)),
                obj_support_id_iterator=scene.support_generator(f'support{cnt}'),
                obj_position_iterator=obj_position_iterator[4],
                obj_orientation_iterator=utils.orientation_generator_uniform_around_z(),
            )
            cnt += 1

    scene.colorize()
    scene.colorize(specific_objects={f'shelf{i}': [123, 123, 123] for i in cells})

    scene.show()

if __name__ == '__main__':
    n, m = map(int, input('Room sizes like "N M":\n').split())
    x, y = map(int, input('Door coordinates like "X Y":\n').split())
    mat = [m * [0] for _ in range(n)]
    have_blocked = int(input('Do you need to block some coords [yes-1/no-0]?\n'))
    if have_blocked:
        print(f'Input matrix {n}X{m} with 1 - blocked, 0 - nothing without whitespaces')
        for i in range(n):
            s = input()
            mat[i] = [int(x) for x in s]

    name_to_cnt = {}
    for i, name in enumerate(NAMES_OF_PRODUCTS.keys()):
        count = int(input(f'Write count of {name}:\n'))
        name_to_cnt[name] = count
    is_gen, room = add_many_products((x, y), mat, name_to_cnt)

    is_rotate = get_orientation((x, y), room)

    if not is_gen:
        raise UserError('retry to generate a scene')

    try_shelf_placement(room, is_rotate)
