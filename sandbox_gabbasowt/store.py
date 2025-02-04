import numpy as np
import scene_synthesizer as synth
from scene_synthesizer import procedural_assets as pa
from scene_synthesizer import procedural_scenes as ps
from scene_synthesizer import utils
import trimesh.transformations as tra
import json
import trimesh
from scene_synthesizer.assets import TrimeshSceneAsset

TRY = 10

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


def try_shelf_placement():
    scene = synth.Scene()

    shelf = MyShelfAsset(
        width=1.517,
        depth=0.5172,
        height=2.0,
        board_thickness=0.05135,
        num_boards=5,
        num_side_columns=2,
        bottom_board=True,
        cylindrical_columns=False,
        num_vertical_boards=0,
        shift_bottom=0.131952 - 0.05135 / 2,
        shift_top=0.2288 + 0.05135 / 2
    )
    scene.add_object(shelf, 'shelf')

    support_data = scene.label_support(
        label="support",
        min_area=0.05,
        gravity=np.array([0, 0, -1]),
    )
    milk = synth.assets.MeshAsset(
        '/home/timur/dark/sandbox_gabbasowt/milk.glb',
        scale=1.1,
        origin=("com", "com", "bottom"),
    )

    obj_position_iterator = [
        utils.PositionIteratorUniform(),
        utils.PositionIteratorGaussian(params=[0, 0, 0.08, 0.08]),
        utils.PositionIteratorPoissonDisk(k=30, r=0.1),
        utils.PositionIteratorGrid(step_x=0.02, step_y=0.02, noise_std_x=0.04, noise_std_y=0.04),
        utils.PositionIteratorGrid(step_x=0.2, step_y=0.02, noise_std_x=0.0, noise_std_y=0.0),
        utils.PositionIteratorFarthestPoint(sample_count=1000),
    ]

    for i in range(TRY):
        scene.place_object(
            obj_id=f"milk_1{i}",
            obj_support_id_iterator=utils.cycle_list(support_data, [0]),
            obj_asset=milk,
            support_id='support',
            obj_orientation_iterator=utils.orientation_generator_uniform_around_z(),
            obj_position_iterator=obj_position_iterator[4],
        )

    for i in range(TRY):
        scene.place_object(
            obj_id=f"milk_2{i}",
            obj_support_id_iterator=utils.cycle_list(support_data, [1]),
            obj_asset=milk,
            support_id='support',
            obj_orientation_iterator=utils.orientation_generator_uniform_around_z(),
            obj_position_iterator=obj_position_iterator[4],
        )
    for i in range(TRY):
        scene.place_object(
            obj_id=f"milk_3{i}",
            obj_support_id_iterator=utils.cycle_list(support_data, [2]),
            obj_asset=milk,
            support_id='support',
            obj_orientation_iterator=utils.orientation_generator_uniform_around_z(),
            obj_position_iterator=obj_position_iterator[4],
        )

    for i in range(TRY):
        scene.place_object(
            obj_id=f"milk_4{i}",
            obj_support_id_iterator=utils.cycle_list(support_data, [3]),
            obj_asset=milk,
            support_id='support',
            obj_orientation_iterator=utils.orientation_generator_uniform_around_z(),
            obj_position_iterator=obj_position_iterator[4],
        )

    json_str = synth.exchange.export.export_json(scene, include_metadata=False)

    with open('shelf.json', 'w') as f:
        # f.write(json_str)
        json.dump(json.loads(json_str), f, indent=4)

    scene.colorize()
    scene.show()


if __name__ == '__main__':
    try_shelf_placement()