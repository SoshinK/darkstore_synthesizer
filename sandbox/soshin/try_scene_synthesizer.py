import numpy as np
import scene_synthesizer as synth
from scene_synthesizer import procedural_assets as pa
from scene_synthesizer import procedural_scenes as ps
from scene_synthesizer import utils
import trimesh.transformations as tra
import json 
def try1():
    table = pa.TableAsset(
        width=1.0,
        depth=1.0,
        height=0.75,
        thickness=0.025,
        leg_thickness=0.025
        )

    table.show(use_collision_geometry=False)


def try2():
    my_scene = synth.Scene()

    table = pa.TableAsset(1, 1, 0.75)

    my_scene.add_object(
        asset=table,
        obj_id="table",
        transform=np.eye(4),
    )
    my_scene.show_graph() # show scene graph


def try3():
    table = pa.TableAsset(1, 1, 0.75)
    shelf = pa.ShelfAsset(1, 0.5, 2.0, num_boards=6)

    bowl = pa.BowlAsset()

    s = synth.Scene()
    s.add_object(table, 'table')

    s.add_object(
        shelf,
        'shelf',
        connect_parent_id='table',
        connect_parent_anchor=('right', 'back', 'bottom'),
        connect_obj_anchor=('left', 'back', 'bottom')
    )
    
    s.add_object(
        bowl,
        'bowl',
        connect_parent_id='table',
        connect_parent_anchor=('center', 'back', 'top'),
        connect_obj_anchor=('center', 'back', 'bottom')
    )

    s.colorize() # make it a bit more colorful

    s.show() # show scene in trimesh visualizer

def kitchen():
    kitchen = ps.kitchen_island()
    kitchen.label_support(
        label="support",
        min_area=0.05,
        gravity=np.array([0, 0, -1]),
    )

    kitchen.show_supports()

def table():
    table_scene = pa.TableAsset(1.0, 1.4, 0.7).scene('table')
    table_scene.label_support(label="support")

    table_scene.show_supports()

def placement():
    table_scene = pa.TableAsset(1.0, 1.4, 0.7).scene('table')
    table_scene.label_support(label="support")

    mug = pa.MugAsset(
        # this is needed since the place_object method
        # will only sample a point on the surface
        origin=("com", "com", "bottom"),
    )

    table_scene.place_object(
        obj_id="mug",
        obj_asset=mug,
        support_id="support",  # this is a reference to
        # a surface that was labelled via label_support
    )
    table_scene.colorize()
    table_scene.show()

def many_placements():
    table_scene = pa.TableAsset(1.0, 1.4, 0.7).scene('table')
    table_scene.label_support(label="support")

    mug = pa.MugAsset(
        # this is needed since the place_object method
        # will only sample a point on the surface
        origin=("com", "com", "bottom"),
    )

    for i in range(50):
        table_scene.place_object(
            obj_id=f"mug{i}",
            obj_asset=mug,
            support_id="support",
            obj_orientation_iterator=utils.orientation_generator_stable_poses(mug),
        )

    table_scene.colorize()
    table_scene.show()

def distr():
    obj_position_iterator = [
        utils.PositionIteratorUniform(),
        utils.PositionIteratorGaussian(params=[0, 0, 0.08, 0.08]), 
        utils.PositionIteratorPoissonDisk(k=30, r=0.1),
        utils.PositionIteratorGrid(step_x=0.02, step_y=0.02, noise_std_x=0.04, noise_std_y=0.04),
        utils.PositionIteratorGrid(step_x=0.2, step_y=0.02, noise_std_x=0.0, noise_std_y=0.0),
        utils.PositionIteratorFarthestPoint(sample_count=1000),
    ]

    mug = pa.MugAsset(origin=('com', 'com', 'bottom'))
    table = pa.TableAsset(1.0, 1.4, 0.7)

    s = synth.Scene()

    cnt = 0
    for x in range(3):
        for y in range(2):
            s.add_object(table, f'table{cnt}', transform=tra.translation_matrix((x * 1.5, y * 1.5, 0.0)))
            s.label_support(f'support{cnt}', obj_ids=[f'table{cnt}'])
            s.place_objects(
                obj_id_iterator=utils.object_id_generator(f"Mug{cnt}_"),
                obj_asset_iterator=(mug for _ in range(20)),
                obj_support_id_iterator=s.support_generator(f'support{cnt}'),
                obj_position_iterator=obj_position_iterator[cnt],
                obj_orientation_iterator=utils.orientation_generator_uniform_around_z(),
            )
            cnt += 1

    s.colorize()
    s.colorize(specific_objects={f'table{i}': [123, 123, 123] for i in range(6)})
    s.export('shit.glb')
    s.export('shit_urdf/shit.urdf')
    json_str = synth.exchange.export.export_json(s, include_metadata=False)


    with open('shit.json', 'w') as f:
        # f.write(json_str)
        json.dump(json.loads(json_str), f, indent=4)
    # s.show()

def placement_can():
    table_scene = pa.TableAsset(1.0, 1.4, 0.7).scene('table')
    table_scene.label_support(label="support")

    # can = pa.MugAsset(
    #     # this is needed since the place_object method
    #     # will only sample a point on the surface
    #     origin=("com", "com", "bottom"),
    # )
    can = synth.assets.MeshAsset(
        '/home/kvsoshin/Work/AIRI/ManiSkill/simple_cola_can.glb',
        scale=0.05,
        origin=("bottom", "bottom", "bottom"),
        transform=np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ]),

    )

    table_scene.place_object(
        obj_id="can",
        obj_asset=can,
        support_id="support",  # this is a reference to
        # a surface that was labelled via label_support
    )
    table_scene.colorize()
    table_scene.show()

def many_placements_cans():
    table_scene = pa.TableAsset(1.0, 1.4, 0.7).scene('table')
    table_scene.label_support(label="support")

    can = synth.assets.MeshAsset(
        '/home/kvsoshin/Work/AIRI/ManiSkill/simple_cola_can.glb',
        scale=0.05,
        origin=("com", "com", "bottom"),
    )

    for i in range(20):
        table_scene.place_object(
            obj_id=f"can{i}",
            obj_asset=can,
            support_id="support",
            obj_orientation_iterator=utils.orientation_generator_uniform_around_z(),
        )

    table_scene.colorize()
    table_scene.export('cans.obj')

    json_str = synth.exchange.export.export_json(table_scene, include_metadata=False)

    with open('cans.json', 'w') as f:
        # f.write(json_str)
        json.dump(json.loads(json_str), f, indent=4)
    table_scene.show()

def many_placements_milks():
    table_scene = pa.TableAsset(1.0, 1.4, 0.7).scene('table')
    table_scene.label_support(label="support")

    can = synth.assets.MeshAsset(
        '/home/kvsoshin/Work/AIRI/ManiSkill/untitled.glb',
        scale=0.1,
        origin=("com", "com", "bottom"),
        # transform=np.array([
        #     [1, 0, 0, 0],
        #     [0, 0, 1, 0],
        #     [0, 1, 0, 0],
        #     [0, 0, 0, 1]
        # ]),
    )
    obj_position_iterator = [
        utils.PositionIteratorUniform(),
        utils.PositionIteratorGaussian(params=[0, 0, 0.08, 0.08]), 
        utils.PositionIteratorPoissonDisk(k=30, r=0.1),
        utils.PositionIteratorGrid(step_x=0.02, step_y=0.02, noise_std_x=0.04, noise_std_y=0.04),
        utils.PositionIteratorGrid(step_x=0.2, step_y=0.02, noise_std_x=0.0, noise_std_y=0.0),
        utils.PositionIteratorFarthestPoint(sample_count=1000),
    ]

    for i in range(20):
        table_scene.place_object(
            obj_id=f"milk{i}",
            obj_asset=can,
            support_id="support",
            obj_orientation_iterator=utils.orientation_generator_uniform_around_z(),
            obj_position_iterator=obj_position_iterator[1],
        )

    table_scene.colorize()
    # table_scene.export('cans.obj')

    json_str = synth.exchange.export.export_json(table_scene, include_metadata=False)

    with open('milks.json', 'w') as f:
        # f.write(json_str)
        json.dump(json.loads(json_str), f, indent=4)
    table_scene.show()

def many_placements_apples():
    table_scene = pa.TableAsset(1.0, 1.4, 0.7).scene('table')
    table_scene.label_support(label="support")

    can = synth.assets.MeshAsset(
        '/home/kvsoshin/Work/AIRI/ManiSkill/apple.glb',
        scale=0.002,
        origin=("com", "com", "bottom"),
        # transform=np.array([
        #     [1, 0, 0, 0],
        #     [0, 0, 1, 0],
        #     [0, 1, 0, 0],
        #     [0, 0, 0, 1]
        # ]),
    )
    obj_position_iterator = [
        utils.PositionIteratorUniform(),
        utils.PositionIteratorGaussian(params=[0, 0, 0.08, 0.08]), 
        utils.PositionIteratorPoissonDisk(k=30, r=0.1),
        utils.PositionIteratorGrid(step_x=0.02, step_y=0.02, noise_std_x=0.04, noise_std_y=0.04),
        utils.PositionIteratorGrid(step_x=0.2, step_y=0.02, noise_std_x=0.0, noise_std_y=0.0),
        utils.PositionIteratorFarthestPoint(sample_count=1000),
    ]

    for i in range(10):
        table_scene.place_object(
            obj_id=f"can{i}",
            obj_asset=can,
            support_id="support",
            # obj_orientation_iterator=utils.orientation_generator_uniform_around_z(),
            obj_position_iterator=obj_position_iterator[1],
        )

    table_scene.colorize()
    # table_scene.export('cans.obj')

    json_str = synth.exchange.export.export_json(table_scene, include_metadata=False)

    with open('apples.json', 'w') as f:
        # f.write(json_str)
        json.dump(json.loads(json_str), f, indent=4)
    table_scene.show()


def try_kitchen_scene():
    scene = ps.kitchen_galley(seed=69)
    scene.colorize()
    scene.show()

def try_shelf_asset():
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

    synth.exchange.export.export_urdf(scene, 'shelf.urdf')
    scene.export('shelf.obj')
    scene.show()

import trimesh
from scene_synthesizer.assets import TrimeshSceneAsset

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
    # print(support_data)
    # print("====")
    # print(scene.metadata["support_polygons"])
    milk = synth.assets.MeshAsset(
        '/home/kvsoshin/Work/AIRI/3d_assets/milk_karton_processed.glb',
        scale=1.0,
        origin=("com", "com", "bottom"),
    )
    # milk2 = synth.assets.MeshAsset(
    #     '/home/kvsoshin/Work/AIRI/3d_assets/milk_processed.glb',
    #     scale=1.0,
    #     origin=("com", "com", "bottom"),
    # )

    obj_position_iterator = [
            utils.PositionIteratorUniform(),
            utils.PositionIteratorGaussian(params=[0, 0, 0.08, 0.08]), 
            utils.PositionIteratorPoissonDisk(k=30, r=0.1),
            utils.PositionIteratorGrid(step_x=0.02, step_y=0.02, noise_std_x=0.04, noise_std_y=0.04),
            utils.PositionIteratorGrid(step_x=0.2, step_y=0.02, noise_std_x=0.0, noise_std_y=0.0),
            utils.PositionIteratorFarthestPoint(sample_count=1000),
        ]

    id_iterator = utils.cycle_list(support_data, [3])
    for i in range(10):
        scene.place_object(
            obj_id=f"milk_1{i}",
            obj_support_id_iterator=id_iterator,
            obj_asset=milk,
            support_id='support',
            obj_orientation_iterator=utils.orientation_generator_uniform_around_z(),
            obj_position_iterator=obj_position_iterator[1],
        )

    for i in range(10):
        scene.place_object(
            obj_id=f"milk_2{i}",
            obj_support_id_iterator=utils.cycle_list(support_data, [2]),
            obj_asset=milk,
            support_id='support',
            obj_orientation_iterator=utils.orientation_generator_uniform_around_z(),
            obj_position_iterator=obj_position_iterator[4],
        )
    for i in range(10):
        scene.place_object(
            obj_id=f"milk_3{i}",
            obj_support_id_iterator=utils.cycle_list(support_data, [1]),
            obj_asset=milk,
            support_id='support',
            obj_orientation_iterator=utils.orientation_generator_uniform_around_z(),
            obj_position_iterator=obj_position_iterator[4],
        )

    for i in range(10):
        scene.place_object(
            obj_id=f"milk_4{i}",
            obj_support_id_iterator=utils.cycle_list(support_data, [0]),
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