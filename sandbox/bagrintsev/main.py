import scene_synthesizer as synth
import scene_synthesizer.procedural_assets as pa
import json

# create procedural assets
table = synth.procedural_assets.TableAsset(width=1.2, depth=0.8, height=0.75)
shelf = synth.procedural_assets.ShelfAsset(width=1.2, depth=0.8, height=2, num_boards=6)
bowl = pa.BowlAsset(radius=0.15, height=0.07)
#print(dir(synth.procedural_assets))
cabinet = synth.procedural_assets.CabinetAsset(
    width=0.5,
    height=0.5,
    depth=0.4,
    compartment_mask=[[0], [1]],
    compartment_types=["drawer", "drawer"],
)
#print(dir(synth.procedural_assets))
# load asset from file
# Make sure to first download the file:
# wget https://raw.githubusercontent.com/clemense/kitchen-assets-cc-by/refs/heads/main/assets/chair/meshes/chair.{mtl,obj}
# chair = synth.Asset("chair.obj", up=(0, 0, 1), front=(-1, 0, 0))

# create scene
scene = synth.Scene()
# add table to scene
# scene.add_object(table)
scene.add_object(shelf)
scene.label_support('shelf_surface', obj_ids='shelf')
scene.add_object(shelf, connect_parent_anchor=('right', 'front', 'bottom'), connect_obj_anchor=('left', 'front', 'bottom'))
milk = synth.Asset('textures/milk_carton.glb', up=(0, 1, 0))

# scene.add_object(cabinet, connect_parent_anchor=('right', 'front', 'bottom'), connect_obj_anchor=('left', 'front', 'bottom'))
# scene.label_support('table_surface', obj_ids='table')
# for i in range(5):
#     scene.place_object(f'plate{i}', synth.procedural_assets.PlateAsset(), support_id='table_surface')
for i in range(25):
    scene.place_object(f'milk{i}', milk, support_id='shelf_surface')
scene.colorize()
scene.show()
