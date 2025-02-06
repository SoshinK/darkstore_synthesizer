import scene_synthesizer as synth
import scene_synthesizer.procedural_assets as pa
import json

# create procedural assets
table = synth.procedural_assets.TableAsset(width=1.2, depth=0.8, height=0.75)
cabinet = synth.procedural_assets.CabinetAsset(width=0.5, height=0.5, depth=0.4, compartment_mask=[[0], [1]], compartment_types=['drawer','drawer'])

# load asset from file
# Make sure to first download the file:
# wget https://raw.githubusercontent.com/clemense/kitchen-assets-cc-by/refs/heads/main/assets/chair/meshes/chair.{mtl,obj}
chair = synth.Asset('textures/chair.obj', up=(0, 0, 1), front=(-1, 0, 0))
milk = synth.Asset('textures/milk_carton.glb', up=(0, 1, 0))

# create scene
scene = synth.Scene()

# add table to scene
scene.add_object(table)
# put cabinet next to table
scene.add_object(cabinet, connect_parent_anchor=('right', 'front', 'bottom'), connect_obj_anchor=('left', 'front', 'bottom'))
# put chair in front of table
scene.add_object(chair, connect_parent_id='table', connect_parent_anchor=('center', 'front', 'bottom'), connect_obj_anchor=('center', 'center', 'bottom'))

# randomly place plate and glass on top of table
scene.label_support('table_surface', obj_ids='table')
scene.place_object('plate', synth.procedural_assets.PlateAsset(), support_id='table_surface')

scene.place_object('milk_0', milk, support_id='table_surface')

scene.colorize()



# obj_output = synth.exchange.export.export_obj(scene)

# with open("scene.obj", "w") as f:
#     f.write(obj_output)


# fname = 'obj.urdf'
# urdf_output = synth.exchange.export.export_urdf(scene, fname)
# print(urdf_output)


# preview scene in an OpenGL window
scene.show()

# export scene in various formats
