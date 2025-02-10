import json
import random

COUNT_OF_SHELFS = 10
COUNT_OF_BOARDS = 5
COUNT_OF_PRODUCTS_ON_ONE_BOARD = 5

input_data = {
    'room_size': [5, 5],
    'door_coords': [0, 0],
    'blocked_matrix': [[0 for _ in range(5)] for e in range(5)]
}

with open('models/assets.json', 'r') as f:
    assets = json.load(f)

if isinstance(assets, dict):
    assets = list(assets.keys())

del assets[assets.index('cereal')]

shelfs = []
for i in range(COUNT_OF_SHELFS):
    shelfs.append([
        tuple(random.sample(assets, COUNT_OF_PRODUCTS_ON_ONE_BOARD)) for _ in range(COUNT_OF_BOARDS)
    ])

input_data['random_shelfs'] = shelfs

with open('models/input.json', 'w', encoding='utf-8') as file:
    json.dump(input_data, file, ensure_ascii=False, indent=4)
