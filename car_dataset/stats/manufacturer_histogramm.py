from tqdm import tqdm
import os
import json

src_folder = 'E:/carswithcolors/trainA'

with open('E:/carswithcolors/trainA/data.json') as json_file:
    data = json.load(json_file)

manufacturers = data['labels'][3]['classes']
colors = data['labels'][1]['classes']

count_manufacturer = {}
count_colors = {}

for manufacturer in manufacturers:
    count_manufacturer[manufacturer] = 0

for color in colors:
    count_colors[color] = 0

for folder in tqdm(os.listdir(src_folder)):
    if not folder.endswith('.json'):
        for json_file_name in os.listdir(src_folder + '/' + folder):
            if json_file_name.endswith('.json'):
                with open(src_folder + '/' + folder + '/' + json_file_name) as json_file:
                    car_data = json.load(json_file)
                    color = colors[car_data['labels']['color']]
                    manufacturer = manufacturers[car_data['labels']['manufacturer']]
                    count_manufacturer[manufacturer] += 1
                    count_colors[color] += 1


print(count_colors)
print(count_manufacturer)
