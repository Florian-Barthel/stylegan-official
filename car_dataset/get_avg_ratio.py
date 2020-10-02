from tqdm import tqdm

import os
from PIL import Image

src_folder = 'E:/carswithcolors/ratio_exif'


avg_ratio = 0.0
count = 0

for file in tqdm(os.listdir(src_folder)):
    src_img = src_folder + '/' + file
    count += 1
    img = Image.open(src_img)
    w, h = img.size
    avg_ratio += w / h

print(count)
print(avg_ratio / count)

