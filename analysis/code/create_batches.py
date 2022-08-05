import os
import glob
import math
import shutil
import re

ROOT = '/home/charlie'

BACTH_SIZE = 20
images_dir = os.path.join(ROOT,'forAnnotation')
image_paths = glob.glob(f"{images_dir}/*.*")

def get_last_batch_num(images_dir):
    list_dir_in_dir = [x[0].split('/')[-1] for x in os.walk(images_dir)]
    get_only_batch_names = []

    for element in list_dir_in_dir:
        z = re.match("batch_....", element)

        if z:
            get_only_batch_names.append(element)
    
    last_batch_num = max([int(i.split('/')[-1].split('_')[-1].lstrip('0')) for i in get_only_batch_names])
    return last_batch_num

BATCH_NUM = get_last_batch_num(images_dir)

NUM_BACTHES = math.ceil(len(image_paths)/BACTH_SIZE)

for i in range(NUM_BACTHES):
    image_paths = glob.glob(f"{images_dir}/*.*")
    #image_names = [image_path.split('/')[-1] for image_path in image_paths]
    images_to_move = image_paths[:BACTH_SIZE] 

    batch_num = str(BATCH_NUM+1).zfill(4)
    dir_name = f"batch_{batch_num}"
    batch_dir_name = os.path.join(images_dir, dir_name)
    print(batch_dir_name)

    if not os.path.isdir(batch_dir_name):
        os.mkdir(batch_dir_name)
        for image in images_to_move:
            shutil.move(image, batch_dir_name)
        print(dir_name)
    else:
        print(dir_name)
        print("EXISTS")



