import os
import glob
import math
import shutil
import re
import random


ROOT = '/home/charlie'

BATCH_SIZE = 40
images_dir = os.path.join(ROOT,'forAnnotation')
image_paths = glob.glob(f"{images_dir}/all_images/*")

# shuffle image paths so the photos are not sorted by year
random.shuffle(image_paths)

all_images_dir = os.path.join(ROOT, 'forAnnotation', 'all_images')

def get_last_batch_num(images_dir):
    list_dir_in_dir = [x[0].split('/')[-1] for x in os.walk(images_dir)]
    get_only_batch_names = []

    for element in list_dir_in_dir:
        z = re.match("batch_....", element)

        if z:
            get_only_batch_names.append(element)

    last_batch_num = max([int(i.split('/')[-1].split('_')[-1].lstrip('0')) for i in get_only_batch_names])
    return last_batch_num

#BATCH_NUM = get_last_batch_num(images_dir)

BATCH_NUM = 21
NUM_BACTHES = math.ceil(len(image_paths)/BATCH_SIZE)
START_BATCH = 0

for i in range(NUM_BACTHES):
    #image_names = [image_path.split('/')[-1] for image_path in image_paths]
    images_to_move = image_paths[START_BATCH:START_BATCH + BATCH_SIZE] 

    batch_num = str(BATCH_NUM+1).zfill(4)
    dir_name = f"batch_{batch_num}"
    batch_dir_name = os.path.join(images_dir, dir_name)
    print(batch_dir_name)

    if not os.path.isdir(batch_dir_name):
        os.mkdir(batch_dir_name)
        for image in images_to_move:
            shutil.move(image, batch_dir_name)
        print(dir_name)
        BATCH_NUM += 1
        START_BATCH += BATCH_SIZE
    else:
        print(dir_name)
        print("EXISTS")


def unpack(images_dir, all_images_dir, folder_names):
    for name in folder_names:
        to_unpack = glob.glob(f"{images_dir}/{name}")
        print(to_unpack)

        for folder in to_unpack:
            images = glob.glob(f"{folder}/*")
            for image in images:
                if os.path.exists(f"{all_images_dir}/{image.split('/')[-1]}"):
                    print(f"{image} already in destination folder")
                else:
                    print(f"Moving {image} to {all_images_dir}")
                    shutil.move(image, all_images_dir)


