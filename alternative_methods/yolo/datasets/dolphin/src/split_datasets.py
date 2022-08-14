from config import LABELS_DIR, IMAGES_DIR, ROOT_DIR

import random
import glob
import os
import shutil

def copyfiles(fil, root_dir):
    basename = os.path.basename(fil)
    filename = os.path.splitext(basename)[0]
    
    # image
    src = fil
    dest = os.path.join(root_dir, image_dir, f"{filename}.jpg")
    shutil.copyfile(src, dest)  

    # label
    src = os.path.join(LABELS_DIR, f"{filename}.txt")
    dest = os.path.join(root_dir, label_dir, f"{filename}.txt")
    if os.path.exists(src):
        shutil.copyfile(src, dest)

label_dir = "labels/"
image_dir = "images/"
lower_limit = 0
image_files = glob.glob(os.path.join("data", image_dir, '*.jpg'))
label_files = glob.glob(os.path.join("data", label_dir, '*.xml'))


random.Random(4).shuffle(image_files)

folders = {"train": 0.8, "val": 0.1, "test": 0.1}
check_sum = sum([folders[x] for x in folders])

assert check_sum == 1.0, "Split proportion is not equal to 1.0"

for folder in folders:
    set_dir = f"{ROOT_DIR}/data/{folder}"
    os.mkdir(set_dir)
    temp_label_dir = os.path.join(set_dir, "labels")
    os.mkdir(temp_label_dir)
    temp_image_dir = os.path.join(set_dir, "images")
    os.mkdir(temp_image_dir)    
    
    limit = round(len(image_files) * folders[folder])
    for fil in image_files[lower_limit:lower_limit + limit]:
        copyfiles(fil, set_dir)
    lower_limit = lower_limit + limit

    # folder_images = glob.glob(f"{temp_image_dir}/*.jpg")
    # image_names = [image.split('/')[-1].split(".")[0] for image in folder_images]
    # for image in range(len(image_names)):
    #     print(f"{LABELS_DIR}/{image_names[image]}.txt", set_dir)
    #     copyfiles(f"{LABELS_DIR}/{image_names[image]}.txt", set_dir)
    
