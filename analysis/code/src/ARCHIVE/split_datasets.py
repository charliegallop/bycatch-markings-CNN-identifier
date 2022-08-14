# https://towardsdatascience.com/convert-pascal-voc-xml-to-yolo-for-object-detection-f969811ccba5

from config import DATA_DIR, TRAIN_FOR, MASTER_DIR
import random
import glob
import os
import shutil

data_dir = DATA_DIR
master_dir = MASTER_DIR
lower_limit = 0

set_list = {"train": 0.8, "val": 0.1, "test": 0.1}

# Select all image paths from master directory
image_paths = glob.glob(os.path.join(master_dir, "images", '*.jpg'))

# Shuffle paths
random.Random(4).shuffle(image_paths)

check_sum = sum([set_list[x] for x in set_list])
assert check_sum == 1.0, "Split proportion is not equal to 1.0"

def copyfiles(fil, set_dir):
    basename = os.path.basename(fil)
    filename = os.path.splitext(basename)[0]
    
    # image
    src = fil
    dest = os.path.join(set_dir, "images", f"{filename}.jpg")
    shutil.copyfile(src, dest)  

    # label
    src = os.path.join(MASTER_DIR, "labels", f"{filename}.xml")
    dest = os.path.join(set_dir, "labels", f"{filename}.xml")
    if os.path.exists(src):
        shutil.copyfile(src, dest)

def check_dir_already_exists(sets = {"train": 0.8, "val": 0.1, "test": 0.1}):
    safe_to_create_dir = False
    existing_dir = []
    for s in sets:
        dir_path = os.path.join(DATA_DIR, s)
        if os.path.exists(dir_path):
            user_input = input(f"Remove '{dir_path}' directory and contents? [y][n]: ").lower()
            if user_input == "y":
                shutil.rmtree(dir_path)
            else:
                print("Did not delete directory...")
                print("Not safe to create new directories...Quitting...")
                quit()
        else:
            print(f"'{s}' directory does not already exist...")

    safe_to_create_dir = True
        
    if safe_to_create_dir:
        print("Safe to create new directories")
        return safe_to_create_dir
    else:
        print("Not safe to create new directories")
        return safe_to_create_dir

def create_set_directories(se):
    # se: the set folder to create
    set_dir = os.path.join(DATA_DIR, se)
    os.mkdir(set_dir)
    set_label_dir = os.path.join(set_dir, "labels")
    set_image_dir = os.path.join(set_dir, "images")
    os.mkdir(set_label_dir)
    os.mkdir(set_image_dir)

def split_files(image_paths, lower_limit, set_list):
    for se in set_list:
        limit = round(len(image_paths) * set_list[se])
        set_dir = os.path.join(DATA_DIR, se)
        for image in image_paths[lower_limit:lower_limit + limit]:
            copyfiles(image, set_dir)
        lower_limit = lower_limit + limit


def main():
    if check_dir_already_exists():
        print("SAFE TO PROCEED")
        for se in set_list:
            print("EXECUTING CREATE")
            create_set_directories(se)
        print("SPLIT CREATE")

        split_files(image_paths, lower_limit, set_list)

    else:
        print("Directories already exist")


    
