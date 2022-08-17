# https://towardsdatascience.com/convert-pascal-voc-xml-to-yolo-for-object-detection-f969811ccba5

from config import DATA_DIR, TRAIN_FOR, MASTER_DIR, RESIZE_TO, EVAL_DIR
import random
import glob
import os
import shutil

data_dir = DATA_DIR
master_dir = MASTER_DIR
lower_limit = 0

# Select all image paths from master directory
image_paths = glob.glob(os.path.join(master_dir, "images", '*.jpg'))

set_list = {"train": 0.8, "val": 0.1, "test": 0.1}


def remove_small_iamges():
    global image_paths
    from PIL import Image
    for image in image_paths:
        img=Image.open(image)
        if (img.size[0] < RESIZE_TO) and (img.size[1] < RESIZE_TO):
            image_paths.remove(image)
            print("REMOVING: ", os.path.basename(image), "since size is: ", img.size)

# Shuffle paths
random.Random(4).shuffle(image_paths)

check_sum = sum([set_list[x] for x in set_list])
assert check_sum == 1.0, "Split proportion is not equal to 1.0"

def copyfiles(fil, set_dir):
    global master_dir
    basename = os.path.basename(fil)
    filename = os.path.splitext(basename)[0]
    
    # image
    src = fil
    dest = os.path.join(set_dir, "images", f"{filename}.jpg")
    shutil.copyfile(src, dest)  

    # label
    src = os.path.join(master_dir, "labels", f"{filename}.xml")
    dest = os.path.join(set_dir, "labels", f"{filename}.xml")
    if os.path.exists(src):
        shutil.copyfile(src, dest)

def check_dir_already_exists(sets = set_list):
    global data_dir
    safe_to_create_dir = False
    existing_dir = []
    for s in sets:
        dir_path = os.path.join(data_dir, s)
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
    global data_dir
    # se: the set folder to create
    set_dir = os.path.join(data_dir, se)
    os.mkdir(set_dir)
    set_label_dir = os.path.join(set_dir, "labels")
    set_image_dir = os.path.join(set_dir, "images")
    os.mkdir(set_label_dir)
    os.mkdir(set_image_dir)

def split_files(lower_limit, set_list):
    global image_paths
    global data_dir
    for se in set_list:
        limit = round(len(image_paths) * set_list[se])
        set_dir = os.path.join(data_dir, se)
        for image in image_paths[lower_limit:lower_limit + limit]:
            copyfiles(image, set_dir)
        lower_limit = lower_limit + limit


def main():
    global data_dir, master_dir, lower_limit
    data_dir = DATA_DIR
    master_dir = MASTER_DIR
    lower_limit = 0

    set_list = {"train": 0.8, "val": 0.1, "test": 0.1}

    if check_dir_already_exists():
        print("SAFE TO PROCEED")
        for se in set_list:
            print("EXECUTING CREATE")
            create_set_directories(se)
        print("SPLIT CREATE")
        remove_small_iamges()
        split_files(lower_limit, set_list)
        from config import BACKBONE
        WRITE_XML_TO = os.path.join(EVAL_DIR, TRAIN_FOR.value(), BACKBONE.value(), 'gt')
        import edit_xml
        edit_xml.keep_labels(WRITE_XML_TO, TRAIN_FOR.value())

    else:
        print("Directories already exist")

def main_markings():
    from config import MASTER_MARKINGS_DIR, MARKINGS_DIR
    global data_dir
    global master_dir
    global lower_limit
    global image_paths

    data_dir = MARKINGS_DIR
    master_dir = os.path.join(MASTER_MARKINGS_DIR)
    lower_limit = 0

    set_list = {"train": 0.8, "val": 0.1, "test": 0.1}

    # Select all image paths from master directory
    image_paths = glob.glob(os.path.join(master_dir, "images", '*.jpg'))

    if check_dir_already_exists():
        print("SAFE TO PROCEED")
        for se in set_list:
            print("EXECUTING CREATE")
            create_set_directories(se)
        print("SPLIT CREATE")
        remove_small_iamges()
        split_files(lower_limit, set_list)
        from config import BACKBONE
        WRITE_XML_TO = os.path.join(EVAL_DIR, TRAIN_FOR.value(), BACKBONE.value(), 'gt')
        import edit_xml
        edit_xml.keep_labels(WRITE_XML_TO, TRAIN_FOR.value())

    else:
        print("Directories already exist")
    
