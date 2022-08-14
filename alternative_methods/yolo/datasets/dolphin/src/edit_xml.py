from config import TEST_DIR, EVAL_DIR, TRAIN_FOR, BACKBONE
from xml.etree import ElementTree as et
import glob
import os

ANNOT_DIR = os.path.join(TEST_DIR, "labels")
# WRITE_TO = os.path.join(EVAL_DIR, TRAIN_FOR.value(), BACKBONE.value(), "gt")
ANNOT_PATHS = glob.glob(f"{ANNOT_DIR}/*")

def keep_labels(WRITE_TO, label_to_keep = 'dolphin'):

    if label_to_keep == 'dolphin':
        print("Making XML files for 'dolphin'....")
        print(ANNOT_PATHS)

        for i in ANNOT_PATHS:
            image_name = os.path.basename(i)
            tree = et.parse(i)
            root = tree.getroot()
            for member in root.findall('object'):
                if member.find('name').text != 'dolphin':
                    root.remove(member)
            save_as = os.path.join(WRITE_TO, image_name)
            print("SAVING TO: ", save_as)
            tree.write(save_as)

    elif label_to_keep == 'markings':
        print("Making XML files for 'markings'....")
        for i in ANNOT_PATHS:
            image_name = os.path.basename(i)
            tree = et.parse(i)
            root = tree.getroot()

            for member in root.findall('object'):
                if member.find('name').text == 'dolphin':
                    root.remove(member)

            save_as = os.path.join(WRITE_TO, image_name)
            print("SAVING TO: ", save_as)
            tree.write(save_as)
    else:
        print("NOT A VALID LABEL")

#keep_labels('markings')
WRITE_TO = os.path.join(EVAL_DIR, "dolphin", "mobilenet_320", "gt")
keep_labels(WRITE_TO, 'dolphin')