from config import TEST_DIR, EVAL_DIR, TRAIN_FOR, BACKBONE
from xml.etree import ElementTree as et
import glob
import os

def keep_labels(WRITE_TO, label_dir, label_to_keep = 'dolphin'):


    ANNOT_DIR = label_dir
    # WRITE_TO = os.path.join(EVAL_DIR, TRAIN_FOR.value(), BACKBONE.value(), "gt")
    annot_paths_dir = os.path.join(ANNOT_DIR)
    print("ANNOT_PATHS_DIR: ", annot_paths_dir)
    ANNOT_PATHS = glob.glob(annot_paths_dir)
    print(ANNOT_PATHS)

    if label_to_keep == 'dolphin':
        print("Making XML files for 'dolphin'....")
        for i in ANNOT_PATHS:
            print(i)
            image_name = os.path.basename(i)
            tree = et.parse(i)
            root = tree.getroot()
            for member in root.findall('object'):
                if member.find('name').text != 'dolphin':
                    root.remove(member)
            save_as = os.path.join(WRITE_TO, image_name)
            tree.write(save_as)
        print("XML files saved to: ", WRITE_TO)

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
            tree.write(save_as)
        print("XML files saved to: ", WRITE_TO)
        
    else:
        print("NOT A VALID LABEL")

#keep_labels('markings')
#WRITE_TO = os.path.join(EVAL_DIR, "dolphin", "mobilenet_320", "gt")
#keep_labels(WRITE_TO, 'dolphin')