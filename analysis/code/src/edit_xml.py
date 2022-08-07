from xml.etree import ElementTree as et
import glob

ANNOT_DIR = "/home/charlie/Documents/Uni/Exeter - Data Science/Dissertation/analysis/data/train_set/valid_annotations_all"
WRITE_TO = "/home/charlie/Documents/Uni/Exeter - Data Science/Dissertation/analysis/data/train_set/valid_predictions/ground_truth_annotations" 
ANNOTATIONS = glob.glob(f"{ANNOT_DIR}/*")

def remove_labels(label_to_keep = 'dolphin'):

    if label_to_keep == 'dolphin':
        for i in ANNOTATIONS:
            image_name = i.split('/')[-1]
            tree = et.parse(i)
            root = tree.getroot()

            for member in root.findall('object'):
                if member.find('name').text != 'dolphin':
                    root.remove(member)

            tree.write(f'{WRITE_TO}/{label_to_keep}/{image_name}')

    elif label_to_keep == 'markings':
        for i in ANNOTATIONS:
            image_name = i.split('/')[-1]
            tree = et.parse(i)
            root = tree.getroot()

            for member in root.findall('object'):
                if member.find('name').text == 'dolphin':
                    root.remove(member)

            tree.write(f'{WRITE_TO}/{label_to_keep}/{image_name}')

remove_labels('markings')