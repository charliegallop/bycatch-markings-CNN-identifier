from xml.etree import ElementTree as et
import glob

ANNOT_DIR = "/home/charlie/Documents/Uni/Exeter - Data Science/Dissertation/analysis/data/train_set/valid_annotations"
WRITE_TO = "/home/charlie/Documents/Uni/Exeter - Data Science/Dissertation/analysis/data/train_set/valid_predictions/ground_truth" 

annotation_dir = glob.glob(f"{ANNOT_DIR}/*")

for i in annotation_dir:
    image_name = i.split('/')[-1]
    tree = et.parse(i)
    root = tree.getroot()

    for member in root.findall('object'):
        if member.find('name').text != 'dolphin':
            root.remove(member)

    tree.write(f'{WRITE_TO}/dolphin/{image_name}')

