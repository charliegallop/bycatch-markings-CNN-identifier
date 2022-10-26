import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
import os
from xml.etree import ElementTree as et

from albumentations.pytorch import ToTensorV2
# from config import DEVICE
# from config import ROOT, BACKBONE, THRESHOLD

# this class keeps track of the training and validation loss values...
#... and helps tp get the average for each epoch

class Averager:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0

    def send(self, value):
        self.current_total += value
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total/self.iterations
    
    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0
    
def collate_fn(batch):
    """
    To handle the data loading as different images may have different number
    of objects and to handle varying size tensors as well.]
    """
    return tuple(zip(*batch))

#define the training transforms
def get_train_transform():
    return A.Compose([
        A.Flip(0.5),
        A.RandomRotate90(0.5),
        A.MotionBlur(p=0.2),
        A.MedianBlur(blur_limit = 3, p = 0.1),
        A.Blur(blur_limit = 3, p=0.1),
        A.RandomBrightnessContrast(p=0.3),
        A.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=0.3),
        ToTensorV2(p=1.0)
    ], bbox_params = {
        'format': 'pascal_voc',
        'label_fields': ['labels']
    })

def make_save_xml(boxes, labels, WRITE_TO, label_name, img_size, remove = None):
    label_name = label_name.split('.')[0]
    img_width = img_size[1]
    img_height = img_size[0]
    img_depth = img_size[2]

    if remove == 'dolphin':
        new_boxes = []
        new_labels = []
        for i,box in enumerate(boxes):
            if labels[i] != 2:
                new_boxes.append(box)
                new_labels.append(labels[i])
        
        boxes = new_boxes
        labels = new_labels

    if remove == 'markings':
        new_boxes = []
        new_labels = []
        for i,box in enumerate(boxes):
            if labels[i] == 2:
                new_boxes.append(box)
                new_labels.append(labels[i])
        
        boxes = new_boxes
        labels = new_labels


    label_list = [
    'background',
    'impression',
    'dolphin',
    'fin_slice',
    'amputation',
    'notch']

    root = et.Element("annotation")

    def create_subEl(parent, name, text=None):
        subEl = et.SubElement(parent, name)
        if text:
            subEl.text = text
        return subEl
    
    create_subEl(root, "folder", "images")
    create_subEl(root, "filename", label_name)

    source = create_subEl(root, "source")
    create_subEl(source, "database", "MyDatabase")
    create_subEl(source, "annotation", "COCO2017")
    create_subEl(source, "image", "flickr")
    create_subEl(source, "flickrid", "NULL")
    create_subEl(source, "annotator", "1")

    owner = create_subEl(root, "owner")
    create_subEl(owner, "flickrid", "NULL")
    create_subEl(owner, "name", "Label Studio")

    size = create_subEl(root, "size")
    create_subEl(size, "width", f"{img_width}")
    create_subEl(size, "height", f"{img_height}")
    create_subEl(size, "depth", f"{img_depth}")

    size = create_subEl(root, "segmented", "0")

    for i, box in enumerate(boxes):
        object = create_subEl(root, "object")
        create_subEl(object, "name", f"{label_list[labels[i]]}")
        create_subEl(object, "pose", "Unspecified")
        create_subEl(object, "truncated", "0")
        create_subEl(object, "difficult", "0")
        bndbox = create_subEl(object, "bndbox")
        create_subEl(bndbox, "xmin", f"{int(box[0])}")
        create_subEl(bndbox, "ymin", f"{int(box[1])}")
        create_subEl(bndbox, "xmax", f"{int(box[2])}")
        create_subEl(bndbox, "ymax", f"{int(box[3])}")

    tree = et.ElementTree(root)
    
    filename = f"{label_name}.xml"
    save_to = os.path.join(WRITE_TO, filename)
    with open (save_to, "wb") as files :
        tree.write(files)


def get_orig_labels(image_path):
    from xml.etree import ElementTree as et
    label_dict = {
        'background': 0,
        'impression': 1,
        'dolphin': 2,
        'fin_slice': 3,
        'amputation': 4,
        'notch': 5,
        }
    label_name = os.path.basename(image_path).split('.')[0]
    label_dir = os.path.dirname(os.path.dirname(image_path))
    label_path = os.path.join(label_dir, 'labels', f"{label_name}.xml")
    tree = et.parse(label_path)
    root = tree.getroot()
    labels = []
    boxes = []
    for member in root.findall('object'):
        name = member.find('name').text
        label = label_dict[name]
        labels.append(label)

        for box in member.findall('bndbox'):
            xmin = int(box.find('xmin').text)
            ymin = int(box.find('ymin').text)
            xmax = int(box.find('xmax').text)
            ymax = int(box.find('ymax').text)
            box = [xmin, ymin, xmax, ymax]
            boxes.append(box)
    return boxes, labels

def crop_transform(xmin, ymin, xmax, ymax, image, image_path):
    buffer =0.05

    coords = {
        "ymin": ymin, 
        "ymax": ymax, 
        "xmin": xmin, 
        "xmax": xmax
        }
    
    img_width = image.shape[1]
    img_height = image.shape[0]

    # checks if adding the buffer would push the box over the edges of the image.
    # If it doesn't it sets the new coordinate + buffer
    if ((coords['ymin']  - img_height*buffer) > 0):
        coords['ymin'] = int(coords['ymin']  - img_height*buffer)
    if ((coords['xmin']  - img_width*buffer) > 0):
        coords['xmin'] = int(coords['xmin']  - img_width*buffer)
    if ((coords['ymax']  + img_height*buffer) < img_height):
        coords['ymax'] = int(coords['ymax']  + img_height*buffer)
    if ((coords['xmax']  + img_width*buffer) < img_width):
        coords['xmax'] = int(coords['xmax']  + img_width*buffer)

    bboxes, labels = get_orig_labels(image_path)
    transform = A.Compose([
        A.Crop(coords['xmin'],coords['ymin'],coords['xmax'],coords['ymax'], always_apply=True, p=1),
        ToTensorV2(p=1.0)
    ], bbox_params = {
        'format': 'pascal_voc',
        'label_fields': ['labels']
    })

    return transform(image=image, bboxes=bboxes, labels=labels)

# define the validation transforms
def get_valid_transform():
    return A.Compose([
        ToTensorV2(p=1.0),
    ], bbox_params={
        'format': 'pascal_voc',
        'label_fields': ['labels']
    })


# Show transformed images after augmentation. VISUALIZE_TRANSFORMED_IMAGES
# ... in config.py file controls this

def show_transformed_image(train_loader):
    """
    This function shows the transformed images from the 'train_loader'.
    Helps to check whether the transformed images along with the corresponding
    labels are correct or not.
    Only runs if 'VISUALIZE_TRANSFORMED_IMAGES = True' in config.py
    """

    if len(train_loader) > 0:
        for i in range(1):
            images, targets = next(iter(train_loader))
            images = list(image.to(DEVICE) for image in images)
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
            boxes = targets[i]['boxes'].cpu().numpy().astype(np.int32)
            sample = images[i].permute(1, 2, 0).cpu().numpy()
            for box in boxes:
                cv2.rectangle(sample,(box[0], box[1]),(box[2], box[3]), (0, 0, 255), 2)
                cv2.imshow('Transformed image', sample)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

def write_preds_to_file(predictions, image_name, save_dir):
    save_as = os.path.join(save_dir, 'preds', image_name + ".txt")
    with open(save_as, 'w') as f:
        for pred in predictions:
            f.write(f"{pred}\n")

def save_predictions_as_txt(prediction_tensor, image_name, save_dir):
    predictions = prediction_tensor
    conv_predictions = []

    num_of_pred = predictions[0]['boxes'].size(dim = 0)

    for pred in range(num_of_pred):
        label = predictions[0]['labels'][pred].tolist()
        string = predictions[0]['scores'][pred].tolist()
        box = predictions[0]['boxes'][pred]

        
        text = []
        text.append(str(label))
        text.append(str(string))
        for point in box:
            text.append(str(int(point)))
        text = " ".join(text)
        conv_predictions.append(text)

    write_preds_to_file(conv_predictions, image_name, save_dir)

def get_labels(classes, member, width, height, boxes, labels, image_width, image_height):

    # map the current object name to 'classes' list to get...
    #... the label index and append to 'labels' list
    labels.append(classes.index(member.find('name').text))

    # xmin = left corner x-coordinates
    xmin = int(member.find('bndbox').find('xmin').text)
    # xmax = right corner x-coordinates
    xmax = int(member.find('bndbox').find('xmax').text)
    # ymin = left corner y-coordinates
    ymin = int(member.find('bndbox').find('ymin').text)
    # ymax = left corner y-coordinates
    ymax = int(member.find('bndbox').find('ymax').text)

    # resize the bounding boxes acording to the..
    # ... desired 'width', 'height'
    xmin_final = (xmin/image_width)*width
    xmax_final = (xmax/image_width)*width
    ymin_final = (ymin/image_height)*height
    ymax_final = (ymax/image_height)*height

    boxes.append([xmin_final, ymin_final, xmax_final, ymax_final])
    return boxes, labels

def save_metrics(boxes, pred_classes, scores, image_name, master_list):
    metrics = []
    boxes = boxes.tolist()
    scores = scores.tolist()
    for i in range(len(boxes)):
        metrics = [[image_name], [pred_classes[i]], boxes[i], [scores[i]]]
        metrics = [x for l in metrics for x in l]
        master_list.append(metrics)


    #temp_df = pd.DataFrame(columns = ["image_name", "class", "xmin", "ymin", "xmax", "ymax", "xmin_pred", "ymin_pred", "xmax_pred", "ymax_pred", "score"])

    return master_list

def move_images_to_project_folder(annot_dir, images_dir, original_path):

    """ move the images from 'forAnnotations' folder to either test or train folder
        based on the annotation files """

    import shutil
    annot_paths = glob.glob(f"{annot_dir}/*")
    annot_names = [annot.split('/')[-1].split('.')[0] for annot in annot_paths]

    # get list of images already in folder
    image_paths = glob.glob(f"{images_dir}/*")
    image_names = [image.split('/')[-1].split('.')[0] for image in image_paths]

    for annot in annot_names:
        if annot not in image_names:
            path_of_file_to_move = glob.glob(f"{original_path}/{annot}.*")[0]
            if path_of_file_to_move:
                print(f"MOVING {annot} TO {images_dir}....")
                shutil.move(path_of_file_to_move, images_dir)
