import albumentations as A
import cv2
import numpy as np
import torch
import os

from albumentations.pytorch import ToTensorV2
from config import DEVICE, CLASSES_MARKINGS as classes_markings, CLASSES_DOLPHIN as classes_dolphins
from config import ROOT, BACKBONE, THRESHOLD, TEST_PREDS_DIR

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
        ToTensorV2(p=1.0)
    ], bbox_params = {
        'format': 'pascal_voc',
        'label_fields': ['labels']
    })

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
    save_dir = save_dir
    with open(f"{save_dir}/{image_name}.txt", 'w') as f:
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

        if string >= THRESHOLD:
            text = []
            text.append(str(label))
            text.append(str(string))
            for point in box:
                text.append(str(int(point)))
            text = " ".join(text)
            conv_predictions.append(text)
        else:
            pass

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