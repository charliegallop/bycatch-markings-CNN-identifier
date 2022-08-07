import torch
import cv2
import numpy as np
import os
import glob as glob

from xml.etree import ElementTree as et
from config import CLASSES, RESIZE_TO, TRAIN_DIR, TRAIN_ANNOT_DIR, VALID_DIR, VALID_ANNOT_DIR, BATCH_SIZE
from torch.utils.data import Dataset, DataLoader
from utils import collate_fn, get_train_transform, get_valid_transform

# the dataset class
class BycatchDataset(Dataset):
    def __init__(self, dir_path, annot_path, width, height, classes, transforms=None):
        self.transforms = transforms
        self.dir_path = dir_path
        self.annot_path = annot_path
        self.height = height
        self.width = width
        self.classes = classes

        # get all the image paths in sorted order
        self.image_paths = glob.glob(f"{self.dir_path}/*")
        self.all_images = [image_path.split('/')[-1] for image_path in self.image_paths]
        self.all_images = sorted(self.all_images)

    def __getitem__(self, idx):
        # capture the image name and the full image path
        image_name = self.all_images[idx]
        image_path = os.path.join(self.dir_path, image_name)

        # read the image
        image = cv2.imread(image_path).astype(np.float32)
        image_resized = cv2.resize(image, (self.width, self.height))
        image_resized /= 255.0

        # capture the corresponding XML file for getting the annotations
        annot_filename = image_name[:-4] + '.xml'
        annot_file_path = os.path.join(self.annot_path, annot_filename)

        boxes = []
        labels = []
        tree = et.parse(annot_file_path)
        root = tree.getroot()

        # get the height and width of the image
        image_width = image.shape[1]
        image_height = image.shape[0]

        # get list of all objects labelled in image
        objects_list = [name.find('name').text for name in root.findall('object')]
        #print(image_name)
        if 'dolphin' in objects_list:
        # box coordinates for xml files are extracted and correct for image size given
            for member in root.findall('object'):
                if member.find('name').text == 'dolphin':
                    # map the current object name to 'classes' list to get...
                    #... the label index and append to 'labels' list
                    labels.append(self.classes.index(member.find('name').text))

                    crop_coords = {
                        # xmin = left corner x-coordinates
                        'xmin': int(member.find('bndbox').find('xmin').text),
                        # xmax = right corner x-coordinates
                        'xmax': int(member.find('bndbox').find('xmax').text),
                        # ymin = left corner y-coordinates
                        'ymin': int(member.find('bndbox').find('ymin').text),
                        # ymax = left corner y-coordinates
                        'ymax': int(member.find('bndbox').find('ymax').text)
                    }
                    # resize the bounding boxes acording to the..
                    # ... desired 'width', 'height'
                    final_crop = {
                        'xmin_final':(crop_coords['xmin']/image_width)*self.width,
                        'xmax_final':(crop_coords['xmax']/image_width)*self.width,
                        'ymin_final':(crop_coords['ymin']/image_height)*self.height,
                        'ymax_final':(crop_coords['ymax']/image_height)*self.height
                    }
                    boxes.append([final_crop['xmin_final'], final_crop['ymin_final'], final_crop['xmax_final'], final_crop['ymax_final']])
                else:
                    labels.append(self.classes.index(member.find('name').text))

                    box_coords = {
                        # xmin = left corner x-coordinates
                        'xmin': int(member.find('bndbox').find('xmin').text),
                        # xmax = right corner x-coordinates
                        'xmax': int(member.find('bndbox').find('xmax').text),
                        # ymin = left corner y-coordinates
                        'ymin': int(member.find('bndbox').find('ymin').text),
                        # ymax = left corner y-coordinates
                        'ymax': int(member.find('bndbox').find('ymax').text)
                    }
                    # resize the bounding boxes acording to the..
                    # ... desired 'width', 'height'
                    final_box = {
                        'xmin_final':(box_coords['xmin']/image_width)*self.width,
                        'xmax_final':(box_coords['xmax']/image_width)*self.width,
                        'ymin_final':(box_coords['ymin']/image_height)*self.height,
                        'ymax_final':(box_coords['ymax']/image_height)*self.height
                    }
                    #print(final_box)
                    boxes.append([final_box['xmin_final'], final_box['ymin_final'], final_box['xmax_final'], final_box['ymax_final']])
        else: 
            final_crop = {
                'xmin_final': 0.0,
                'xmax_final': self.width,
                'ymin_final': 0.0,
                'ymax_final': self.height
            }
            # create 'dolphin' bounding box assuming it covers the whole image
            boxes.append([final_crop['xmin_final'], final_crop['ymin_final'], final_crop['xmax_final'], final_crop['ymax_final']])
            labels.append(2)

            for member in root.findall('object'):
                if member.find('name').text != 'dolphin':
                    # map the current object name to 'classes' list to get...
                    #... the label index and append to 'labels' list
                    labels.append(self.classes.index(member.find('name').text))

                    box_coords = {
                        # xmin = left corner x-coordinates
                        'xmin': int(member.find('bndbox').find('xmin').text),
                        # xmax = right corner x-coordinates
                        'xmax': int(member.find('bndbox').find('xmax').text),
                        # ymin = left corner y-coordinates
                        'ymin': int(member.find('bndbox').find('ymin').text),
                        # ymax = left corner y-coordinates
                        'ymax': int(member.find('bndbox').find('ymax').text)
                    }
                    # resize the bounding boxes acording to the..
                    # ... desired 'width', 'height'
                    final_box = {
                        'xmin_final':(box_coords['xmin']/image_width)*self.width,
                        'xmax_final':(box_coords['xmax']/image_width)*self.width,
                        'ymin_final':(box_coords['ymin']/image_height)*self.height,
                        'ymax_final':(box_coords['ymax']/image_height)*self.height
                    }
                    #print(final_box)
                    boxes.append([final_box['xmin_final'], final_box['ymin_final'], final_box['xmax_final'], final_box['ymax_final']])

        # bounding box to tensor
        boxes = torch.as_tensor(boxes, dtype = torch.float32)
        # area of the bounding boxes
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # no crowd instances
        iscrowd = torch.zeros((boxes.shape[0],), dtype = torch.int64)
        # labels to tensor
        labels = torch.as_tensor(labels, dtype = torch.int64)

        # prepare the final 'target' dictionary
        target = {}
        #target["image_name"] = image_name # for debugging
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = iscrowd
        image_id = torch.tensor([idx])
        target["image_id"] = image_id
        # apply the image transforms
        if self.transforms:
            sample = self.transforms(image = image_resized,
            bboxes = target['boxes'],
            labels = labels)
            image_resized = sample['image']
            target['boxes'] = torch.Tensor(sample['bboxes'])

        crop_to = {k: int(v) for k, v in final_crop.items()}
        # xstart, xstop, ystart, ystop = bbox_coordinates[0], bbox_coordinates[2], bbox_coordinates[1], bbox_coordinates[3]
        cropped_image = image_resized[:, crop_to['ymin_final']:crop_to['ymax_final'], crop_to['xmin_final']:crop_to['xmax_final']]
        #print(cropped_image.shape, image_name, objects_list)
        # print('IMAGE RESIZED: ', image_resized.shape)
        # print('CROPPED: ', cropped_image.shape)
        return cropped_image, target

    def __len__(self):
        return len(self.all_images)

# move the images from 'forAnnotations' folder to either test or train folder...
#... based on the annotation files

def move_images_to_project_folder(annot_dir, images_dir, original_path):
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

    

move_images_to_project_folder(TRAIN_ANNOT_DIR, TRAIN_DIR, '/home/charlie/forAnnotation')
move_images_to_project_folder(VALID_ANNOT_DIR, VALID_DIR, '/home/charlie/forAnnotation')


# prepare the final datasets and data loaders
train_dataset = BycatchDataset(TRAIN_DIR, TRAIN_ANNOT_DIR, RESIZE_TO, RESIZE_TO, CLASSES, get_train_transform())
valid_dataset = BycatchDataset(VALID_DIR, VALID_ANNOT_DIR, RESIZE_TO, RESIZE_TO, CLASSES, get_valid_transform())
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    collate_fn=collate_fn
)
valid_loader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    collate_fn=collate_fn
)
print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of validation samples: {len(valid_dataset)}\n")


# execute datasets.py using Python command from Terminal...
# ... to visualize sample images
# USAGE: python datasets.py
if __name__ == '__main__':
    # sanity check of the Dataset pipeline with sample visualization
    dataset = BycatchDataset(
        TRAIN_DIR, TRAIN_ANNOT_DIR, RESIZE_TO, RESIZE_TO, CLASSES
    )
    dataset_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn
    )
    print(f"Number of training images: {len(dataset)}")
    
    # function to visualize a single sample
    def visualize_sample(image, target):
        box = target['boxes'][0]
        label = CLASSES[target['labels'][0]]
        cv2.rectangle(
            image, 
            (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
            (0, 255, 0), 1
        )
        cv2.putText(
            image, label, (int(box[0]), int(box[1]-5)), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
        )
        cv2.imshow('Image', image)
        cv2.waitKey(0)
        
    # NUM_SAMPLES_TO_VISUALIZE = 18
    # for i in range(NUM_SAMPLES_TO_VISUALIZE):
    #     image, target = dataset[i]
    #     visualize_sample(image, target)

    for data in dataset_loader: 
        images, targets = data
        img_num = 1

        img_sizes = [image.size() for image in images]

        target_names = [target['image_name'] for target in targets]

        for i, size in enumerate(img_sizes):
            if size[0] < 3:
                print(target_names[i],': ', size)
        # for image in images:
            
        #     print('IMAGE SIZE: ', image.size())
        #     img_num += 1
        
        # target_num = 1
        # for target in targets:
        #     print('IMAGE NAME: ', target['image_name'] )
        #     target_num += 1
        #     visualize_sample(image, target)
