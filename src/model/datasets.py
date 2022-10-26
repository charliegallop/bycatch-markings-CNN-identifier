import torch
import cv2
import numpy as np
import os
import glob as glob
from PIL import Image
import shutil

from xml.etree import ElementTree as et
from torch.utils.data import Dataset, DataLoader
from utils import collate_fn, get_train_transform, get_valid_transform, get_labels
from split_datasets import SplitDatasets

class ModelConfig():

    def __init__(self,train_for = None, images_dir = None, labels_dir = None, batch_size = None, resize = None, classes = None, transforms=None):
        self.train_for = train_for
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.classes = classes
        self.batch_size = batch_size
        self.transforms = transforms
        self.height = resize
        self.width = resize

        self.image_paths = glob.glob(f"{self.images_dir}/*")

# the dataset class
class BycatchDataset(Dataset):
    def __init__(self, image_dir = None, label_dir = None, resize_to = None, classes = None, train_for = None, transforms = None):
        self.train_for = train_for
        self.images_dir = image_dir
        self.labels_dir = label_dir
        self.classes = classes
        self.transforms = transforms
        self.all_image_paths = glob.glob(os.path.join(self.images_dir, "*.jpg"))
        self.width = resize_to
        self.height = resize_to

    def __getitem__(self, idx):
        # capture the image name and the full image path
        image_path = self.all_image_paths[idx]
        image_name = os.path.basename(image_path).split(".")[0]

        # read the image
        image = cv2.imread(image_path).astype(np.float32)
        image_resized = cv2.resize(image, (self.width, self.height))
        image_resized /= 255.0

        # capture the corresponding XML file for getting the annotations
        # filename = os.path.basename(image_name)
        annot_filename = image_name + '.xml'
        annot_file_path = os.path.join(self.labels_dir, annot_filename)

        boxes = []
        labels = []
        tree = et.parse(annot_file_path)
        root = tree.getroot()

        # get the height and width of the image
        image_width = image.shape[1]
        image_height = image.shape[0]


    
        # box coordinates for xml files are extracted and 
        # corrected for image size depending on what classes are being trained for
        # get list of all objects labelled in image
        objects_list = [name.find('name').text for name in root.findall('object')]
        if self.train_for == 'dolphin':
            if len(root.findall('object')) != 0:
                for member in root.findall('object'):
                    if 'dolphin' in objects_list:
                        #TODO: Might need to add 'background here as well
                        if member.find('name').text == 'dolphin':
                            boxes, labels = get_labels(self.classes, member, self.width, self.height, boxes, labels, image_width=image_width, image_height = image_height)
                    else:

                        background_bb = {
                            'xmin_final': 0.0,
                            'xmax_final': self.width,
                            'ymin_final': 0.0,
                            'ymax_final': self.height
                        }
                        # create 'dolphin' bounding box assuming it covers the whole image
                        boxes.append([background_bb['xmin_final'], background_bb['ymin_final'], background_bb['xmax_final'], background_bb['ymax_final']])
                        labels.append(0)
            else:

                background_bb = {
                    'xmin_final': 0.0,
                    'xmax_final': self.width,
                    'ymin_final': 0.0,
                    'ymax_final': self.height
                }
                # create 'background' bounding box assuming no dolphin in image
                boxes.append([background_bb['xmin_final'], background_bb['ymin_final'], background_bb['xmax_final'], background_bb['ymax_final']])
                labels.append(0)

        elif self.train_for == 'markings':
            for member in root.findall('object'):
                if member.find('name').text == 'dolphin':
                    root.remove(member)
                else:
                    boxes, labels = get_labels(self.classes, member, self.width, self.height, boxes, labels, image_width=image_width, image_height = image_height)
            if not boxes:
                background_bb = {
                                'xmin_final': 0.0,
                                'xmax_final': self.width,
                                'ymin_final': 0.0,
                                'ymax_final': self.height
                            }
                # create 'background' bounding box assuming no dolphin
                boxes.append([background_bb['xmin_final'], background_bb['ymin_final'], background_bb['xmax_final'], background_bb['ymax_final']])
                labels.append(0)

        elif self.train_for == 'all':
            for member in root.findall('object'):
                xmin = int(member.find('bndbox').find('xmin').text)
                ymin = int(member.find('bndbox').find('ymin').text)
                xmax = int(member.find('bndbox').find('xmax').text)
                ymax = int(member.find('bndbox').find('ymax').text)

                xmin_final = (xmin/image_width)*self.width
                xmax_final = (xmax/image_width)*self.width
                ymin_final = (ymin/image_height)*self.height
                ymax_final = (ymax/image_height)*self.height

                boxes.append([xmin_final, ymin_final, xmax_final, ymax_final])
                labels.append(classes.index(member.find('name').text))


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
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = iscrowd
        image_id = torch.tensor([idx])
        target["image_id"] = image_id
        

        # apply the image transforms
        if self.transforms:
            sample = self.transforms(
                image = image_resized,
                bboxes = target['boxes'],
                labels = labels
                )
            # resized image
            image_resized = sample['image']
            # resized boxes
            target['boxes'] = torch.Tensor(sample['bboxes'])
        return image_resized, target

    def __len__(self):
        return len(self.all_image_paths)

class CreateDataLoaders():

    def __init__(self, train_for = None, images_dir = None, labels_dir = None, classes = None, batch_size = None, resize = None, backbone = None):
        self.train_for = train_for
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.classes = classes
        self.batch_size = batch_size
        self.resize = resize
        self.set_dirs = []
        self.transforms = {
            'train': get_train_transform(),
            'valid': get_valid_transform()
        }
        self.backbone = backbone

    def make_dataloader(self, directory, transforms, shuffle):
        image_dir = os.path.join(directory, "images")
        label_dir = os.path.join(directory, "labels")

        dataset = BycatchDataset(
            image_dir= image_dir, 
            label_dir= label_dir, 
            resize_to = self.resize,
            classes= self.classes, 
            train_for= self.train_for, 
            transforms= transforms
            )


        # initiate train loader
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=0,
            collate_fn=collate_fn
        )
        
        return loader, dataset

    def get_set_dirs(self, datasets_dir = None):
        return {
            "train":os.path.join(datasets_dir, "train"),
            "test":os.path.join(datasets_dir, "test"),
            "valid":os.path.join(datasets_dir, "valid")
        }

    def get_data_loaders(self):
        
        model_specs = ModelConfig(
            train_for=self.train_for,
            images_dir=self.images_dir,
            labels_dir = self.labels_dir,
            batch_size = self.batch_size,
            resize = self.resize, 
            classes = self.classes,
            transforms= [get_train_transform(), get_valid_transform()]
        )

        # Where to copy the train test split sets to
        datasets_dir = "/home/charlie/Documents/Uni/Exeter - Data Science/Dissertation/data/processed"
        # set_dirs = check_datasets(datasets_dir=datasets_dir)
        self.set_dirs = self.get_set_dirs(datasets_dir)

        split_datasets = SplitDatasets(
            images_dir = self.images_dir, 
            labels_dir = self.labels_dir,
            resize_to = self.resize,
            set_dirs = self.set_dirs,
            backbone = self.backbone,
            train_for = self.train_for
            )
        
        split_datasets.main()

        print('_'*50)
        print(f"Creating 'train' dataloaders for {self.train_for} dataset")
        print('_'*50)

        train_loader, train_dataset = self.make_dataloader(
            directory = self.set_dirs['train'],
            transforms= self.transforms['train'],
            shuffle = True
            )

        print('_'*50)
        print(f"Creating 'valid' dataloaders for {self.train_for} dataset")
        print('_'*50)

        valid_loader, valid_dataset = self.make_dataloader(
            directory = self.set_dirs['valid'],
            transforms= self.transforms['valid'],
            shuffle = False
            )
        return train_loader, valid_loader
