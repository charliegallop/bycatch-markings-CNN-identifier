import torch
import cv2
import numpy as np
import os
import glob as glob
from PIL import Image

from xml.etree import ElementTree as et
from config import CLASSES_DOLPHIN, CLASSES_MARKINGS, RESIZE_TO, TRAIN_DIR, VAL_DIR, BATCH_SIZE, TRAIN_FOR, NUM_WORKERS
from torch.utils.data import Dataset, DataLoader
from utils import collate_fn, get_train_transform, get_valid_transform, get_labels

# the dataset class
class BycatchDataset(Dataset):
    def __init__(self, dir_path, annot_path, width, height, classes, selection, transforms=None):
        self.selection = TRAIN_FOR.value()
        self.transforms = transforms
        self.dir_path = dir_path
        self.height = height
        self.width = width
        self.classes = classes
        self.annot_dir = os.path.join(self.dir_path, "labels")
        # get all the image paths in sorted order
        self.image_dir = os.path.join(self.dir_path, "images")
        self.image_paths = glob.glob(f"{self.image_dir}/*")
        self.all_images = [image_path.split('/')[-1] for image_path in self.image_paths]
        self.all_images = sorted(self.all_images)

    
    def __getitem__(self, idx):
        # capture the image name and the full image path
        image_name = self.all_images[idx]
        image_path = os.path.join(self.image_dir, image_name)
        # read the image
        image = cv2.imread(image_path).astype(np.float32)
        image_resized = cv2.resize(image, (self.width, self.height))
        image_resized /= 255.0

        # capture the corresponding XML file for getting the annotations
        filename = os.path.basename(image_name)
        annot_filename = filename[:-4] + '.xml'
        annot_file_path = os.path.join(self.annot_dir, annot_filename)

        boxes = []
        labels = []
        tree = et.parse(annot_file_path)
        root = tree.getroot()

        # get the height and width of the image
        image_width = image.shape[1]
        image_height = image.shape[0]


        # get list of all objects labelled in image
        objects_list = [name.find('name').text for name in root.findall('object')]
       
        # box coordinates for xml files are extracted and correct for image size given
        if self.selection == 'dolphin':
            for member in root.findall('object'):
                if 'dolphin' in objects_list:
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

        elif self.selection == 'markings':
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
                # create 'dolphin' bounding box assuming it covers the whole image
                boxes.append([background_bb['xmin_final'], background_bb['ymin_final'], background_bb['xmax_final'], background_bb['ymax_final']])
                labels.append(0)
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
            sample = self.transforms(image = image_resized,
            bboxes = target['boxes'],
            labels = labels)
            image_resized = sample['image']
            target['boxes'] = torch.Tensor(sample['bboxes'])
        #print(image_resized.size(), image_name)
        return image_resized, target

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

    

#move_images_to_project_folder(TRAIN_ANNOT_DIR, TRAIN_DIR, '/home/charlie/forAnnotation')
#move_images_to_project_folder(VALID_ANNOT_DIR, VALID_DIR, '/home/charlie/forAnnotation')


# TODO: Make the creation of the datasets a function so custom BATCH_SIZE and RESIZE_TO values can be added from the main script

def make_dataloader(train_dirs, valid_dirs, classes, train_for, batch_size = BATCH_SIZE, transforms = [get_train_transform(), get_valid_transform()], resize_to = RESIZE_TO):
    print('_'*50)
    print(f"Creating dataloaders for {TRAIN_FOR.value()} dataset")
    print('_'*50)

    transforms = [get_train_transform(), get_valid_transform()]

    train_dataset = BycatchDataset(
        dir_path= train_dirs[0], 
        annot_path= train_dirs[1], 
        width= resize_to, 
        height= resize_to, 
        classes= classes, 
        selection= train_for, 
        transforms= transforms[0]
        )

    valid_dataset = BycatchDataset(
        dir_path= valid_dirs[0], 
        annot_path= valid_dirs[1], 
        width= resize_to, 
        height= resize_to, 
        classes= classes, 
        selection= train_for, 
        transforms= transforms[1]
        )
    # initiate train loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn
    )

    # initiate validation loader
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn
    )
    
    return train_loader, valid_loader, train_dataset, valid_dataset

training_dirs = [f"{TRAIN_DIR}", f"{TRAIN_DIR}"]
valid_dirs = [f"{VAL_DIR}", f"{VAL_DIR}"]


if TRAIN_FOR.value() == 'dolphin':
    training_dirs = [f"{TRAIN_DIR}", f"{TRAIN_DIR}"]
    valid_dirs = [f"{VAL_DIR}", f"{VAL_DIR}"]
    train_loader, valid_loader, train_dataset, valid_dataset = make_dataloader(training_dirs, valid_dirs, CLASSES_DOLPHIN, "dolphin")
elif TRAIN_FOR.value() == 'markings':
    from config import MARKINGS_DIR
    training_dirs = [os.path.join(MARKINGS_DIR, "train"), os.path.join(MARKINGS_DIR, "train")]
    valid_dirs = [os.path.join(MARKINGS_DIR, "val"), os.path.join(MARKINGS_DIR, "val")]
    train_loader, valid_loader, train_dataset, valid_dataset = make_dataloader(training_dirs, valid_dirs, CLASSES_MARKINGS, "markings")

print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of validation samples: {len(valid_dataset)}\n")

# execute datasets.py using Python command from Terminal...
# ... to visualize sample images
# USAGE: python datasets.py
# if __name__ == '__main__':
#     # sanity check of the Dataset pipeline with sample visualization
#     if TRAIN_FOR == 'dolphin':
#         classes = CLASSES_DOLPHIN
#         # dataset = BycatchDataset(
#         #     training_dirs[0], training_dirs[1], RESIZE_TO, RESIZE_TO, CLASSES_DOLPHIN, "dolphin"
#         # )
#     if TRAIN_FOR == 'markings':
#         classes = CLASSES_MARKINGS
#     #     dataset = BycatchDataset(
#     #         training_dirs[0], training_dirs[1], RESIZE_TO, RESIZE_TO, CLASSES_MARKINGS, "markings"
#     #     )
#     #     print(dataset)
#     # print(f"Number of training images: {len(dataset)}")
    
#     # dataset_loader = DataLoader(
#     #     dataset,
#     #     batch_size=BATCH_SIZE,
#     #     shuffle=False,
#     #     num_workers=NUM_WORKERS,
#     #     collate_fn=collate_fn
#     # )

#     # dataset_loader, _, dataset, _ = make_dataloader(training_dirs, valid_dirs, classes, TRAIN_FOR)
#     # print(dataset)


#     # function to visualize a single sample
#     def visualize_sample(image, target):
#         box = target['boxes'][0]
#         label = classes[target['labels'][0]]
#         cv2.rectangle(
#             image, 
#             (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
#             (0, 255, 0), 1
#         )
#         cv2.putText(
#             image, label, (int(box[0]), int(box[1]-5)), 
#             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
#         )
#         cv2.imshow('Image', image)
#         cv2.waitKey(0)
        
#     NUM_SAMPLES_TO_VISUALIZE = 3
#     for i in range(NUM_SAMPLES_TO_VISUALIZE):
#         image, target = train_dataset[i]
#         image = train_dataset[0][0].numpy()*255
#         print(image.T.shape)
#         visualize_sample(image.T, target)