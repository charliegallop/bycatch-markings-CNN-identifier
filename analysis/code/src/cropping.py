import numpy as np
import pandas as pd
import cv2
import torch
import glob as glob
import os
import shutil
import torch

from model import create_model
from config import ROOT, NUM_EPOCHS, COLOURS, BACKBONE, THRESHOLD, TRAIN_FOR, RESIZE_TO
from config import TEST_DIR, EVAL_DIR, VAL_DIR, MASTER_MARKINGS_DIR,  MARKINGS_DIR, TRAIN_DIR
from utils import save_predictions_as_txt, save_metrics
from edit_xml import keep_labels

class Cropping_engine():

    def __init__(self, BACKBONE, TRAIN_FOR, IMAGES_DIR, LABELS_DIR, MODEL_PATH=None,  MODEL=None,):

        self.train_for = TRAIN_FOR
        self.saved_images = 0
        self.crop_model = None
        self.images_dir = IMAGES_DIR
        self.labels_dir = LABELS_DIR
        self.crop = False
        self.backbone = BACKBONE
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # width and height to resize images to
        self.width = RESIZE_TO
        self.height = RESIZE_TO


        if self.train_for == 'dolphin':
            self.output_dir = os.path.join(MARKINGS_DIR)
            self.crop = True
            from config import CLASSES_DOLPHIN, NUM_CLASSES_DOLPHIN
            self.num_classes = NUM_CLASSES_DOLPHIN
            self.classes = CLASSES_DOLPHIN
        else:
            self.output_dir = os.path.join(EVAL_DIR, self.train_for, self.backbone)
            self.crop = False
            from config import CLASSES_MARKINGS, NUM_CLASSES_MARKINGS
            self.num_classes = NUM_CLASSES_MARKINGS
            self.classes = CLASSES_MARKINGS

        if MODEL_PATH:
            self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            
            # load model and trained weights
            self.model = create_model(
                num_classes=self.num_classes, 
                backbone = BACKBONE
                ).to(self.device)

            self.model.load_state_dict(torch.load(
                MODEL_PATH, 
                map_location = self.device
            ))
        else:
            self.model = MODEL 


        # define the detection threshold...
        #... any detection having score below this will be discarded
        self.detection_threshold = THRESHOLD
        print('-'*50)
        print(f"Cropping image for '{self.train_for}' with model backbone '{self.backbone}' with detection_threshold = {self.detection_threshold}")

        self.all_stats = []

        # # set the computation device
       

    def _getArea(self, box):
        return (box[2] - box[0]) * (box[3] - box[1])
    
    def crop_image(self, img, box):
        # buffer to add to the crop to account for any error of the box cropping too small
        buffer =0.05

        coords = {
            "ymin": box[1], 
            "ymax": box[3], 
            "xmin": box[0], 
            "xmax": box[2]
            }
        
        img_width = img.shape[1]
        img_height = img.shape[0]

        # checks if adding the buffer would push the box over the edges of the image.
        # If it doesn't it sets the new coordinate + buffer
        if ((coords['ymin']  - img_height*buffer) >= 0):
            coords['ymin'] = int(coords['ymin']  - img_height*buffer)
        if ((coords['xmin']  - img_width*buffer) >= 0):
            coords['xmin'] = int(coords['xmin']  - img_width*buffer)
        if ((coords['ymax']  + img_height*buffer) >= 0):
            coords['ymax'] = int(coords['ymax']  + img_height*buffer)
        if ((coords['xmax']  + img_width*buffer) >= 0):
            coords['xmax'] = int(coords['xmax']  + img_width*buffer)
        
        # Crops image
        crop = img[coords['ymin']:coords['ymax'], coords['xmin']:coords['xmax']]
        return crop

    def get_cropped_image(self, image_path):
            self.model.eval()

            # get the image file name for saving output later on
            image_name = os.path.basename(image_path)
            image = cv2.imread(image_path).astype(np.float32)
            orig_image = image.copy()


            # resize the image
            image = cv2.resize(image, (self.width, self.height))

            # make the pixel range between 0 and 1
            image /= 255.0

            # bring color channels to front
            image = np.transpose(image, (2, 0, 1)).astype(np.float64)

            # convert to tensor
            image = torch.tensor(image, dtype = torch.float).cuda()

            # add batch dimension
            image = torch.unsqueeze(image, 0)

            with torch.no_grad():
                outputs = self.model(image) # outputs will consit of two tensors [targets, images]

            # load all detection to CPU for further operations
            outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]

            # saves the box with the largets area. Will use to crop the image
            max_area = 0
            # saves the box coordinates to crop the image to
            box_to_crop = []

            # carry further only if there are detected boxes
            if len(outputs[0]['boxes']) != 0:
                boxes = outputs[0]['boxes'].data.numpy()
                scores = outputs[0]['scores'].data.numpy()
                save_preds_to = os.path.join(EVAL_DIR, self.train_for, self.train_for())
                pred_classes = [self.classes[i] for i in outputs[0]['labels'].cpu().numpy()]

                self.all_stats = save_metrics(boxes, pred_classes, scores, image_name, self.all_stats)
                # save_predictions_as_txt(outputs, image_name, save_preds_to)

                # filter out boxes according to the detection threshold
                boxes = boxes[scores >= self.detection_threshold].astype(np.int32)
                draw_boxes = boxes.copy()
                # get all the predicted class names
                pred_classes = [self.classes[i] for i in outputs[0]['labels'].cpu().numpy()]

                # draw the bounding boxes and write class name on top of it
                for j, box in enumerate(draw_boxes):                   
                    # only do this is self.crop is true which should only be for dophins
                    if self.crop:
                        area = self._getArea(box)
                        if area > max_area:
                            max_area = area
                            box_to_crop = box
                
            # check meant to be cropping and there are bounding box predictions 
            # to crop to.    
            if (self.crop) & (len(box_to_crop) != 0):
                print("Cropping image...and sending to markings predictions")
                cropped_img = self.crop_image(orig_image, box_to_crop)
                return cropped_img
                # write_to_dir = os.path.join(MASTER_MARKINGS_DIR, 'images', f'{image_name}.jpg')
                # cv2.imwrite(write_to_dir, cropped_img)
            
            else:
                print("No bounding boxes predicted. Sending original image to markings predicitons...")
                return orig_image
                # # if not bounding boxes exist just save original image
                # write_to_dir = os.path.join(MASTER_MARKINGS_DIR, 'images', f'{image_name}.jpg')
                # cv2.imwrite(write_to_dir, orig_image)    

    def crop_and_save(self):
        count = 0 
        # set the computation device
        # print("NUM_CLASSES: ", self.num_classes, 
        # "\nBACKBONE: ", self.backbone, 
        # "\nCROP_MODEL: ", self.crop_model,
        # "\nMODEL: ", self.model,
        # "\nDEVICE: ", self.device
        # )
        # print("MODEL: ", self.model)

        # load model and trained weights
        self.crop_model = self.model
        self.crop_model.eval()

        image_paths = glob.glob(f"{self.images_dir}/*")
        label_paths = glob.glob(f"{self.labels_dir}/*")

        print(f"Images to crop: {len(image_paths)}")

        self.all_stats = []

        # Looping over the image paths and carrying out inference
        for i in image_paths:
            # get the image file name for saving output later on
            image_name = os.path.basename(i).split('.')[0]
            image = cv2.imread(i).astype(np.float32)
            orig_image = image.copy()
            orig_image_bb = image.copy()
            
            # make the pixel range between 0 and 1
            image /= 255.0
            # bring color channels to front
            image = np.transpose(image, (2, 0, 1)).astype(np.float64)
            # convert to tensor
            image = torch.tensor(image, dtype = torch.float).cuda()
            # add batch dimension
            image = torch.unsqueeze(image, 0)
            with torch.no_grad():
                outputs = self.crop_model(image) # outputs will consit of two tensors [targets, images]
            
            # load all detection to CPU for further operations
            outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]

            # saves the box with the largets area. Will use to crop the image
            max_area = 0
            # saves the box coordinates to crop the image to
            box_to_crop = []

            # carry further only if there are detected boxes
            if len(outputs[0]['boxes']) != 0:
                boxes = outputs[0]['boxes'].data.numpy()
                scores = outputs[0]['scores'].data.numpy()
                save_preds_to = os.path.join(EVAL_DIR, self.train_for, self.backbone)
                pred_classes = [self.classes[i] for i in outputs[0]['labels'].cpu().numpy()]

                self.all_stats = save_metrics(boxes, pred_classes, scores, image_name, self.all_stats)
                save_predictions_as_txt(outputs, image_name, save_preds_to)

                # filter out boxes according to the detection threshold
                boxes = boxes[scores >= self.detection_threshold].astype(np.int32)
                draw_boxes = boxes.copy()
                # get all the predicted class names
                pred_classes = [self.classes[i] for i in outputs[0]['labels'].cpu().numpy()]
                # draw the bounding boxes and write class name on top of it
                print("DRAW_BOXES BEFORE IF: ", draw_boxes)
                print("LEN DRAW_BOXES BEFORE IF: ", len(draw_boxes))
                saved = False
                if len(draw_boxes) != 0:
                    print("DRAW_BOXES AFTER IF: ", draw_boxes)
                    print("LEN DRAW_BOXES AFTER IF: ", len(draw_boxes))

                    for j, box in enumerate(draw_boxes):
                        print("J: ", j)
                        print("DRAW_BOXES FOR: ", draw_boxes[j], draw_boxes)     
                        print("BOX FOR: ", box, box)         


                        # only do this is self.crop is true which should only be for dophins
                        if self.crop:
                            area = self._getArea(box)
                            if area > max_area:
                                max_area = area
                                box_to_crop = box

                        
                        # Draw predictions to image and save to eval directory
                        cv2.rectangle(orig_image_bb,
                                    (int(box[0]), int(box[1])),
                                    (int(box[2]), int(box[3])),
                                    (COLOURS[pred_classes[j]]), 3)
                        cv2.putText(orig_image_bb, pred_classes[j].upper() + " CONF: " + str(round(scores[j], 2)), 
                                    (int(box[0]), int(box[1]-5)),
                                    cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 255, 255), 
                                    2, lineType=cv2.LINE_AA)
                        write_to_dir = os.path.join(EVAL_DIR, self.train_for, self.backbone, 'images', f'{image_name}.jpg')
                        print("SAVING TO: ", write_to_dir)
                        cv2.imwrite(write_to_dir, orig_image_bb)
                else:
                    print("No predictions over threshold...saving original image...")
                    write_to_dir = os.path.join(EVAL_DIR, self.train_for, self.backbone, 'images', f'{image_name}.jpg')
                    cv2.imwrite(write_to_dir, orig_image_bb)
            else:
                print(f"Image {image_name} saved")
                write_to_dir = os.path.join(EVAL_DIR, self.train_for, self.backbone, "images", f'{image_name}.jpg')
                cv2.imwrite(write_to_dir, orig_image)


            if (self.crop):
                if (len(box_to_crop) != 0):
                    print("Cropping image....")
                    cropped_img = self.crop_image(orig_image, box_to_crop)

                    write_to_dir = os.path.join(MASTER_MARKINGS_DIR, 'images', f'{image_name}.jpg')
                    cv2.imwrite(write_to_dir, cropped_img)
                    print(f"Image {image_name} done... cropped and saved")

                else:
                    print(f"Image {image_name} not cropped but saved")
                    write_to_dir = os.path.join(MASTER_MARKINGS_DIR, 'images', f'{image_name}.jpg')
                    cv2.imwrite(write_to_dir, orig_image)
            

            print("Saved to: ", write_to_dir)
            count += 1
            print(f"Processed {count}/{len(image_paths)}")
            print('-'*50)
        
        if (self.crop):
            # move edited xml files with only dolphin label to directory
            copy_to = os.path.join(MASTER_MARKINGS_DIR, 'labels')
            keep_labels(
                label_dir=self.labels_dir, 
                WRITE_TO=copy_to, 
                label_to_keep="dolphin"
                )
                
            copy_to = os.path.join(EVAL_DIR, self.train_for, self.backbone, "gt")
            keep_labels(
                label_dir=self.labels_dir, 
                WRITE_TO=copy_to, 
                label_to_keep="dolphin"
                )
        else:
            # move edited xml files with only markings label to directory
            copy_to = os.path.join(EVAL_DIR, self.train_for, self.backbone, "gt")
            keep_labels(
                label_dir=self.labels_dir, 
                WRITE_TO=copy_to, 
                label_to_keep="markings"
                )

        
        # TODO: add ground truth bouding boxes to the df
        df = pd.DataFrame(columns = ["image_name", "class", "xmin_pred", "ymin_pred", "xmax_pred", "ymax_pred", "score"])
        for i in self.all_stats:
            df.loc[len(df)] = i
        
        save_df = os.path.join(self.output_dir, "all_predictions.csv")
        df.to_csv(save_df)
        if self.crop:
            print('CROPPING OF TRAINING IMAGES COMPLETE')
            print(f'CROPPED {count} images')
        else:
            print('INFERENCE OF VAL IMAGES COMPLETE')
            print(f'EVALUATED {count} images')

        cv2.destroyAllWindows()
        