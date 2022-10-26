from config import DEVICE, NUM_CLASSES_DOLPHIN, NUM_CLASSES_MARKINGS, NUM_EPOCHS, OUT_DIR, NUM_WORKERS
from config import THRESHOLD, CLASSES_DOLPHIN,CLASSES_MARKINGS, COLOURS, BATCH_SIZE, RESIZE_TO
from config import VISUALIZE_TRANSFORMED_IMAGES, BACKBONE
from config import SAVE_PLOTS_EPOCH, SAVE_MODEL_EPOCH
from config import FINAL_PREDS_DIR, ROOT
from model import create_model
from utils import save_predictions_as_txt, save_metrics
from cropping import Cropping_engine
from custom_eval import calc_metrics

import torch
import matplotlib.pyplot as plt
import time
import cv2
import os
import glob
import numpy as np
import pandas as pd
import re

plt.style.use('Solarize_Light2')

class predict_engine():

    def __init__(self, dolphin_backbone = None, markings_backbone = None, predict_for = None, images_dir = None, dolphin_pred_model = None, markings_pred_model = None, output_dir = None, detection_threshold = None, crop = None):


        self.count = 0
        self.predict_for = predict_for
        self.dolphin_backbone = dolphin_backbone
        self.markings_backbone = markings_backbone
        self.crop_model_path = dolphin_pred_model
        self.pred_model_path = markings_pred_model
        self.images_dir = images_dir
        self.output_dir = output_dir

        self.image_dir = IMAGES_DIR

        self.detection_threshold = detection_threshold
        self.crop = crop


        
        if CROP_MODEL_PATH:
            # CROPPING MODEL
            # initialize the model and move to the computation device
            self.crop_classes = CLASSES_DOLPHIN
            self.start_epoch = 0
            self.all_stats = []
            self.crop_model = create_model(num_classes=len(self.crop_classes), backbone = self.crop_backbone).to(DEVICE)
            self.crop_model.load_state_dict(torch.load(
                self.crop_model_path, map_location = DEVICE
                ))

            self.start_epoch = self.crop_model_path.split('/')[-1]
            x = re.search("\d+", self.start_epoch)
            self.start_epoch = int(x.group())-1

            print("-"*50)
            print(f"Loaded model number {self.start_epoch + 1} for cropping")
            print("-"*50)
        else:
            print("PLEASE SET PATH FOR CROPPING MODEL")



        if PRED_MODEL_PATH:
            # PREDICTION MODEL
            # initialize the model and move to the computation device
            self.pred_classes = CLASSES_MARKINGS
            self.pred_model = create_model(num_classes=len(self.pred_classes), backbone = self.pred_backbone).to(DEVICE)
            self.start_epoch = 0
            self.pred_model_path = PRED_MODEL_PATH
            print(self.pred_model_path)
            self.pred_model.load_state_dict(torch.load(
                self.pred_model_path, map_location = DEVICE
                ))

            self.start_epoch = self.pred_model_path.split('/')[-1]
            x = re.search("\d+", self.start_epoch)
            self.start_epoch = int(x.group())-1

            print("-"*50)
            print(f"Loaded model number {self.start_epoch + 1} for prediction")
            print("-"*50)
        else:
            print("PLEASE SET PATH FOR PREDICTION MODEL")
    
    def predict(self, image, image_path):
        print("Making predictions...")
        self.pred_model.eval()
        image_name = os.path.basename(image_path).split('.')[-2]

        # image = cv2.imread(image).astype(np.float32)
        orig_image = image.copy()

        # make the pixel range between 0 and 1
        image /= 255.0

        # bring color channels to front
        image = np.transpose(image, (2, 0, 1)).astype(np.float64)

        # convert to tensor
        image = torch.tensor(image, dtype = torch.float).cuda()

        # add batch dimension
        image = torch.unsqueeze(image, 0)

        with torch.no_grad():
            outputs = self.pred_model(image) # outputs will consit of two tensors [targets, images]
            
        # load all detection to CPU for further operations
        outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]

        write_to_dir = os.path.join(FINAL_PREDS_DIR, 'images')
        if os.path.isdir(write_to_dir):
            save_as = os.path.join(write_to_dir, f'{image_name}.jpg')
        else:
            os.mkdir(write_to_dir)
            save_as = os.path.join(write_to_dir, f'{image_name}.jpg')
        
        # carry further only if there are detected boxes
        if len(outputs[0]['boxes']) != 0:
            boxes = outputs[0]['boxes'].data.numpy()
            scores = outputs[0]['scores'].data.numpy()
            save_preds_to = os.path.join(FINAL_PREDS_DIR)
            pred_classes = [self.pred_classes[i] for i in outputs[0]['labels'].cpu().numpy()]

            self.all_stats = save_metrics(boxes, pred_classes, scores, image_name, self.all_stats)
            save_predictions_as_txt(outputs, image_name, save_preds_to)

            # filter out boxes according to the detection threshold
            boxes = boxes[scores >= 0.3].astype(np.int32)
            draw_boxes = boxes.copy()

            # get all the predicted class names
            pred_classes = [self.pred_classes[i] for i in outputs[0]['labels'].cpu().numpy()]

            # draw the bounding boxes and write class name on top of it
            if len(draw_boxes) != 0:
                for j, box in enumerate(draw_boxes):
                    cv2.rectangle(orig_image,
                                (int(box[0]), int(box[1])),
                                (int(box[2]), int(box[3])),
                                (COLOURS[pred_classes[j]]), 5)
                    cv2.putText(orig_image, pred_classes[j].upper() + " CONF: " + str(round(scores[j], 2)), 
                                (int(box[0]), int(box[1]-5)),
                                cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 255, 255), 
                                4, lineType=cv2.LINE_AA)
                    
                    # cv2.imshow('Prediction', orig_image)
                    # cv2.waitKey(1)

                    # Save image with box prediction
                    # If > 1 box is predicted it will re-write the image with each new box added
                    cv2.imwrite(save_as, orig_image)
                    print(f"Predicitons made....Saved Image {image_name}")
                    print('-'*50)

            else:

                # If there are no bounding boxes above threshold, save orig images
                cv2.imwrite(save_as, orig_image)
                print(f"Scores too low, no boxes saved....Saved Image {image_name}")
                print('-'*50)
        else:
            print("No predictions, saving original image...")
            cv2.imwrite(save_as, orig_image)
            print(f"Saved Image {image_name}")
            print('-'*50)

        print("Image saved to: ", write_to_dir)

    def run(self):

        image_paths = glob.glob(f"{self.image_dir}/*")
        from config import TRAIN_FOR
        for image_path in image_paths:
            cropping_engine = Cropping_engine(
                BACKBONE = self.crop_backbone,
                TRAIN_FOR= TRAIN_FOR.value(),
                MODEL= self.crop_model,
                IMAGES_DIR=self.image_dir)
            cropped_image = cropping_engine.get_cropped_image(image_path)
            if cropped_image is not None:
                self.predict(cropped_image, image_path)
            else:
                print("No dolphin detected")

        # TODO: add ground truth bouding boxes to the df
        df = pd.DataFrame(columns = ["image_name", "class", "xmin_pred", "ymin_pred", "xmax_pred", "ymax_pred", "score"])
        for i in self.all_stats:
            df.loc[len(df)] = i
        
        save_df = os.path.join(self.output_dir, "all_predictions.csv")
        df.to_csv(save_df)
        print('PREDICTIONS COMPLETE')
        print("Predictions saved as: ", save_df)
        cv2.destroyAllWindows()

        # print("_"*50)
        # print(f"RUN COMPLETED. Outputs saved to: '{self.output_dir}'")

        torch.cuda.empty_cache() # PyTorch to clear GPU cache



            

        

        

        
