import numpy as np
import pandas as pd
import cv2
import torch
import glob as glob
import os

from model import create_model
from config import ROOT, NUM_EPOCHS, COLOURS, BACKBONE, THRESHOLD, TEST_DIR, TRAIN_FOR, EVAL_DIR, VAL_DIR, OUT_DIR, TRAIN_FOR
from utils import save_predictions_as_txt, save_metrics

class Inference_engine():

    def __init__(self, BACKBONE, TRAIN_FOR, MODEL_DIR):

        self.saved_images = 0
        self.loaded_model_dir = MODEL_DIR
        self.output_dir = os.path.join(OUT_DIR, TRAIN_FOR, BACKBONE)

        if TRAIN_FOR == 'dolphin':
            from config import CLASSES_DOLPHIN, NUM_CLASSES_DOLPHIN
            self.num_classes = NUM_CLASSES_DOLPHIN
            self.classes = CLASSES_DOLPHIN
            self.crop = True

            print('-'*50)
            print("INFERING FOR DOLPHIN")
            print('-'*50)

        elif TRAIN_FOR == 'markings':
            from config import CLASSES_MARKINGS, NUM_CLASSES_MARKINGS
            self.num_classes = NUM_CLASSES_MARKINGS
            self.classes = CLASSES_MARKINGS
            self.crop = False

            print('-'*50)
            print("INFERING FOR MARKINGS")
            print('-'*50)
        else:
            print("NOT A VALID SELECTION")

        # define the detection threshold...
        #... any detection having score below this will be discarded
        self.detection_threshold = THRESHOLD
        print('-'*50)
        print(f"Infering for {TRAIN_FOR} with model backbone '{BACKBONE}' with detection_threshold = {self.detection_threshold}")
        print('-'*50)

        # set the computation device
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # load model and trained weights
        self.model = create_model(num_classes=self.num_classes, backbone = BACKBONE).to(self.device)
        self.model.load_state_dict(torch.load(
            self.loaded_model_dir, map_location = self.device
        ))

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

        if ((coords['ymin']  - img_height*buffer) >= 0):
            coords['ymin'] = int(coords['ymin']  - img_height*buffer)
        if ((coords['xmin']  - img_width*buffer) >= 0):
            coords['xmin'] = int(coords['xmin']  - img_width*buffer)
        if ((coords['ymax']  + img_height*buffer) >= 0):
            coords['ymax'] = int(coords['ymax']  + img_height*buffer)
        if ((coords['xmax']  + img_width*buffer) >= 0):
            coords['xmax'] = int(coords['xmax']  + img_width*buffer)
            
        crop = img[coords['ymin']:coords['ymax'], coords['xmin']:coords['xmax']]
        return crop

    def infer(self):
        self.model.eval()

        # directory where all the images are present
        DIR_TEST = os.path.join(TEST_DIR, 'images')
        test_images = glob.glob(f"{DIR_TEST}/*")
        print(f"Test instances: {len(test_images)}")

        self.all_stats = []

        # Looping over the image paths and carrying out inference
        for i in range(len(test_images)):
            # get the image file name for saving output later on
            image_name = test_images[i].split('/')[-1].split('.')[0]
            image = cv2.imread(test_images[i]).astype(np.float32)
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
                save_preds_to = os.path.join(EVAL_DIR, TRAIN_FOR.value(), BACKBONE.value())
                pred_classes = [self.classes[i] for i in outputs[0]['labels'].cpu().numpy()]

                self.all_stats = save_metrics(boxes, pred_classes, scores, image_name, self.all_stats)
                save_predictions_as_txt(outputs, image_name, save_preds_to)

                # filter out boxes according to the detection threshold
                boxes = boxes[scores >= self.detection_threshold].astype(np.int32)
                draw_boxes = boxes.copy()
                # get all the predicted class names
                pred_classes = [self.classes[i] for i in outputs[0]['labels'].cpu().numpy()]
                # draw the bounding boxes and write class name on top of it
                saved = False
                for j, box in enumerate(draw_boxes):
                    cv2.rectangle(orig_image,
                                (int(box[0]), int(box[1])),
                                (int(box[2]), int(box[3])),
                                (COLOURS[pred_classes[j]]), 3)
                    cv2.putText(orig_image, pred_classes[j].upper() + " CONF: " + str(round(scores[j], 2)), 
                                (int(box[0]), int(box[1]-5)),
                                cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 255, 255), 
                                2, lineType=cv2.LINE_AA)
                    
                    # cv2.imshow('Prediction', orig_image)
                    # cv2.waitKey(1)

                    # Save image with box prediction
                    # If > 1box is predicted it will re-write the image with each new box added
                    write_to_dir = os.path.join(EVAL_DIR, TRAIN_FOR.value(), BACKBONE.value(), 'images', f'{image_name}.jpg')
                    cv2.imwrite(write_to_dir, orig_image)
                    
                    if saved == False:
                        self.saved_images += 1
                        saved = True
                    print(f"Saved Image {i+1}")
                    print('-'*50)

                    # only do this is self.crop is true which should only be for dophins
                    if self.crop:
                        area = self._getArea(box)
                        if area > max_area:
                            max_area = area
                            box_to_crop = box

                print(f"Image {i+1} done... {self.saved_images} saved")
                print('-'*50)

            if (self.crop) & (len(box_to_crop) != 0):
                print("Cropping image....")
                cropped_img = self.crop_image(orig_image, box_to_crop)
                write_to_dir = os.path.join(EVAL_DIR, TRAIN_FOR.value(), BACKBONE.value(), 'images', f'{image_name}_cropped.jpg')
                cv2.imwrite(write_to_dir, cropped_img)
                print("Croped image saved!")
        # move edit xml files with only dolphin label to directory
        copy_to = os.path.join(MASTER_MARKINGS_DIR, 'labels')
        keep_labels(
                label_dir=self.labels_dir, 
                WRITE_TO=copy_to, 
                label_to_keep=TRAIN_FOR
                )

        # TODO: add ground truth bouding boxes to the df
        df = pd.DataFrame(columns = ["image_name", "class", "xmin_pred", "ymin_pred", "xmax_pred", "ymax_pred", "score"])
        for i in self.all_stats:
            df.loc[len(df)] = i
        
        save_df = os.path.join(self.output_dir, "all_predictions.csv")
        df.to_csv(save_df)
        print('TEST PREDICTIONS COMPLETE')
        cv2.destroyAllWindows()