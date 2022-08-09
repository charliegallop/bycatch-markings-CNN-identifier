import numpy as np
import cv2
import torch
import glob as glob
import os

from model import create_model
from config import ROOT, NUM_EPOCHS, COLOURS, BACKBONE, THRESHOLD, VALID_DIR, TRAIN_FOR
from utils import save_predictions_as_txt

class Inference_engine():

    def __init__(self, BACKBONE, TRAIN_FOR, MODEL_DIR):

        self.saved_images = 0
        self.loaded_model_dir = MODEL_DIR

        if TRAIN_FOR == 'dolphin':
            from config import CLASSES_DOLPHIN, NUM_CLASSES_DOLPHIN
            self.num_classes = NUM_CLASSES_DOLPHIN
            self.classes = CLASSES_DOLPHIN

            print('-'*50)
            print("INFERING FOR DOLPHIN")
            print('-'*50)

        elif TRAIN_FOR == 'markings':
            from config import CLASSES_MARKINGS, NUM_CLASSES_MARKINGS
            num_classes = NUM_CLASSES_MARKINGS
            classes = CLASSES_MARKINGS

            print('-'*50)
            print("INFERING FOR MARKINGS")
            print('-'*50)
        else:
            print("NOT A VALID SELECTION")

        # define the detection threshold...
        #... any detection having score below this will be discarded
        self.detection_threshold = THRESHOLD
        print('-'*50)
        print(f"Infereing on model with backbone '{BACKBONE}' with detection_threshold = {self.detection_threshold}")
        print('-'*50)

        # set the computation device
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # load model and trained weights
        self.model = create_model(num_classes=self.num_classes, backbone = BACKBONE).to(self.device)
        self.model.load_state_dict(torch.load(
            self.loaded_model_dir, map_location = self.device
        ))

    def infer(self):
        self.model.eval()

        # directory where all the images are present
        DIR_TEST = os.path.join(VALID_DIR, 'valid')
        test_images = glob.glob(f"{DIR_TEST}/*")
        print(f"Test instances: {len(test_images)}")




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
            # carry further only if there are detected boxes
            if len(outputs[0]['boxes']) != 0:
                boxes = outputs[0]['boxes'].data.numpy()
                scores = outputs[0]['scores'].data.numpy()

                save_predictions_as_txt(outputs, image_name, f"{ROOT}/data/infered_images/{TRAIN_FOR}")

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
                    write_to_dir = os.path.join(ROOT,'data', 'infered_images', TRAIN_FOR, f'{image_name}.jpg')
                    cv2.imwrite(write_to_dir, orig_image)
                    if saved == False:
                        self.saved_images += 1
                        saved = True
                    print(f"Saved Image {i+1}")
                    print('-'*50)

                print(f"Image {i+1} done... {self.saved_images} saved")
                print('-'*50)

        print('TEST PREDICTIONS COMPLETE')
        cv2.destroyAllWindows()