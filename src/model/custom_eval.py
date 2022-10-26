from config import DEVICE, VAL_DIR, EVAL_DIR, THRESHOLD, COLOURS
from tqdm.auto import tqdm
from model_eval import eval_forward
import numpy as np
import os
import glob
import cv2
import torch
from utils import Averager, save_predictions_as_txt
from model import create_model

# function for running validation iterations
def validate(valid_data_loader, model):
    print("-"*50)
    print('Validating')
    print("-"*50)
    epoch_loss = []
    epoch_loss_classifier = []
    epoch_loss_box_reg = []
    epoch_loss_objectness = []
    epoch_loss_rpn_box_reg = []
    # initialize tqdm progress bar
    prog_bar = tqdm(valid_data_loader, total = len(valid_data_loader))

    # Set model to testing mode
    model.eval()
    for i, data in enumerate(prog_bar):
        images, labels = data
        #labels =   ect_labels(LABELS_TO_TRAIN, labels)
        images = list(image.to(DEVICE) for image in images)
        labels = [{k: v.to(DEVICE) for k, v in l.items()} for l in labels]

        losses, detections = eval_forward(model, images, labels)
        summed_losses = sum(loss for loss in losses.values())
        losses_dict = {k: v.item() for k, v in losses.items()}

        epoch_loss.append(summed_losses.item())
        
        epoch_loss_classifier.append(losses_dict['loss_classifier'])
        epoch_loss_box_reg.append(losses_dict['loss_box_reg'])
        epoch_loss_objectness.append(losses_dict['loss_objectness'])
        epoch_loss_rpn_box_reg.append(losses_dict['loss_rpn_box_reg'])


        # update the loss value beside the progress bar for each iteration
        prog_bar.set_description(desc=f"Loss: {summed_losses.item():.4f}")


    all_losses_dict = {'loss': epoch_loss,
        'loss_classifier': epoch_loss_classifier,
        'loss_box_reg': epoch_loss_box_reg,
        'loss_objectness': epoch_loss_objectness,
        'loss_rpn_box_reg': epoch_loss_rpn_box_reg}
    
    
    all_losses_dict = {k: np.mean(v) for k, v in all_losses_dict.items()}

    return all_losses_dict

def calc_metrics(model, classes):
        print("-"*50)
        print('Infering')
        print("-"*50)

        model.eval()

        # directory where all the images are present
        DIR_VAL = os.path.join(VAL_DIR, 'images')
        test_images = glob.glob(f"{DIR_VAL}/*")
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
                outputs = model(image) # outputs will consit of 3 tensors [boxes:, labels:, scores:]
            
            # load all detection to CPU for further operations
            outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
            # carry further only if there are detected boxes
            if len(outputs[0]['boxes']) != 0:
                boxes = outputs[0]['boxes'].data.numpy()
                scores = outputs[0]['scores'].data.numpy()

                # save predictions in a text file so can be used to get metrics
                save_predictions_as_txt(outputs, image_name, f"{EVAL_DIR}")

                # filter out boxes according to the detection threshold
                boxes = boxes[scores >= THRESHOLD].astype(np.int32)
                draw_boxes = boxes.copy()
                # get all the predicted class names
                pred_classes = [classes[i] for i in outputs[0]['labels'].cpu().numpy()]

                # draw the bounding boxes and write class name on top of it
                for j, box in enumerate(draw_boxes):
                    cv2.rectangle(orig_image,
                                (int(box[0]), int(box[1])),
                                (int(box[2]), int(box[3])),
                                (COLOURS[pred_classes[j]]), 3)
                    cv2.putText(orig_image, pred_classes[j].upper() + " CONF: " + str(round(scores[j], 2)), 
                                (int(box[0]), int(box[1]-5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 
                                2, lineType=cv2.LINE_AA)
                    
                    # cv2.imshow('Prediction', orig_image)
                    # cv2.waitKey(1)
                    write_to_dir = os.path.join(EVAL_DIR, "images", f'{image_name}.jpg')
                    cv2.imwrite(write_to_dir, orig_image)
                    print(f"Image {i+1} image saved...")
                print(f"Image {i+1} done...")
                print('-'*50)

        print('TEST PREDICTIONS COMPLETE')
        cv2.destroyAllWindows()