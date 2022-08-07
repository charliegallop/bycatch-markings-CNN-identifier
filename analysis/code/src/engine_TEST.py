
from config import DEVICE, NUM_CLASSES, NUM_EPOCHS, OUT_DIR, BACKBONE, THRESHOLD, CLASSES, COLOURS, LABELS_TO_TRAIN
from config import VISUALIZE_TRANSFORMED_IMAGES
from config import SAVE_PLOTS_EPOCH, SAVE_MODEL_EPOCH
from config import VALID_PRED_DIR, VALID_DIR
from model import create_model
from utils import Averager, save_predictions_as_txt
from tqdm.auto import tqdm
from datasets import train_loader, valid_loader
from torch_utils.engine import (
    train_one_epoch, evaluate
)

from model_eval import eval_forward

import torch
import matplotlib.pyplot as plt
import time
import cv2
import os
import glob
import numpy as np

plt.style.use('ggplot')

def select_labels(selection, targets):
    if selection == "dolphin":
        for image in targets:

            image['boxes'] = image['boxes'][image['labels'] == 2]
            image['labels'] = image['labels'][image['labels'] == 2]
    elif selection == "markings":
        for image in targets:

            image['boxes'] = image['boxes'][image['labels'] != 2]
            image['labels'] = image['labels'][image['labels'] != 2]
    elif selection == "all":
        targets = targets

    return targets

# function for running training iterations
def train(train_data_loader, model):
    print('Training')
    global train_itr
    global train_loss_all

    # initialize tqdm progress bar
    prog_bar = tqdm(train_data_loader, total = len(train_data_loader))

    for i, data in enumerate(prog_bar):
        model.train()
        optimizer.zero_grad()

        images, labels = data
        labels = select_labels(LABELS_TO_TRAIN, labels)
        images = list(image.to(DEVICE) for image in images)
        labels = [{k: v.to(DEVICE) for k, v in l.items()} for l in labels]
        
        loss_dict = model(images, labels)
        summed_losses = sum(loss for loss in loss_dict.values())
        loss_value = summed_losses.item()
        train_loss_all.append(loss_value)

        train_loss_epoch.send(loss_value)

        summed_losses.backward()
        optimizer.step()

        train_itr += 1

        # upgrade the loss value beside the progress bar for each iteration
        prog_bar.set_description(desc=f"Loss: {summed_losses:.4f}")
    return train_loss_all

# function for running validation iterations
def validate(valid_data_loader, model):
    print('Validating')
    global val_itr
    global val_loss_all

    # initialize tqdm progress bar
    prog_bar = tqdm(valid_data_loader, total = len(valid_data_loader))

    # Set model to testing mode
    model.eval()

    for i, data in enumerate(prog_bar):
        images, labels = data
        labels = select_labels(LABELS_TO_TRAIN, labels)
        images = list(image.to(DEVICE) for image in images)
        labels = [{k: v.to(DEVICE) for k, v in l.items()} for l in labels]

        losses, detections = eval_forward(model, images, labels)

        summed_losses = sum(loss for loss in losses.values())
    
        loss_value = summed_losses.item()
        val_loss_all.append(loss_value)

        val_loss_epoch.send(loss_value)

        val_itr += 1

        # save detections to calculate metrics
        # image_data = images[0].data.cpu().numpy() * 255
        # cv2.imwrite(f"{VALID_PRED_DIR}/images/image_{val_itr}.jpg", image_data.T)
        # save_predictions_as_txt(detections, f"image_{val_itr}.jpg", f"{VALID_PRED_DIR}/predictions")
        # image_data = images[0].data.cpu().numpy() * 255
        # cv2.imwrite(f"{VALID_PRED_DIR}/images/image_{val_itr}.jpg", image_data.T)

        # update the loss value beside the progress bar for each iteration
        prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")
    return val_loss_all

def calc_metrics(model):
    print('Validating')
    model = model
    model.eval()

    # directory where all the images are present
    DIR_VAL = os.path.join(VALID_DIR)
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
            # print(outputs, test_images[i])

            # save predictions in a text file so can be used to get metrics
            save_predictions_as_txt(outputs, image_name, f"{VALID_PRED_DIR}/predictions")

            # filter out boxes according to the detection threshold
            boxes = boxes[scores >= THRESHOLD].astype(np.int32)
            draw_boxes = boxes.copy()
            # get all the predicted class names
            pred_classes = [CLASSES[i] for i in outputs[0]['labels'].cpu().numpy()]

            # draw the bounding boxes and write class name on top of it
            for j, box in enumerate(draw_boxes):
                cv2.rectangle(orig_image,
                            (int(box[0]), int(box[1])),
                            (int(box[2]), int(box[3])),
                            (COLOURS[pred_classes[j]]), 3)
                cv2.putText(orig_image, pred_classes[j] + str(scores[j]), 
                            (int(box[0]), int(box[1]-5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 
                            2, lineType=cv2.LINE_AA)
                
                # cv2.imshow('Prediction', orig_image)
                # cv2.waitKey(1)
                write_to_dir = os.path.join(VALID_PRED_DIR, "images", f'{image_name}.jpg')
                cv2.imwrite(write_to_dir, orig_image)
            print(f"Image {i+1} done...")
            print('-'*50)

    print('TEST PREDICTIONS COMPLETE')
    cv2.destroyAllWindows()


if __name__ == '__main__':

    print('-'*50)
    print(f"Training on model with backbone '{BACKBONE}'")
    print('-'*50)

    # initialize the model and move to the computation device
    model = create_model(num_classes=NUM_CLASSES, backbone = BACKBONE)
    model = model.to(DEVICE)
    # get the model parameters
    params = [p for p in model.parameters() if p.requires_grad]

    # define the optimizer
    optimizer = torch.optim.SGD(params, lr = 0.001, momentum = 0.9, weight_decay = 0.0005)
    # define loss equation
    criterion = torch.nn.CrossEntropyLoss()


    # initialize the Averager class
    train_loss_epoch = Averager()
    val_loss_epoch = Averager()
    train_itr = 1
    val_itr = 1
    # train and validation loss lists to store loss values of all..
    # ... iteration till end and plot graphs for all iterations
    train_loss_all = []
    val_loss_all = []

    # name to save the trained model with
    MODEL_NAME = 'model'

    # whether to show transformed images from the data loader or not
    if VISUALIZE_TRANSFORMED_IMAGES:
        from utils import show_transformed_image
        show_transformed_image(train_loader)
    
    # start the training epochs:
    for epoch in range(NUM_EPOCHS):
        print(f"\nEPOCH {epoch+1} of {NUM_EPOCHS}")

        #reset the training and validation loss histories for the current epoch
        train_loss_epoch.reset()
        val_loss_epoch.reset()

        # create two subplots, one for each, training and validation
        figure_1, train_ax = plt.subplots()
        figure_2, valid_ax = plt.subplots()

        # start timer and carry out training and validation
        start = time.time()
        train_loss = train(train_loader, model)
        val_loss = validate(valid_loader, model)
        print(f"Epoch #{epoch+1} train loss: {train_loss_epoch.value:.3f}")
        print(f"Epoch #{epoch+1} validation loss: {val_loss_epoch.value:.3f}")
        end = time.time()
        print(f"Took {((end - start) / 60):.3f} minutes for epoch {epoch+1}")
        #evaluate(model, valid_loader, device=DEVICE)

        if (epoch+1) % SAVE_MODEL_EPOCH == 0: # save the model after every n epoch
            torch.save(model.state_dict(), f"{OUT_DIR}/model{epoch+1}.pth")
            print('SAVING MODEL COMPLETE...\n')

        if (epoch+1) % SAVE_PLOTS_EPOCH == 0: # save loss plots after every n epoch
            train_ax.plot(train_loss, color = 'blue')
            train_ax.set_xlabel('iterations')
            train_ax.set_ylabel('train loss')
            valid_ax.plot(val_loss, color='red')
            valid_ax.set_xlabel('iterations')
            valid_ax.set_ylabel('validation loss')
            figure_1.savefig(f"{OUT_DIR}/train_loss_{epoch+1}.png")
            figure_2.savefig(f"{OUT_DIR}/valid_loss_{epoch+1}.png")
            print('SAVING PLOTS COMPLETE...')
        
        if (epoch+1) == NUM_EPOCHS: # save loss plots and model once at the end
            train_ax.plot(train_loss, color = 'blue')
            train_ax.set_xlabel('iterations')
            train_ax.set_ylabel('train loss')
            valid_ax.plot(val_loss, color='red')
            valid_ax.set_xlabel('iterations')
            valid_ax.set_ylabel('validation loss')
            figure_1.savefig(f"{OUT_DIR}/train_loss_{epoch+1}.png")
            figure_2.savefig(f"{OUT_DIR}/valid_loss_{epoch+1}.png")
        
            torch.save(model.state_dict(), f"{OUT_DIR}/model{epoch+1}.pth")
        
        if (epoch+1) ==  NUM_EPOCHS:
            calc_metrics(model)

        
        plt.close('all')
