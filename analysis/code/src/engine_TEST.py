from config import DEVICE, NUM_CLASSES_DOLPHIN, NUM_CLASSES_MARKINGS, NUM_EPOCHS, OUT_DIR
from config import THRESHOLD, CLASSES_DOLPHIN,CLASSES_MARKINGS, COLOURS, TRAIN_FOR
from config import VISUALIZE_TRANSFORMED_IMAGES
from config import SAVE_PLOTS_EPOCH, SAVE_MODEL_EPOCH
from config import VALID_PRED_DIR, VALID_DIR
from model import create_model
from utils import Averager, save_predictions_as_txt
from tqdm.auto import tqdm
from datasets import train_loader_dolphin, valid_loader_dolphin
from datasets import train_loader_markings, valid_loader_markings
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

class engine():

    def __init__(self, BACKBONE, TRAIN_FOR):
        if TRAIN_FOR == 'dolphin':
            print("DOLPHIN DATALOADER SELECTED")
            self.train_loader = train_loader_dolphin
            self.valid_loader = valid_loader_dolphin
            self.classes = CLASSES_DOLPHIN
            self.num_classes = NUM_CLASSES_DOLPHIN
        elif TRAIN_FOR == 'markings':
            self.classes = CLASSES_MARKINGS
            self.num_classes = NUM_CLASSES_MARKINGS
            self.train_loader = train_loader_markings
            self.valid_loader = valid_loader_markings
        else:
            print("NOT A VALID SELECTION")

        
        
        self.train_for = TRAIN_FOR
        # initialize the model and move to the computation device
        self.backbone = BACKBONE
        self.model = create_model(num_classes=self.num_classes, backbone = BACKBONE)
        self.model = self.model.to(DEVICE)
        # get the model parameters
        self.params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optimizer = torch.optim.SGD(self.params, lr = 0.001, momentum = 0.9, weight_decay = 0.0005)

        # initialize the Averager class
        self.train_loss_epoch = Averager()
        self.val_loss_epoch = Averager()
        self.train_itr = 1
        self.val_itr = 1
        # train and validation loss lists to store loss values of all..
        # ... iteration till end and plot graphs for all iterations
        self.train_loss_all = []
        self.val_loss_all = []

        self.output_dir = os.path.join(OUT_DIR, self.backbone, self.train_for)
    



    # def select_labels(self, selection, targets):
    #     if selection == "dolphin":
    #         for image in targets:

    #             image['boxes'] = image['boxes'][image['labels'] == 2]
    #             image['labels'] = image['labels'][image['labels'] == 2]
    #     elif selection == "markings":
    #         for image in targets:

    #             image['boxes'] = image['boxes'][image['labels'] != 2]
    #             image['labels'] = image['labels'][image['labels'] != 2]
    #     elif selection == "all":
    #         targets = targets

    #     return targets

    # function for running training iterations
    def train(self, train_data_loader, model):
        print('Training')
        global train_itr
        global train_loss_all

        # initialize tqdm progress bar
        prog_bar = tqdm(train_data_loader, total = len(train_data_loader))

        for i, data in enumerate(prog_bar):
            model.train()
            self.optimizer.zero_grad()

            images, labels = data
            #labels = select_labels(LABELS_TO_TRAIN, labels)
            images = list(image.to(DEVICE) for image in images)
            labels = [{k: v.to(DEVICE) for k, v in l.items()} for l in labels]
            
            loss_dict = self.model(images, labels)
            summed_losses = sum(loss for loss in loss_dict.values())
            loss_value = summed_losses.item()
            self.train_loss_all.append(loss_value)

            self.train_loss_epoch.send(loss_value)

            summed_losses.backward()
            self.optimizer.step()

            self.train_itr += 1

            # upgrade the loss value beside the progress bar for each iteration
            prog_bar.set_description(desc=f"Loss: {summed_losses:.4f}")
        return self.train_loss_all

    # function for running validation iterations
    def validate(self, valid_data_loader, model):
        print('Validating')
        global val_itr
        global val_loss_all

        # initialize tqdm progress bar
        prog_bar = tqdm(valid_data_loader, total = len(valid_data_loader))

        # Set model to testing mode
        model.eval()
        for i, data in enumerate(prog_bar):
            images, labels = data
            #labels = select_labels(LABELS_TO_TRAIN, labels)
            images = list(image.to(DEVICE) for image in images)
            labels = [{k: v.to(DEVICE) for k, v in l.items()} for l in labels]

            losses, detections = eval_forward(self.model, images, labels)
            summed_losses = sum(loss for loss in losses.values())
        
            loss_value = summed_losses.item()
            self.val_loss_all.append(loss_value)

            self.val_loss_epoch.send(loss_value)

            self.val_itr += 1

            # save detections to calculate metrics
            # image_data = images[0].data.cpu().numpy() * 255
            # cv2.imwrite(f"{VALID_PRED_DIR}/images/image_{val_itr}.jpg", image_data.T)
            # save_predictions_as_txt(detections, f"image_{val_itr}.jpg", f"{VALID_PRED_DIR}/predictions")
            # image_data = images[0].data.cpu().numpy() * 255
            # cv2.imwrite(f"{VALID_PRED_DIR}/images/image_{val_itr}.jpg", image_data.T)

            # update the loss value beside the progress bar for each iteration
            prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")
        return self.val_loss_all

    def calc_metrics(self, model):
        print('Validating')
        model.eval()

        # directory where all the images are present
        DIR_VAL = os.path.join(VALID_DIR, 'valid')
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
                save_predictions_as_txt(outputs, image_name, f"{VALID_PRED_DIR}/predictions/{self.train_for}")

                # filter out boxes according to the detection threshold
                boxes = boxes[scores >= THRESHOLD].astype(np.int32)
                draw_boxes = boxes.copy()
                # get all the predicted class names
                pred_classes = [self.classes[i] for i in outputs[0]['labels'].cpu().numpy()]

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
                    write_to_dir = os.path.join(VALID_PRED_DIR, "images", self.train_for, f'{image_name}.jpg')
                    cv2.imwrite(write_to_dir, orig_image)
                print(f"Image {i+1} done...")
                print('-'*50)

        print('TEST PREDICTIONS COMPLETE')
        cv2.destroyAllWindows()

    def run(self):
        print('RUNNING')
        
        print('-'*50)
        print(f"Training '{self.train_for}' annotation set on model with backbone '{self.backbone}'")
        print('-'*50)

        
        # define the optimizer
        #optimizer = torch.optim.SGD(params, lr = 0.001, momentum = 0.9, weight_decay = 0.0005)
        # define loss equation
        criterion = torch.nn.CrossEntropyLoss()

        # name to save the trained model with
        MODEL_NAME = 'model'

        # whether to show transformed images from the data loader or not
        if VISUALIZE_TRANSFORMED_IMAGES:
            from utils import show_transformed_image
            show_transformed_image(self.train_loader)
        
        # start the training epochs:
        for epoch in range(NUM_EPOCHS):
            print(f"\nEPOCH {epoch+1} of {NUM_EPOCHS}")

            #reset the training and validation loss histories for the current epoch
            self.train_loss_epoch.reset()
            self.val_loss_epoch.reset()

            # create two subplots, one for each, training and validation
            figure_1, train_ax = plt.subplots()
            figure_2, valid_ax = plt.subplots()

            # start timer and carry out training and validation
            start = time.time()
            train_loss = self.train(self.train_loader, self.model)
            val_loss = self.validate(self.valid_loader, self.model)
            print(f"Epoch #{epoch+1} train loss: {self.train_loss_epoch.value:.3f}")
            print(f"Epoch #{epoch+1} validation loss: {self.val_loss_epoch.value:.3f}")
            end = time.time()
            print(f"Took {((end - start) / 60):.3f} minutes for epoch {epoch+1}")
            #evaluate(model, valid_loader, device=DEVICE)

            if (epoch+1) % SAVE_MODEL_EPOCH == 0: # save the model after every n epoch
                torch.save(self.model.state_dict(), f"{self.output_dir}/model{epoch+1}.pth")
                print('SAVING MODEL COMPLETE...\n')

            if (epoch+1) % SAVE_PLOTS_EPOCH == 0: # save loss plots after every n epoch
                train_ax.plot(train_loss, color = 'blue')
                train_ax.set_xlabel('iterations')
                train_ax.set_ylabel('train loss')
                valid_ax.plot(val_loss, color='red')
                valid_ax.set_xlabel('iterations')
                valid_ax.set_ylabel('validation loss')
                figure_1.savefig(f"{self.output_dir}/train_loss_{epoch+1}.png")
                figure_2.savefig(f"{self.output_dir}/valid_loss_{epoch+1}.png")
                print('SAVING PLOTS COMPLETE...')
            
            if (epoch+1) == NUM_EPOCHS: # save loss plots and model once at the end
                train_ax.plot(train_loss, color = 'blue')
                train_ax.set_xlabel('iterations')
                train_ax.set_ylabel('train loss')
                valid_ax.plot(val_loss, color='red')
                valid_ax.set_xlabel('iterations')
                valid_ax.set_ylabel('validation loss')
                figure_1.savefig(f"{self.output_dir}/train_loss_{epoch+1}.png")
                figure_2.savefig(f"{self.output_dir}/valid_loss_{epoch+1}.png")
            
                torch.save(self.model.state_dict(), f"{self.output_dir}/model{epoch+1}.pth")
            
            if (epoch+1) ==  NUM_EPOCHS:
                self.calc_metrics(self.model)

            
            plt.close('all')

