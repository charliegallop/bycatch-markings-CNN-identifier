from config import DEVICE, NUM_CLASSES_DOLPHIN, NUM_CLASSES_MARKINGS, NUM_EPOCHS, OUT_DIR
from config import THRESHOLD, CLASSES_DOLPHIN,CLASSES_MARKINGS, COLOURS, TRAIN_FOR
from config import VISUALIZE_TRANSFORMED_IMAGES, BACKBONE
from config import SAVE_PLOTS_EPOCH, SAVE_MODEL_EPOCH
from config import VALID_PRED_DIR, VALID_DIR, ROOT
from model import create_model
from utils import Averager, save_predictions_as_txt
from tqdm.auto import tqdm
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
import pandas as pd
import re

plt.style.use('Solarize_Light2')

class engine():

    def __init__(self, BACKBONE, TRAIN_FOR, EPOCHS, MODEL_PATH = None):
        
        if TRAIN_FOR == 'dolphin':
            from datasets import train_loader_dolphin, valid_loader_dolphin
            
            print('-'*50)
            print("DOLPHIN DATALOADER SELECTED")
            print('-'*50)

            self.train_loader = train_loader_dolphin
            self.valid_loader = valid_loader_dolphin
            self.classes = CLASSES_DOLPHIN
            self.num_classes = NUM_CLASSES_DOLPHIN
        elif TRAIN_FOR == 'markings':
            from datasets import train_loader_markings, valid_loader_markings

            print('-'*50)
            print("MARKINGS DATALOADER SELECTED")
            print('-'*50)

            self.classes = CLASSES_MARKINGS
            self.num_classes = NUM_CLASSES_MARKINGS
            self.train_loader = train_loader_markings
            self.valid_loader = valid_loader_markings
        else:
            print("NOT A VALID SELECTION")
        
        self.train_for = TRAIN_FOR
        self.backbone = BACKBONE
        self.output_dir = os.path.join(OUT_DIR, self.backbone, self.train_for)
        self.learning_rate = 0.001
        
        self.num_epochs = EPOCHS
        print("Num of Epochs: " ,self.num_epochs)
        # initialize the model and move to the computation device
        self.model = create_model(num_classes=self.num_classes, backbone = BACKBONE).to(DEVICE)
        self.start_epoch = 0
        self.model_path = MODEL_PATH

        # train and validation loss lists to store loss values of all..
        # ... iteration till end and plot graphs for all iterations
        self.train_loss_all = []
        self.val_loss_all = []
        self.lr_all = []
        self.train_loss_classifier_all = []
        self.train_loss_box_reg_all = []
        self.train_loss_objectness_all = []
        self.train_loss_rpn_box_reg_all = []

        self.val_loss_classifier_all = []
        self.val_loss_box_reg_all = []
        self.val_loss_objectness_all = []
        self.val_loss_rpn_box_reg_all = []

        if MODEL_PATH:
            self.model_path = MODEL_PATH
            self.model.load_state_dict(torch.load(
                self.model_path, map_location = DEVICE
                ))

            self.start_epoch = self.model_path.split('/')[-1]
            x = re.search("\d+", self.start_epoch)
            self.start_epoch = int(x.group())-1

            print("-"*50)
            print(f"Loaded model number {self.start_epoch + 1}")
            print("-"*50)
            print(f"Starting from epoch {self.start_epoch}")
            print("-"*50)
            
            
            
            self.loss_hist =glob.glob(f"{self.output_dir}/losses_df.csv")[0]
            if self.loss_hist:
                self.loss_hist_df = pd.read_csv(self.loss_hist)

                self.train_loss_all = self.loss_hist_df['train_loss'].tolist()
                self.lr_all = self.loss_hist_df['lr'].tolist()
                self.train_loss_classifier_all = self.loss_hist_df['train_loss_classifier'].tolist()
                self.train_loss_box_reg_all = self.loss_hist_df['train_loss_box_reg'].tolist()
                self.train_loss_objectness_all = self.loss_hist_df['train_loss_objectness'].tolist()
                self.train_loss_rpn_box_reg_all = self.loss_hist_df['train_loss_rpn_box_reg'].tolist()

                self.val_loss_all = self.loss_hist_df['val_loss'].tolist()
                self.val_loss_classifier_all = self.loss_hist_df['val_loss_classifier'].tolist()
                self.val_loss_box_reg_all = self.loss_hist_df['val_loss_box_reg'].tolist()
                self.val_loss_objectness_all = self.loss_hist_df['val_loss_objectness'].tolist()
                self.val_loss_rpn_box_reg_all = self.loss_hist_df['val_loss_rpn_box_reg'].tolist()



                print(f"Loaded loss history from {self.loss_hist}. Num of losses: [{len(self.train_loss_all)}, {len(self.val_loss_all)}]")
                print("-"*50)
                self.learning_rate = self.lr_all[-1]

        # get the model parameters
        self.params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optimizer = torch.optim.SGD(self.params, lr = self.learning_rate, momentum = 0.9, weight_decay = 0.0005)
        #self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 25, gamma = 0.1)
        self.lr_scheduler_RLROP = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', min_lr = 0.000001, factor = 0.5)

        # initialize the Averager class
        self.train_loss_epoch = Averager()
        self.val_loss_epoch = Averager()
        self.train_itr = 1
        self.val_itr = 1

    

    # function for running training iterations
    def train(self, train_data_loader, model):
        print("-"*50)
        print('Training')
        print("-"*50)
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
        print("-"*50)
        print('Validating')
        print("-"*50)
        global val_itr
        global val_loss_all
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
            #labels = select_labels(LABELS_TO_TRAIN, labels)
            images = list(image.to(DEVICE) for image in images)
            labels = [{k: v.to(DEVICE) for k, v in l.items()} for l in labels]

            losses, detections = eval_forward(self.model, images, labels)
            summed_losses = sum(loss for loss in losses.values())
            losses_dict = {k: v.item() for k, v in losses.items()}

            epoch_loss.append(summed_losses.item())
            
            epoch_loss_classifier.append(losses_dict['loss_classifier'])
            epoch_loss_box_reg.append(losses_dict['loss_box_reg'])
            epoch_loss_objectness.append(losses_dict['loss_objectness'])
            epoch_loss_rpn_box_reg.append(losses_dict['loss_rpn_box_reg'])

            self.val_itr += 1

            # update the loss value beside the progress bar for each iteration
            prog_bar.set_description(desc=f"Loss: {summed_losses.item():.4f}")


        all_losses_dict = {'loss': epoch_loss,
            'loss_classifier': epoch_loss_classifier,
            'loss_box_reg': epoch_loss_box_reg,
            'loss_objectness': epoch_loss_objectness,
            'loss_rpn_box_reg': epoch_loss_rpn_box_reg}
        
        
        all_losses_dict = {k: np.mean(v) for k, v in all_losses_dict.items()}
    
        return all_losses_dict

    def calc_metrics(self, model):
        print("-"*50)
        print('Infering')
        print("-"*50)

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
                    cv2.putText(orig_image, pred_classes[j].upper() + " CONF: " + str(round(scores[j], 2)), 
                                (int(box[0]), int(box[1]-5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 
                                2, lineType=cv2.LINE_AA)
                    
                    # cv2.imshow('Prediction', orig_image)
                    # cv2.waitKey(1)
                    write_to_dir = os.path.join(VALID_PRED_DIR, "images", self.train_for, f'{image_name}.jpg')
                    cv2.imwrite(write_to_dir, orig_image)
                    print(f"Image {i+1} image saved...")
                print(f"Image {i+1} done...")
                print('-'*50)

        print('TEST PREDICTIONS COMPLETE')
        cv2.destroyAllWindows()

    def run(self):
        print('-'*50)
        print(f"Training '{self.train_for}' annotation set on model with backbone '{self.backbone}'")
        print('-'*50)

        # define loss equation
        criterion = torch.nn.CrossEntropyLoss()

        # name to save the trained model with
        MODEL_NAME = 'model'

        # whether to show transformed images from the data loader or not
        if VISUALIZE_TRANSFORMED_IMAGES:
            from utils import show_transformed_image
            show_transformed_image(self.train_loader)
        
        # start the training epochs:
        if self.start_epoch < self.num_epochs:
            for epoch in range(self.start_epoch, self.num_epochs):
                print(f"\nEPOCH {epoch+1} of {self.num_epochs}")

                #reset the training and validation loss histories for the current epoch
                self.train_loss_epoch.reset()
                self.val_loss_epoch.reset()

                # create two subplots, one for each, training and validation
                figure_1, train_ax = plt.subplots()
                figure_2, valid_ax = plt.subplots()

                losses = train_one_epoch(self.model, self.optimizer, self.train_loader, DEVICE, epoch, 10)
                summed_losses = losses.meters['loss'].value
                lr = losses.meters['lr'].value
                loss_classifier = losses.meters['loss_classifier'].value
                loss_box_reg = losses.meters['loss_box_reg'].value
                loss_objectness = losses.meters['loss_objectness'].value
                loss_rpn_box_reg = losses.meters['loss_rpn_box_reg'].value

                self.train_loss_all.append(summed_losses)
                self.lr_all.append(lr)
                self.train_loss_classifier_all.append(loss_classifier)
                self.train_loss_box_reg_all.append(loss_box_reg)
                self.train_loss_objectness_all.append(loss_objectness)
                self.train_loss_rpn_box_reg_all.append(loss_rpn_box_reg)

                evaluate(self.model, self.valid_loader, DEVICE)

                val_losses = self.validate(self.valid_loader, self.model)
                val_loss = val_losses['loss']
                val_loss_classifier = val_losses['loss_classifier']
                val_loss_box_reg = val_losses['loss_box_reg']
                val_loss_objectness = val_losses['loss_objectness']
                val_loss_rpn_box_reg = val_losses['loss_rpn_box_reg']

                self.val_loss_all.append(val_loss)
                self.val_loss_classifier_all.append(val_loss_classifier)
                self.val_loss_box_reg_all.append(val_loss_box_reg)
                self.val_loss_objectness_all.append(val_loss_objectness)
                self.val_loss_rpn_box_reg_all.append(val_loss_rpn_box_reg)
            
                #self.lr_scheduler.step()
                self.lr_scheduler_RLROP.step(val_loss)

                print(f"Epoch #{epoch+1} train loss: {summed_losses:.3f}")
                print(f"Epoch #{epoch+1} validation loss: {val_loss:.3f}")
                print("_"*50)


                if (epoch+1) % SAVE_MODEL_EPOCH == 0: # save the model after every n epoch
                    torch.save(self.model.state_dict(), f"{self.output_dir}/model{epoch+1}.pth")
                    print('SAVING MODEL COMPLETE...\n')
                    print("_"*50)

                    

                if (epoch+1) % SAVE_PLOTS_EPOCH == 0: # save loss plots after every n epoch
                    train_ax.plot(self.train_loss_all, color = 'dodgerblue')
                    train_ax.set_xlabel('Epochs')
                    train_ax.set_ylabel('train loss')
                    valid_ax.plot(self.val_loss_all, color='peru')
                    valid_ax.set_xlabel('Epochs')
                    valid_ax.set_ylabel('validation loss')
                    figure_1.savefig(f"{self.output_dir}/train_loss_{epoch+1}.png")
                    figure_2.savefig(f"{self.output_dir}/valid_loss_{epoch+1}.png")
                    print('SAVING PLOTS COMPLETE...')
                    print("_"*50)

                    torch.save(self.model.state_dict(), f"{self.output_dir}/model{epoch+1}.pth")

                    losses_df = pd.DataFrame()
                    losses_df['train_loss'] = self.train_loss_all
                    losses_df['lr'] = self.lr_all
                    losses_df['train_loss_classifier'] = self.train_loss_classifier_all
                    losses_df['train_loss_box_reg'] = self.train_loss_box_reg_all
                    losses_df['train_loss_objectness'] = self.train_loss_objectness_all
                    losses_df['train_loss_rpn_box_reg'] = self.train_loss_rpn_box_reg_all

                    losses_df['val_loss'] = self.val_loss_all
                    losses_df['val_loss_classifier'] = self.val_loss_classifier_all
                    losses_df['val_loss_box_reg'] = self.val_loss_box_reg_all
                    losses_df['val_loss_objectness'] = self.val_loss_objectness_all
                    losses_df['val_loss_rpn_box_reg'] = self.val_loss_rpn_box_reg_all


                    losses_df.to_csv(f"{self.output_dir}/losses_df.csv")
                    print('SAVING LOSSES AS CSV COMPLETE...')
                    print("_"*50)


                if epoch+1 == self.num_epochs:
                    losses_df = pd.DataFrame()
                    losses_df['train_loss'] = self.train_loss_all
                    losses_df['lr'] = self.lr_all
                    losses_df['train_loss_classifier'] = self.train_loss_classifier_all
                    losses_df['train_loss_box_reg'] = self.train_loss_box_reg_all
                    losses_df['train_loss_objectness'] = self.train_loss_objectness_all
                    losses_df['train_loss_rpn_box_reg'] = self.train_loss_rpn_box_reg_all

                    losses_df['val_loss'] = self.val_loss_all
                    losses_df['val_loss_classifier'] = self.val_loss_classifier_all
                    losses_df['val_loss_box_reg'] = self.val_loss_box_reg_all
                    losses_df['val_loss_objectness'] = self.val_loss_objectness_all
                    losses_df['val_loss_rpn_box_reg'] = self.val_loss_rpn_box_reg_all


                    train_ax.plot(losses_df['train_loss'], color = 'dodgerblue')
                    train_ax.set_xlabel('Epochs')
                    train_ax.set_ylabel('train loss')

                    valid_ax.plot(losses_df['val_loss'], color='peru')
                    valid_ax.set_xlabel('Epochs')
                    valid_ax.set_ylabel('validation loss')

                    print(losses_df['train_loss'])
                    figure_1.savefig(f"{self.output_dir}/train_loss_{self.num_epochs}.png")
                    figure_2.savefig(f"{self.output_dir}/valid_loss_{self.num_epochs}.png")
                    print('SAVING FINAL PLOTS COMPLETE...')
                    print("_"*50)


                    losses_df.to_csv(f"{self.output_dir}/losses_df.csv")
                    print('SAVING LOSSES AS CSV COMPLETE...')
                    print("_"*50)

                plt.close('all')
                    

        self.calc_metrics(self.model)

        print("_"*50)
        print(f"RUN COMPLETED. Outputs saved to: '{self.output_dir}'")


            

        

        

        
