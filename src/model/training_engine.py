from config import DEVICE, NUM_CLASSES_DOLPHIN, NUM_CLASSES_MARKINGS, NUM_EPOCHS, OUT_DIR, NUM_WORKERS
from config import THRESHOLD, CLASSES_DOLPHIN,CLASSES_MARKINGS, COLOURS, BATCH_SIZE, RESIZE_TO
from config import VISUALIZE_TRANSFORMED_IMAGES
from config import SAVE_PLOTS_EPOCH, SAVE_MODEL_EPOCH
from config import VAL_DIR, ROOT
from model import create_model
from utils import Averager
from tqdm.auto import tqdm
from torch_utils.engine import (
    train_one_epoch, evaluate
)

from custom_eval import validate

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

    def __init__(self, backbone = None, model_path = None, images_dir = None, labels_dir = None, epochs = None, batch_size = None, train_for = None, resize = None):

        self.backbone = backbone
        self.model_path = model_path
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.num_epochs = epochs
        self.batch_size = batch_size
        self.train_for = train_for
        self.resize = resize

        # Save the trained model .pth files and .csv files to this directory
        self.output_dir = os.path.join("/home/charlie/Documents/Uni/Exeter - Data Science/Dissertation/models/runs")
        existing_runs = os.listdir(self.output_dir)

        next_run = 0
        for run in existing_runs:
            temp = run.split("exp")[-1]
            if int(temp) > 0:
                next_run = int(temp)
            else:
                next_run = next_run

        self.output_folder = "exp" + str(next_run+1)

        self.output_save_to = os.path.join(self.output_dir, self.output_folder)

        if not os.path.isdir(self.output_save_to):
            os.mkdir(self.output_save_to)

        # Select classes list depending on what classes are being trained    
        if self.train_for == 'dolphin':
            
            print('-'*50)
            print("DOLPHIN DATALOADER SELECTED")
            print('-'*50)

            
            self.classes = CLASSES_DOLPHIN
            self.num_classes = len(self.classes)
        elif self.train_for == 'markings':

            print('-'*50)
            print("MARKINGS DATALOADER SELECTED")
            print('-'*50)

            self.classes = CLASSES_MARKINGS
            self.num_classes = len(self.classes)
        elif self.train_for == "all":
            print('-'*50)
            print("ALL CLASSES DATALOADER SELECTED")
            print('-'*50)

            self.classes = CLASSES_ALL
            self.num_classes = len(self.classes)
        else:
            print("NOT A VALID SELECTION")
        
        print("Num of Epochs: " ,self.num_epochs)

        # initialize the model and move to the computation device
        self.model = create_model(num_classes=self.num_classes, backbone = self.backbone).to(DEVICE)
        self.start_epoch = 0
        self.learning_rate = 0.001
        self.momentum = 0.9

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
        self.map_stats = []

        # Loads the pre-trained weights if a model path is specified
        if self.model_path:
            self.model.load_state_dict(torch.load(
                self.model_path, map_location = DEVICE
                ))

            # Resume from previous amount of epochs
            self.start_epoch = self.model_path.split('/')[-1]
            x = re.search("\d+", self.start_epoch)
            self.start_epoch = int(x.group())-1

            print("-"*50)
            print(f"Loaded model number {self.start_epoch + 1}")
            print("-"*50)
            print(f"Starting from epoch {self.start_epoch}")
            print("-"*50)
            
            # Load loss previous loss history file
            loss_hist_find = os.path.join(os.path.dirname(self.model_path), "losses_df.csv")
            self.loss_hist =glob.glob(loss_hist_find)[0]

            if self.loss_hist:
                self.loss_hist_df = pd.read_csv(self.loss_hist)

                # save previous losses as attributes to continue saving from epoch
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
        self.optimizer = torch.optim.SGD(
            self.params, 
            lr = self.learning_rate, 
            momentum = self.momentum, 
            nesterov=True ,
            weight_decay = 0.0005
            )
        
        # Set scheduler
        #self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 25, gamma = 0.1)
        # self.lr_scheduler_RLROP = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', min_lr = 0.000001, factor = 0.5)
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma = 0.96, verbose = True)

        # initialize the Averager class
        self.train_loss_epoch = Averager()
        self.val_loss_epoch = Averager()
        self.train_itr = 1
        self.val_itr = 1

        
    def save_map_stats(self, stats):

        """ Extract the mAp stats while training the model and save them in a csv file"""

        headers = [
            "AP_0.5-0.95", 
            "AP_0.5", 
            "AP_0.75",
            "AP_0.5-0.95_small", 
            "AP_0.5-0.95_medium", 
            "AP_0.5-0.95_large", 
            "AR_0.5-0.95_1Dets", 
            "AR_0.5-0.95_10Dets", 
            "AR_0.5-0.95_100Dets",
            "AR_0.5-0.95_small",
            "AR_0.5-0.95_medium", 
            "AR_0.5-0.95_large"
            ]
        df = pd.DataFrame(stats, columns=headers)
        save_df_as = os.path.join(self.output_save_to, "mAp_stats.csv")
        df.to_csv(save_df_as)

    def run(self):

        """ Start training the model"""

        # Print summary of the model to get parameters and architecture etc
        from torchinfo import summary
        print("-"*50)
        print(summary(self.model))
        print("-"*50)


        # TODO: Commented out for now but should implement button in the app that can do this
        # Choose whether to create new test/train/val sets from the master data directory
        # prompt = input("Do you want to resplit the dataset? [y][n]")

        # if prompt.lower() == "y":
        #     print('_'*50)
        #     print('Reconfiguring datasets....')
        #     if self.train_for == 'dolphin':
        #         import split_datasets
        #         split_datasets.main()
        #     elif self.train_for == 'markings':
        #         import split_datasets
        #         split_datasets.main_markings()
        #     print('_'*50)

        # else:
        #     print('_'*50)
        #     print("Keeping previous datasets")

        from datasets import CreateDataLoaders
        dl = CreateDataLoaders(
            train_for = self.train_for, 
            images_dir = self.images_dir, 
            labels_dir = self.labels_dir, 
            classes = self.classes, 
            batch_size = self.batch_size, 
            resize = self.resize, 
            backbone = self.backbone
            )
        self.train_loader, self.valid_loader = dl.get_data_loaders()

        print('-'*50)
        print(f"Training with '{self.backbone}' backbone to detect '{self.train_for}'")
        print(f"Batch Size: '{BATCH_SIZE}'..........Resize To: '{RESIZE_TO}'")
        print('-'*50)

        # # whether to show transformed images from the data loader or not
        # if VISUALIZE_TRANSFORMED_IMAGES:
        #     from utils import show_transformed_image
        #     show_transformed_image(self.train_loader)
        
        # name to save the trained model with
        MODEL_NAME = f'{self.backbone}_{self.train_for}_model'

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

                coco_eval = evaluate(self.model, self.valid_loader, DEVICE)
                self.map_stats.append(coco_eval[0].save_stats())
                val_losses = validate(self.valid_loader, self.model)
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
                self.lr_scheduler.step()

                print(f"Epoch #{epoch+1} train loss: {summed_losses:.3f}")
                print(f"Epoch #{epoch+1} validation loss: {val_loss:.3f}")
                print("_"*50)


                if (epoch+1) % SAVE_MODEL_EPOCH == 0: # save the model after every n epoch
                    save_mod_to=os.path.join(self.output_save_to, f"model{epoch+1}.pth")
                    torch.save(self.model.state_dict(), save_mod_to)
                    print('SAVING MODEL COMPLETE...\n')
                    print("_"*50)

                    

                if (epoch+1) % SAVE_PLOTS_EPOCH == 0: # save loss plots after every n epoch
                    train_ax.plot(self.train_loss_all, color = 'dodgerblue')
                    train_ax.set_xlabel('Epochs')
                    train_ax.set_ylabel('train loss')
                    valid_ax.plot(self.val_loss_all, color='peru')
                    valid_ax.set_xlabel('Epochs')
                    valid_ax.set_ylabel('validation loss')

                    save_fig_train = os.path.join(self.output_save_to, f"train_loss_{self.num_epochs}.png")
                    save_fig_val = os.path.join(self.output_save_to, f"val_loss_{self.num_epochs}.png")
                    figure_1.savefig(save_fig_train)
                    figure_2.savefig(save_fig_val)
                    
                    print('SAVING PLOTS COMPLETE...')
                    print("_"*50)

                    save_mod_to=os.path.join(self.output_save_to, f"model{epoch+1}.pth")
                    torch.save(self.model.state_dict(), save_mod_to)


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

                    save_losses_as = os.path.join(self.output_save_to, "losses_df.csv")
                    losses_df.to_csv(save_losses_as)
                    self.save_map_stats(self.map_stats)
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

                    save_fig_train = os.path.join(self.output_save_to, f"train_loss_{self.num_epochs}.png")
                    save_fig_val = os.path.join(self.output_save_to, f"val_loss_{self.num_epochs}.png")
                    figure_1.savefig(save_fig_train)
                    figure_2.savefig(save_fig_val)
                    print('SAVING FINAL PLOTS COMPLETE...')
                    print("_"*50)

                    save_losses_as = os.path.join(self.output_save_to, "losses_df.csv")
                    losses_df.to_csv(save_losses_as)
                    self.save_map_stats(self.map_stats)

                    print('SAVING LOSSES AS CSV COMPLETE...')
                    print("_"*50)

                    save_mod_to=os.path.join(self.output_save_to, f"model{epoch+1}.pth")
                    torch.save(self.model.state_dict(), save_mod_to)


                plt.close('all')
        from config import MARKINGS_DIR
        from cropping import Cropping_engine
        
        print("TRAIN_FOR: ", self.train_for)
        if self.train_for == 'dolphin':
            val_images_dir = os.path.join(VAL_DIR, "images")
            image_paths_dir = os.path.join(val_images_dir, "*")
            image_paths = glob.glob(image_paths_dir)

            mod_path = os.path.join(self.output_save_to, f"model{epoch+1}.pth")
            cropping_engine = Cropping_engine(
                self.backbone, 
                self.train_for, 
                MODEL_PATH=mod_path,
                IMAGES_DIR=val_images_dir, 
                MODEL=self.model
                )
            cropped_image = cropping_engine.crop_and_save()
        else:
            val_images_dir = os.path.join(MARKINGS_DIR, "val", "images")
            image_paths_dir = os.path.join(val_images_dir, "*")
            image_paths = glob.glob(image_paths_dir)

            mod_path = os.path.join(self.output_save_to, f"model{epoch+1}.pth")
            cropping_engine = Cropping_engine(
                self.backbone, 
                self.train_for, 
                MODEL_PATH=mod_path, 
                IMAGES_DIR=val_images_dir, 
                MODEL=self.model
                )
            cropped_image = cropping_engine.crop_and_save()

        print("_"*50)
        print(f"RUN COMPLETED. Outputs saved to: '{self.output_save_to}'")

        torch.cuda.empty_cache() # PyTorch to clear GPU cache



            

        

        

        
