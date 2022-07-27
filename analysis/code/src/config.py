import torch
import os

ROOT = '/home/charlie/Documents/Uni/Exeter - Data Science/Dissertation/analysis'

BATCH_SIZE = 4 # increase / decrease according to GPU memory
RESIZE_TO = 512
NUM_EPOCHS = 100 

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# training images and XML fuiles directory
TRAIN_DIR = os.path.join(ROOT, 'data/train')

# validation images and XML files directory
VALID_DIR = os.path.join(ROOT, 'data/test')

# classes: 0 index is reserved for background

CLASSES = ['background', 'impression', 'dolphin', 'fin_slice', 'amputation']

NUM_CLASSES = 5

# whether to visualize images after creating the data loaders
VISUALIZE_TRANSFORMED_IMAGES = False

# location to save model and plots
OUT_DIR = os.path.join(ROOT, 'outputs')
SAVE_PLOTS_EPOCH = 2 # save loss plots after these many epochs
SAVE_MODEL_EPOCH = 2 # save model after these many epochs



