import torch
import os

ROOT = '/home/charlie/Documents/Uni/Exeter - Data Science/Dissertation/analysis'

BATCH_SIZE = 4 # increase / decrease according to GPU memory
RESIZE_TO = 2048
NUM_EPOCHS = 50 

# choose which backbone to load for the faster r-cnn model
# Choices:
#   - 'mobilenet'
#   - 'resnet'
BACKBONE = 'resnet'

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# training images and XML fuiles directory
TRAIN_DIR = os.path.join(ROOT, 'data/NEW/train')
TRAIN_ANNOT_DIR = os.path.join(ROOT, 'data/NEW/train_annotations')

# validation images and XML files directory
VALID_DIR = os.path.join(ROOT, 'data/NEW/test')
VALID_ANNOT_DIR = os.path.join(ROOT, 'data/NEW/test_annotations')


# classes: 0 index is reserved for background

CLASSES = ['background', 'impression', 'dolphin', 'fin_slice', 'amputation', 'notch']

NUM_CLASSES = len(CLASSES)

COLOURS = {'impression': (255, 0, 0),'dolphin': (0, 0, 255), 'fin_slice':(0, 255, 0), 'amputation': (0, 255, 255), 'notch': (255, 0, 255)}

# whether to visualize images after creating the data loaders
VISUALIZE_TRANSFORMED_IMAGES = False



# location to save model and plots
OUT_DIR = os.path.join(ROOT, 'outputs', BACKBONE)
SAVE_PLOTS_EPOCH = 2 # save loss plots after these many epochs
SAVE_MODEL_EPOCH = 2 # save model after these many epochs

NUM_WORKERS = 2




