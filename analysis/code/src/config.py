import torch
import os

ROOT = '/home/charlie/Documents/Uni/Exeter - Data Science/Dissertation/analysis'

BATCH_SIZE = 2 # increase / decrease according to GPU memory
RESIZE_TO = 1024
NUM_EPOCHS = 50 

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# training images and XML fuiles directory
TRAIN_DIR = os.path.join(ROOT, 'data/train')
TRAIN_ANNOT_DIR = os.path.join(ROOT, 'data/train_annotations')

# validation images and XML files directory
VALID_DIR = os.path.join(ROOT, 'data/test')
VALID_ANNOT_DIR = os.path.join(ROOT, 'data/test_annotations')


# classes: 0 index is reserved for background

CLASSES = ['background', 'impression', 'dolphin', 'fin_slice', 'amputation']

NUM_CLASSES = len(CLASSES)

COLOURS = {'impression': (255, 0, 0),'dolphin': (0, 0, 255), 'fin_slice':(0, 255, 0), 'amputation': (0, 255, 255)}

# whether to visualize images after creating the data loaders
VISUALIZE_TRANSFORMED_IMAGES = False

# location to save model and plots
OUT_DIR = os.path.join(ROOT, 'outputs')
SAVE_PLOTS_EPOCH = 2 # save loss plots after these many epochs
SAVE_MODEL_EPOCH = 2 # save model after these many epochs

NUM_WORKERS = 2



