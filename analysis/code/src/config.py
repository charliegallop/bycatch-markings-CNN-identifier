import torch
import os

# Select which labels to run the model on:
# - "dolphins"
# - "markings"
# - "all"

TRAIN_FOR = "dolphin"

# detection threshold, any predictions with confidence lower than this will
# be disregarded

THRESHOLD = 0.5

BATCH_SIZE = 3 # increase / decrease according to GPU memory
RESIZE_TO = 256
NUM_EPOCHS = 4

# choose which backbone to load for the faster r-cnn model
# Choices:
#   - 'mobilenet'
#   - 'resnet'
BACKBONE = 'mobilenet'

ROOT = '/home/charlie/Documents/Uni/Exeter - Data Science/Dissertation/analysis'

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# training images and XML files directory
TRAIN_DIR = os.path.join(ROOT, 'data/train_set/train')
TRAIN_ANNOT_DIR = os.path.join(ROOT, 'data/train_set/train_annotations')

# validation images and XML files directory
VALID_DIR = os.path.join(ROOT, 'data/train_set')
VALID_ANNOT_DIR = os.path.join(ROOT, 'data/train_set/valid_annotations')
VALID_PRED_DIR = os.path.join(ROOT, 'data/train_set/valid_predictions')

# TEST SET
# Directory of images
TEST_DIR = os.path.join(ROOT, 'data/test_set/test_prediction_images')
# Directory of ground truth annotations to use for metrics eval
TEST_ANNOT_DIR = ""
# Output predicted images dir
TEST_IMG_PREDS_DIR = os.path.join(ROOT, 'data/test_set/test_prediction_images', BACKBONE)
# Directory containing text files of predictions
TEST_PREDS_DIR = os.path.join(ROOT, 'data/test_set/test_predictions', BACKBONE)


# classes: 0 index is reserved for background

CLASSES_MARKINGS = ['background', 'impression', 'dolphin', 'fin_slice', 'amputation', 'notch']
CLASSES_DOLPHIN = ['background', 'dolphin']


NUM_CLASSES_MARKINGS = len(CLASSES_MARKINGS)
NUM_CLASSES_DOLPHIN = len(CLASSES_DOLPHIN)

COLOURS = {'impression': (255, 0, 0),'dolphin': (0, 0, 255), 'fin_slice':(0, 255, 0), 'amputation': (0, 255, 255), 'notch': (255, 0, 255)}

# whether to visualize images after creating the data loaders
VISUALIZE_TRANSFORMED_IMAGES = False



# location to save model and plots
OUT_DIR = os.path.join(ROOT, 'outputs')
SAVE_PLOTS_EPOCH = 2 # save loss plots after these many epochs
SAVE_MODEL_EPOCH = 2 # save model after these many epochs

NUM_WORKERS = 2





