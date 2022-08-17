import torch
import os

# Select which labels to run the model on:
# - "dolphins"
# - "markings"
# - "all"

class train_for:
    def __init__(self): 
        self.train_for = "dolphin"
    
    def __str__(self):
        return self.train_for

    def change(self, change):
        assert isinstance(change, str)
        self.train_for = change

    def value(self):
        return self.train_for

class backbone:
    def __init__(self): 
        self.backbone = "mobilenet"
    
    def __str__(self):
        return self.backbone
    
    def change(self, change):
        assert isinstance(change, str)
        self.backbone = change

    def value(self):
        return self.backbone

class num_epochs:
    def __init__(self): 
        self.num_epochs = 50
    
    def __int__(self):
        return self.num_epochs
    
    def change(self, change):
        assert isinstance(change, int)
        self.num_epochs = change

    def value(self):
        return self.num_epochs


    

# detection threshold, any predictions with confidence lower than this will
# be disregarded

THRESHOLD = 0.5

BATCH_SIZE = 64 # increase / decrease according to GPU memory
RESIZE_TO = 224

# choose which backbone to load for the faster r-cnn model
# Choices:
#   - 'mobilenet'
#   - 'resnet'

TRAIN_FOR = train_for()
BACKBONE = backbone()
NUM_EPOCHS = num_epochs()


#ROOT = os.path.join('C:\\Users\\cg639\\OneDrive - University of Exeter\\MASTER BACKUP\\dissy\\analysis')
ROOT = os.path.join(os.getcwd(), 'analysis')

DATA_DIR = os.path.join(ROOT, 'data')
MASTER_DIR = os.path.join(ROOT, 'master')
# location to save model and plots
OUT_DIR = os.path.join(ROOT, 'outputs')
# training images and XML files directory
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
# validation images and XML files directory
VAL_DIR = os.path.join(DATA_DIR, 'val')
# Directory of test images and XML files
TEST_DIR = os.path.join(DATA_DIR, 'test')
# Markings root dir for images and labels
MARKINGS_DIR = os.path.join(DATA_DIR, 'markings')
# Master dir for marking images that have been cropped from dolphin model
MASTER_MARKINGS_DIR = os.path.join(MARKINGS_DIR, 'master')

FINAL_PREDS_DIR = os.path.join(DATA_DIR, "evaluation", 'final_predictions')

EVAL_DIR = os.path.join(DATA_DIR, 'evaluation', 'eval')

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Directory of ground truth annotations to use for metrics eval
# Output predicted images dir
# TEST_IMG_PREDS_DIR = os.path.join(ROOT, 'data/test_set/test_prediction_images', BACKBONE)
# # Directory containing text files of predictions
# TEST_PREDS_DIR = os.path.join(ROOT, 'data/test_set/test_predictions', BACKBONE)


# classes: 0 index is reserved for background

CLASSES_MARKINGS = ['background', 'impression', 'dolphin', 'fin_slice', 'amputation', 'notch']
CLASSES_DOLPHIN = ['background', 'dolphin']


NUM_CLASSES_MARKINGS = len(CLASSES_MARKINGS)
NUM_CLASSES_DOLPHIN = len(CLASSES_DOLPHIN)

COLOURS = {'impression': (255, 0, 0),'dolphin': (0, 0, 255), 'fin_slice':(0, 255, 0), 'amputation': (0, 255, 255), 'notch': (255, 0, 255)}

# whether to visualize images after creating the data loaders
VISUALIZE_TRANSFORMED_IMAGES = False



SAVE_PLOTS_EPOCH = 2 # save loss plots after these many epochs
SAVE_MODEL_EPOCH = 10 # save model after these many epochs

NUM_WORKERS = 0





