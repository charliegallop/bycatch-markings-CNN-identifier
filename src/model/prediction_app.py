import tkinter as tk
from config import BACKBONE, TRAIN_FOR, NUM_EPOCHS, OUT_DIR, DATA_DIR
from tkinter import filedialog
import os

# Backbone options
backbones_crop = [
    "mobilenet",
    "mobilenet_320",
    "resnet"
]

backbones_pred = [
    "resnet",
    "mobilenet_320",
    "mobilenet"
]

fs = 16
tf = 14
# # Train_for options:
# train_fors = [
#     "dolphin",
#     "markings"
# ]


CROP_MODEL_PATH = None
PRED_MODEL_PATH = None
CROP_BACKBONE = backbones_crop[0]
PRED_BACKBONE = backbones_pred[0]
IMAGES_DIR = None

# Create object
root = tk.Tk()
  
root.title('Prediction App')
# Adjust size
root.geometry( "" )


#define variables
clicked_bb = tk.StringVar(root, value = backbones_crop[0])
clicked_pred = tk.StringVar(root, value = backbones_pred[0])
epochs = tk.IntVar(root, value = 50)

# Create Label
label = tk.Label( root , text = "Select BACKBONE for CROPPING:" )
label.pack()

# Create Dropdown menu
drop = tk.OptionMenu(root , clicked_bb , *backbones_crop )
drop.pack()

def browseFilescrop():
    global CROP_MODEL_PATH
    filename = filedialog.askopenfilename(initialdir = f"{OUT_DIR}/dolphin/",
                                          title = "Select a File",
                                          filetypes = (("Model PTH files",
                                                        "*.pth*"),
                                                       ("all files",
                                                        "*.*")))

    CROP_MODEL_PATH = filename
    path = filename.split('/')[-2] + "/" + filename.split('/')[-1]

    label1.configure(text = f"Selected model path: \n{path}")


      
button_explore = tk.Button(root,
                        text = "Select model to\n use for cropping",
                        command = browseFilescrop)
button_explore.pack()

 # Create Label
label1 = tk.Label( root , text = f"Selected model path: \n{CROP_MODEL_PATH}", width = 100, height = 4, fg = "blue", font=('Arimo', fs) )
label1.pack()




def callback_bb(*args):
    global CROP_BACKBONE
    CROP_BACKBONE = clicked_bb.get()
    label_predict.configure(text=f"These options will use a {CROP_BACKBONE} to crop for dolphins and a {PRED_BACKBONE} model to detect markings!\n|\nV")
    label_crop.configure(text=f"\nOR\n\nit will crop dolphin images using a {CROP_BACKBONE} model!\n|\nV")
clicked_bb.trace("w", callback_bb)

def callback_pred(*args):
    global PRED_BACKBONE
    PRED_BACKBONE = clicked_pred.get()
    label_predict.configure(text=f"These options will use a {CROP_BACKBONE} to crop for dolphins and a {PRED_BACKBONE} model to detect markings!\n|\nV")
    label_crop.configure(text=f"\nOR\n\nit will crop dolphin images using a {CROP_BACKBONE} model!\n|\nV")
clicked_pred.trace("w", callback_pred)

# Create Label
label = tk.Label( root , text = "\nSelect BACKBONE for PREDICTION:" )
label.pack()

# Create Dropdown menu
pred_drop = tk.OptionMenu(root , clicked_pred , *backbones_pred )
pred_drop.pack()

def browseFilespred():
    global PRED_MODEL_PATH
    filename = filedialog.askopenfilename(initialdir = f"{OUT_DIR}/markings/",
                                          title = "Select a File",
                                          filetypes = (("Model PTH files",
                                                        "*.pth*"),
                                                       ("all files",
                                                        "*.*")))

    PRED_MODEL_PATH = filename
    path = filename.split('/')[-2] + "/" + filename.split('/')[-1]

    label2.configure(text = f"Selected model path: \n{path}")


button_explore2 = tk.Button(root,
                        text = "Browse Files",
                        command = browseFilespred)
button_explore2.pack()

 # Create Label
label2 = tk.Label( root , text = f"Selected model path: \n{CROP_MODEL_PATH}", width = 100, height = 4, fg = "blue", font=('Arimo', fs) )
label2.pack()


def selectImgDir():
    global IMAGES_DIR
    dirname = filedialog.askdirectory(initialdir = f"{DATA_DIR}/",
                                          title = "Select images DIR\n for cropping/predict",
                                         )

    IMAGES_DIR = dirname

    label3.configure(text = f"Selected image directory: \n{IMAGES_DIR}")

button_explore3 = tk.Button(
    root, 
    text = "Select Image Directory",
    command = selectImgDir
    )
button_explore3.pack()

 # Create Label
label3 = tk.Label( 
    root, 
    text = f"Selected image directory: \n{IMAGES_DIR}", 
    width = 100, 
    height = 4, 
    fg = "blue", 
    font=('Arimo', fs)
    )
label3.pack()



def predict_model():
    from prediction_engine import predict_engine
    TRAIN_FOR = 'markings'
    pred_model_path = PRED_MODEL_PATH
    crop_model_path = CROP_MODEL_PATH
    model_engine = predict_engine(
        PRED_BACKBONE=PRED_BACKBONE,
        CROP_BACKBONE=CROP_BACKBONE, 
        TRAIN_FOR= TRAIN_FOR, 
        CROP_MODEL_PATH=crop_model_path,
        PRED_MODEL_PATH= pred_model_path,
        IMAGES_DIR=IMAGES_DIR
        )
    model_engine.run()

def crop_model():
    from cropping import Cropping_engine
    TRAIN_FOR = "dolphin"
    crop_engine = Cropping_engine(
        BACKBONE=CROP_BACKBONE,
        TRAIN_FOR=TRAIN_FOR,
        MODEL_PATH= CROP_MODEL_PATH,
        IMAGES_DIR= IMAGES_DIR
        )
    crop_engine.crop_and_save()


#Label
label_predict = tk.Label(
    text="Select what model to use for prediction and what to predict for...", 
    font=('Arimo', fs), 
    fg="#e36414")
label_predict.pack(side="top")


button = tk.Button(
    root, 
    text= "Make Predictions", 
    font = ('Ariel', tf), 
    command = predict_model
    )
button.pack()



#Label
label_crop = tk.Label(
    text="Select the model to be used for cropping...", 
    font=('Arimo', fs), 
    fg="#e36414"
    )
label_crop.pack(side="top")

button = tk.Button(
    root, 
    text= "Predict and crop\n(for dolphin)", 
    command = crop_model,
    font = ('Arimo', tf), 
    )
button.pack()

# Function for opening the
# file explorer window




# Execute tkinter
root.mainloop()

