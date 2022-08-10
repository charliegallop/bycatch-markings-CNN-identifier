import tkinter as tk
from engine import engine
from config import BACKBONE, TRAIN_FOR, NUM_EPOCHS

# Backbone options
backbones = [
    "mobilenet",
    "resnet"
]

# Train_for options:
train_fors = [
    "dolphin",
    "markings"
]


MODEL_PATH = None

# Create object
root = tk.Tk()
  
# Adjust size
root.geometry( "" )


#define variables
clicked_bb = tk.StringVar(root, value = backbones[0])
clicked_tf = tk.StringVar(root, value = train_fors[0])
epochs = tk.IntVar(root, value = 50)

# Create Label
label = tk.Label( root , text = "Select BACKBONE:" )
label.pack()

# Create Dropdown menu
drop = tk.OptionMenu(root , clicked_bb , *backbones )
drop.pack()

def callback_bb(*args):
    global BACKBONE
    BACKBONE.change(clicked_bb.get())
    labelTest.configure(text=f"The chosen options will train a {BACKBONE.value()} model on the {TRAIN_FOR.value()} dataset!")

clicked_bb.trace("w", callback_bb)

# Create Label
label = tk.Label( root , text = "Select TRAIN_FOR:" )
label.pack()

# Create Dropdown menu
drop = tk.OptionMenu(root , clicked_tf , *train_fors )
drop.pack()


def callback_tf(*args):
    global TRAIN_FOR
    TRAIN_FOR.change(clicked_tf.get())
    labelTest.configure(text=f"The chosen options will train a {BACKBONE.value()} model on the {TRAIN_FOR.value()} dataset!")

clicked_tf.trace("w", callback_tf)

# Create Label
label = tk.Label( root , text = "Select NUM_EPOCHS:" )
label.pack()

epochs_num = tk.Entry(root)
epochs_num.pack()

def check_model_entry():
    if len(model_path.get()) == 0:
        path = None
    else:
        path = model_path.get()
    return path


# Create Label
label = tk.Label( root , text = "(Optional) Enter the path to a model state to train from: " )
label.pack()

model_path = tk.Entry(root)
model_path.pack()

def train_model():
    global NUM_EPOCHS, TRAIN_FOR, BACKBONE, MODEL_PATH, engine
    NUM_EPOCHS.change(int(epochs_num.get()))
    MODEL_PATH = check_model_entry()
    model_engine = engine(BACKBONE.value(), TRAIN_FOR.value(), int(NUM_EPOCHS.value()), MODEL_PATH)
    model_engine.run()

def infer_model():
    from inference import Inference_engine
    global NUM_EPOCHS, TRAIN_FOR, BACKBONE, MODEL_PATH, engine
    MODEL_PATH = check_model_entry()
    infer_engine = Inference_engine(BACKBONE.value(), TRAIN_FOR.value(), MODEL_PATH)
    infer_engine.infer()


button = tk.Button(root, text= "Run Model", command = train_model)
button.pack()

#Label
labelTest = tk.Label(text="", font=('Helvetica', 12), fg='red')
labelTest.pack(side="top")

button = tk.Button(root, text= "Infer Model", command = infer_model)
button.pack()

#Label
labelTest = tk.Label(text="", font=('Helvetica', 12), fg='red')
labelTest.pack(side="top")

# Execute tkinter
root.mainloop()

