import tkinter as tk
from engine import engine

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

BACKBONE = backbones[0]
TRAIN_FOR = train_fors[0]
NUM_EPOCHS = 50

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
    BACKBONE = clicked_bb.get()
    labelTest.configure(text=f"The chosen options will train a {BACKBONE} model on the {TRAIN_FOR} dataset!")

clicked_bb.trace("w", callback_bb)

# Create Label
label = tk.Label( root , text = "Select TRAIN_FOR:" )
label.pack()

# Create Dropdown menu
drop = tk.OptionMenu(root , clicked_tf , *train_fors )
drop.pack()


def callback_tf(*args):
    global TRAIN_FOR
    TRAIN_FOR = clicked_tf.get()
    labelTest.configure(text=f"The chosen options will train a {BACKBONE} model on the {TRAIN_FOR} dataset!")

clicked_tf.trace("w", callback_tf)

# Create Label
label = tk.Label( root , text = "Select NUM_EPOCHS:" )
label.pack()

epochs_num = tk.Entry(root)
epochs_num.pack()

def train_model():
    global NUM_EPOCHS, TRAIN_FOR, BACKBONE, engine
    NUM_EPOCHS = epochs_num.get()
    model_engine = engine(BACKBONE, TRAIN_FOR, int(NUM_EPOCHS))
    model_engine.run()

button = tk.Button(root, text= "Run Model", command = train_model)
button.pack()

#Label
labelTest = tk.Label(text="", font=('Helvetica', 12), fg='red')
labelTest.pack(side="top")

# Execute tkinter
root.mainloop()

