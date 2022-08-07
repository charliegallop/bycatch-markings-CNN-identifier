from engine_TEST import engine
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-bb", "--backbone", help="Name of backbone")
parser.add_argument("-e", "--epochs", help="Number of epochs")
parser.add_argument("-as", "--annotation_set", help="Annotation set to train on")

args = parser.parse_args()

BACKBONE = str(args.backbone)
EPCOHS =str(args.epochs)
TRAIN_FOR = str(args.annotation_set)

bb_opts = ['mobilenet', 'resnet']
tf_opts = ['dolphin', 'markings']

BACKBONE = 'mobilenet'
EPOCHS = 50
TRAIN_FOR = 'dolphin'

run = False

print("Define the parameters of the model or type 'default' to use the default settings" )
user_bb = input(f"What BACKBONE would you like the model to have? {bb_opts} > ")
if user_bb == 'default':
    run = True
    pass
else:
    bb_valid = False
    e_valid = False
    tf_valid = False
    while run == False:
        while bb_valid == False:
            if user_bb not in bb_opts:
                user_bb = input(f"Not a valid response, choose from {bb_opts}: ")
            else:    
                BACKBONE = user_bb
                bb_valid = True
        while tf_valid == False:
            user_tf = input(f"What labels would you like the model to TRAIN_FOR? {tf_opts} > ")
            if user_tf not in tf_opts:
                user_tf = input("Not a valid response, choose from {tf_opts}: ")
            else:    
                TRAIN_FOR = user_tf
                tf_valid = True
        run = True
        # EPOCHS = int(input("How many EPOCHS would you like the model to train for? > " ))


if run:
    engine = engine(BACKBONE, TRAIN_FOR)
    engine.run()

