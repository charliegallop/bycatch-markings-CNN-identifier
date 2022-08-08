from config import BACKBONE, TRAIN_FOR, NUM_EPOCHS
from engine_TEST import engine

# import argparse

# DEBUG_MODE = False

# parser = argparse.ArgumentParser()
# parser.add_argument("-db", "--debug", help="Run in debug mode")

# args = parser.parse_args()

# DEBUG_MODE = str(args.debug)

# # parser.add_argument("-bb", "--backbone", help="Name of backbone")
# # parser.add_argument("-e", "--epochs", help="Number of epochs")
# # parser.add_argument("-as", "--annotation_set", help="Annotation set to train on")

# # args = parser.parse_args()

# # BACKBONE = str(args.backbone)
# # EPCOHS =str(args.epochs)
# # TRAIN_FOR = str(args.annotation_set)

# bb_opts = ['mobilenet', 'resnet']
# tf_opts = ['dolphin', 'markings']

# BACKBONE = 'resnet'
# EPOCHS = 150
# TRAIN_FOR = 'markings'
# DEBUG_MODE = False

# print(DEBUG_MODE)
# if DEBUG_MODE == False:

#     run = False

#     print("Define the parameters of the model or type 'default' to use the default settings" )
#     user_bb = input(f"What BACKBONE would you like the model to have? {bb_opts} > ")
#     if user_bb == 'default':
#         run = True
#         pass
#     else:
#         bb_valid = False
#         e_valid = False
#         tf_valid = False
#         while run == False:
#             while bb_valid == False:
#                 if user_bb not in bb_opts:
#                     user_bb = input(f"Not a valid response, choose from {bb_opts}: ")
#                 else:    
#                     BACKBONE = user_bb
#                     bb_valid = True
#             while tf_valid == False:
#                 user_tf = input(f"What labels would you like the model to TRAIN_FOR? {tf_opts} > ")
#                 if user_tf not in tf_opts:
#                     user_tf = input(f"Not a valid response, choose from {tf_opts}: ")
#                 else:    
#                     TRAIN_FOR = user_tf
#                     tf_valid = True
#             while e_valid == False:
#                 user_e = input(f"How many EPOCHS do you want? > ")
#                 EPOCH = user_e
#                 e_valid = True
#             run = True
#             # EPOCHS = int(input("How many EPOCHS would you like the model to train for? > " ))


#     if run:
#         engine = engine(BACKBONE, TRAIN_FOR, EPOCHS)
#         engine.run()

# else:
#     engine = engine(BACKBONE, TRAIN_FOR, EPOCHS)
#     engine.run()

model_engine = engine(BACKBONE, TRAIN_FOR, EPOCHS)
model_engine.run()


