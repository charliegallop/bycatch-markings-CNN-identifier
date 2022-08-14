'''
USAGE:
python hog_image_recognition.py --path dolphin
'''
import os
import cv2
import argparse
from sklearn.svm import LinearSVC
from skimage import feature
import glob
import numpy as np

# # construct the argument parser and parser the arguments
# parser = argparse.ArgumentParser()
# parser.add_argument('-p', '--path', help='what folder to use for HOG description', 
#                     choices=['dolphin'])
# args = vars(parser.parse_args())

X_train = []
labels = []
# get all the image folder paths
ROOT_DIR = "/home/charlie/Documents/Uni/Exeter - Data Science/Dissertation/alternative_methods/HOG"
images_dir = os.path.join(ROOT_DIR, "data", "positive")
test_dir = images_dir = os.path.join(ROOT_DIR, "data", "test")
image_paths = glob.glob(f"{images_dir}/*")

for image_path in image_paths:
    image = cv2.imread(image_path)
    # get the HOG descriptor for the image
    hog_desc = feature.hog(image, orientations=9, pixels_per_cell=(8, 8),
        cells_per_block=(2, 2), transform_sqrt=True, block_norm='L2-Hys', channel_axis= -1)

    # update the data and labels
    X_train.append(hog_desc)
    labels.append(image_path)

y_train = np.zeros(X_train.shape[0])
y_train[:image_paths.shape[0]] = 1
