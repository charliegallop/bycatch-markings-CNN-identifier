import os
import glob
import numpy as np
import PIL.Image


from skimage import data, transform
from sklearn.feature_extraction.image import PatchExtractor



ROOT_DIR = "/home/charlie/Documents/Uni/Exeter - Data Science/Dissertation/alternative_methods/HOG"

positive_dir = os.path.join(ROOT_DIR, "data", "positive")
negative_dir = os.path.join(ROOT_DIR, "data", "negative")
positive_paths = glob.glob(f"{positive_dir}/*")
negative_paths = glob.glob(f"{negative_dir}/*")


positive_images = []

for im in positive_paths:
    img = PIL.Image.open(im)
    img = np.array(img)
    positive_images.append(img)

negative_images = []

for im in negative_paths:
    img = PIL.Image.open(im)
    img = np.array(img)
    negative_images.append(img)



# Extract HOG features
from skimage import feature
from itertools import chain

X_train = np.array([feature.hog(im, orientations = 9, multichannel=True) for im in chain(positive_images, negative_images)])

y_train = np.zeros(X_train.shape[0])
y_train[:positive_images.shape[0]] = 1