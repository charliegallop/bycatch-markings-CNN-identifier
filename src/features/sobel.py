## FROM: https://stackoverflow.com/questions/63838719/why-does-cv2-sobel-function-returns-a-black-and-white-image-instead-of-grayscale

import numpy as np
import scipy
import scipy.signal as sig
import os
import glob
import cv2 as cv
import matplotlib.pyplot as plt

# get all the image folder paths
ROOT_DIR = "/home/charlie/Documents/Uni/Exeter - Data Science/Dissertation/alternative_methods/Sobel"
sets = ['train', 'test', 'val']

import cv2
import numpy as np
import skimage.exposure as exposure

for s in sets:

    images_dir = os.path.join(ROOT_DIR, "data", s, "images")
    image_paths = glob.glob(f"{images_dir}/*")

    for path in image_paths:
        image_name = os.path.basename(path)
        img = cv.imread(path, 0)
        # convert to gray
        #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # blur
        blur = cv2.GaussianBlur(img,(3,3),0)

        # apply sobel x derivative
        sobelx = cv2.Sobel(blur,cv2.CV_64F,1,0,ksize=3)

        # apply sobel y derivative
        sobely = cv2.Sobel(blur,cv2.CV_64F,0,1,ksize=3)

        # convert to absolute scale for plotting
        abs_sobelx = cv2.convertScaleAbs(sobelx)
        abs_sobely = cv2.convertScaleAbs(sobely)

        # approximate the combined x and y gradient intensity
        grad = cv2.addWeighted( abs_sobelx, 0.5, abs_sobely, 0.5, 0)

        # save results
        save_to = os.path.join(ROOT_DIR, s, "images", image_name)
        cv2.imwrite(save_to, grad)
        print("Saved to: ", save_to)

