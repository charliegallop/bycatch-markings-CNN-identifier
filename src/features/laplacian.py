## FROM: https://stackoverflow.com/questions/63838719/why-does-cv2-sobel-function-returns-a-black-and-white-image-instead-of-grayscale

import numpy as np
import scipy
import scipy.signal as sig
import os
import glob
import cv2 as cv
import matplotlib.pyplot as plt

# get all the image folder paths
ROOT_DIR = "/home/charlie/Documents/Uni/Exeter - Data Science/Dissertation/alternative_methods/Laplacian"
sets = ['train', 'val', 'test']

import cv2
import numpy as np
import skimage.exposure as exposure

for s in sets:

    images_dir = os.path.join(ROOT_DIR, "data", s, "images")
    image_paths = glob.glob(f"{images_dir}/*")
    for path in image_paths:
        image_name = os.path.basename(path)
        img = cv.imread(path, 0)
        
        blur = cv2.GaussianBlur(img,(3,3),0)

        # apply laplacian derivative
        laplacian = cv2.Laplacian(blur,cv2.CV_64F, ksize=5)
            
        # normalize to range 0 to 255
        laplacian_norm = cv2.convertScaleAbs(laplacian)

        # save results
        save_to = os.path.join(ROOT_DIR, s, "images", image_name)
        print("Saving to: ", save_to)
        cv2.imwrite(save_to, laplacian_norm)
