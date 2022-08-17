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
                
        # # normalize to range -1 to 1
        # sobelx_norm1a = exposure.rescale_intensity(sobelx, in_range='image', out_range=(-1,1))

        # # normalize to range -255 to 255 and clip negatives 
        # sobelx_norm1b = exposure.rescale_intensity(sobelx, in_range='image', out_range=(-255,255)).clip(0,255).astype(np.uint8)
                
        # normalize to range 0 to 255
        #sobel_norm8 = exposure.rescale_intensity(sobel_final, in_range='image', out_range=(0,255)).astype(np.uint8)

        # save results
        save_to = os.path.join(ROOT_DIR, s, "images", image_name)
        cv2.imwrite(save_to, grad)
        # cv2.imwrite('barn_sobel_norm8.jpg', sobelx_norm8)

        # # show results
        # cv2.imshow('sobelx_norm1a', sobelx_norm1a)  
        # cv2.imshow('sobelx_norm1b', sobelx_norm1b)  
        # cv2.imshow('sobelx_norm8', sobelx_norm8) 
        # print(sobelx_norm8.shape)
        # print(img.shape)

        # cv2.waitKey(0)
        # cv2.destroyAllWindows()