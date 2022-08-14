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
sets = ['val']

import cv2
import numpy as np
import skimage.exposure as exposure

for s in sets:

    images_dir = os.path.join(ROOT_DIR, "data", s)
    val_dir = images_dir = os.path.join(ROOT_DIR, "data", s)
    image_paths = glob.glob(f"{images_dir}/*")

    for path in image_paths:
        image_name = os.path.basename(path)
        img = cv.imread(path, 0)
        
        blur = cv2.GaussianBlur(img,(3,3),0)

        # apply laplacian derivative
        laplacian = cv2.Laplacian(blur,cv2.CV_64F, ksize=5)
            
        # normalize to range 0 to 255
        laplacian_norm = exposure.rescale_intensity(laplacian, in_range='image', out_range=(0,255)).astype(np.uint8)

        # save results
        save_to = os.path.join(ROOT_DIR, s, image_name)
        cv2.imwrite(save_to, laplacian_norm)
        # cv2.imwrite('barn_sobel_norm8.jpg', sobelx_norm8)

        # # show results
        # cv2.imshow('sobelx_norm1a', sobelx_norm1a)  
        # cv2.imshow('sobelx_norm1b', sobelx_norm1b)  
        # cv2.imshow('sobelx_norm8', sobelx_norm8) 
        # print(sobelx_norm8.shape)
        # print(img.shape)

        # cv2.waitKey(0)
        # cv2.destroyAllWindows()