#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colors
from skimage.color import rgb2gray, rgb2hsv, hsv2rgb
from skimage.io import imread, imshow
from sklearn.cluster import KMeans

#%%
test = imread('../data/test1.png')
plt.figure(num=None, figsize=(8,6), dpi = 80)
imshow(test)
# %%
def image_to_pandas(image):
    df = pd.DataFrame([image[:,:,0].flatten(), image[:,:,1].flatten(), image[:,:,2].flatten(),
    
    ])