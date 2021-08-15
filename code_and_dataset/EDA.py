import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import sklearn.preprocessing as pre
import keyboard as kb
import pandas as pd


# 影像大小：
wide = []
length = []
B = []
G = []
R = []

current_dir = os.getcwd()
os.chdir('tumor/yes')
image_list = os.listdir()

for pic in image_list:
    img = cv2.imread(pic)
    wide.append(img.shape[0])
    length.append(img.shape[1])
    B.append(img[:,:,0])
    G.append(img[:,:,1])
    R.append(img[:, :, 2])


os.chdir(current_dir)
os.chdir('tumor/no')
image_list = os.listdir()

for pic in image_list:
    img = cv2.imread(pic)
    wide.append(img.shape[0])
    length.append(img.shape[1])
    B.append(img[:,:,0])
    G.append(img[:,:,1])
    R.append(img[:, :, 2])
'''
img = cv2.imread(image_list[89])
img_resize = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)

cv2.imshow('My Image', img_resize)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

