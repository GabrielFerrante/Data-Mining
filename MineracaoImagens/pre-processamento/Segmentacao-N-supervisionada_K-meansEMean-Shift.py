# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 15:33:16 2021

@author: gabriel
"""

import numpy as np
import pandas as pd
import cv2 as cv 
#from google.colab.patches import cv2_imshow # for image display
from skimage import io
from PIL import Image 
import matplotlib.pylab as plt


image = io.imread('http://farm3.static.flickr.com/2026/1544063360_7ec3529266.jpg') 
cv2_imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))

print('Altura: ',image.shape[0])
print('Largura: ',image.shape[1])
print('#Canais: ', image.shape[2])


#Normalizando a imagem

from skimage.io import imread
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from scipy import ndimage

img = image / 255
plt.imshow(img)
plt.show()

#Gerando Dataset

image_2D = img.reshape(img.shape[0]*img.shape[1], img.shape[2])
image_2D

#Segmentação com K-Means

from sklearn.cluster import KMeans

num_segmentos = 5

kmeans = KMeans(n_clusters=num_segmentos, random_state=0).fit(image_2D)
clustered = kmeans.cluster_centers_[kmeans.labels_]

clustered_3D = clustered.reshape(img.shape[0], img.shape[1], img.shape[2])
plt.imshow(clustered_3D)
plt.title('Imagem segmentada k='+str(num_segmentos))
plt.show()

rgb_centers = kmeans.cluster_centers_*1
rgb_centers

rgb_centers[0] = [1,1,1]
rgb_centers[1] = [1,1,1]
rgb_centers[2] = [1,1,1]
rgb_centers[4] = [1,1,1]


rgb_centers
clustered = rgb_centers[kmeans.labels_]

clustered_3D = clustered.reshape(img.shape[0], img.shape[1], img.shape[2])
plt.imshow(clustered_3D)
plt.title('Imagem segmentada')
plt.show()


#Segmentação usando Mean-Shift

import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets import make_blobs


ms = MeanShift(bandwidth=0.1,bin_seeding=True)
ms.fit(image_2D)

num_segmentos = len( np.unique(ms.labels_))

clustered = ms.cluster_centers_[ms.labels_]

clustered_3D = clustered.reshape(img.shape[0], img.shape[1], img.shape[2])
plt.imshow(clustered_3D)
plt.title('Imagem segmentada com MS k='+str(num_segmentos))
plt.show()

rgb_centers = ms.cluster_centers_*1
rgb_centers[0] = [1,1,1]
rgb_centers[4] = [1,1,1]
rgb_centers[5] = [1,1,1]
rgb_centers[6] = [1,1,1]
rgb_centers[7] = [1,1,1]
rgb_centers[8] = [1,1,1]
rgb_centers[9] = [1,1,1]

clustered = rgb_centers[ms.labels_]

clustered_3D = clustered.reshape(img.shape[0], img.shape[1], img.shape[2])
plt.imshow(clustered_3D)
plt.title('Imagem segmentada')
plt.show()

rgb_centers