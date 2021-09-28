# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 15:48:33 2021

@author: gabriel
"""

from skimage import io
from PIL import Image 
from google.colab.patches import cv2_imshow # for image display
#!pip install -U opencv-python
from random import randrange
import matplotlib.pyplot as plt
import numpy as np
import cv2


figsize = (10, 10)

#Lendo imagem #1

# rgb_l = cv2.cvtColor(cv2.imread("left.jpg"), cv2.COLOR_BGR2RGB)
# gray_l = cv2.cvtColor(rgb_l, cv2.COLOR_RGB2GRAY)
# rgb_r = cv2.cvtColor(cv2.imread("right.jpg"), cv2.COLOR_BGR2RGB)
# gray_r = cv2.cvtColor(rgb_r, cv2.COLOR_RGB2GRAY)

rgb_l = io.imread('https://raw.githubusercontent.com/YoniChechik/AI_is_Math/master/c_08_features/left.jpg') 
rgb_l = cv2.cvtColor(rgb_l, cv2.COLOR_BGR2RGB)
cv2_imshow(cv2.cvtColor(rgb_l, cv2.COLOR_BGR2RGB))

#Lendo imagem #2

rgb_r = io.imread('https://raw.githubusercontent.com/YoniChechik/AI_is_Math/master/c_08_features/right.jpg') 
rgb_r = cv2.cvtColor(rgb_r, cv2.COLOR_BGR2RGB)
cv2_imshow(cv2.cvtColor(rgb_r, cv2.COLOR_BGR2RGB))

#Extração de Descritores SIFT

feature_extractor = cv2.SIFT_create()

# find the keypoints and descriptors with chosen feature_extractor
kp_l, desc_l = feature_extractor.detectAndCompute(rgb_l, None)
kp_r, desc_r = feature_extractor.detectAndCompute(rgb_r, None)

test_l = cv2.drawKeypoints(rgb_l, kp_l, None, color = (0, 255, 0))
test_r = cv2.drawKeypoints(rgb_r, kp_r, None, color = (0, 255, 0))

plt.figure(figsize=(20,20))
plt.imshow(test_l)
plt.title("keypoints")
plt.show()

plt.figure(figsize=(20,20))
plt.imshow(test_r)
plt.title("keypoints")
plt.show()


#Verificando os descritores...

for i in range(0,len(kp_l)):
  keypoint = kp_l[i]
  print('keypoint=',i,keypoint.size, keypoint.angle)
  print('keypoint descriptor=',desc_l[i])
  
  break

#Correspondência entre duas imagens usando descritores SIFT
#Usamos apenas os 30 melhores matches

num_match = 30

bf = cv2.BFMatcher()
matches = bf.knnMatch(desc_l, desc_r, k=2)

good_match = []
for m in matches:
    if m[0].distance/m[1].distance < 0.5:
        good_match.append(m)
good_match_arr = np.asarray(good_match)

im_matches = cv2.drawMatchesKnn(rgb_l, kp_l, rgb_r, kp_r, good_match[0:num_match], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

plt.figure(figsize=(30, 30))
plt.imshow(im_matches)
plt.title("keypoints matches")
plt.show()
