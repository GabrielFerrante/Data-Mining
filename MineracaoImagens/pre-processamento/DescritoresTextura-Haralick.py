# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 15:46:21 2021

@author: gabriel
"""

import cv2
import numpy as np
import os
import glob
import pandas as pd
#from google.colab.patches import cv2_imshow # for image display
from skimage import io
from PIL import Image 
import matplotlib.pylab as plt

import mahotas as mt

image = io.imread('http://farm3.static.flickr.com/2225/2062534589_7e473b108c.jpg') 
cv2_imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


print('Altura: ',image.shape[0])
print('Largura: ',image.shape[1])
print('#Canais: ', image.shape[2])


#Extraindo Descritores via Textura de Haralick

def extract_features(image):
        # calculate haralick texture features for 4 types of adjacency
        textures = mt.features.haralick(image)
        ht_mean = textures.mean(axis=0)
        return ht_mean
    

extract_features(image)

#Pegando o dataset

links = ['http://static.flickr.com/9/13803695_ea9f8dfc16.jpg',
'http://farm1.static.flickr.com/43/79256044_1e8945eb82.jpg',
'http://farm4.static.flickr.com/3058/2925112649_2a283607d8.jpg',
'http://farm1.static.flickr.com/206/507474775_da3a48d69b.jpg',
'http://farm3.static.flickr.com/2242/1695888408_940e8dbefc.jpg',
'http://static.flickr.com/1258/910812670_acacab421e.jpg',
'http://farm1.static.flickr.com/176/379081795_c3fa394f6a.jpg',
'http://farm1.static.flickr.com/212/485394549_011a7549e8.jpg',
'http://static.flickr.com/2094/2220184225_b3d09f4738.jpg',
'http://farm1.static.flickr.com/86/227139104_506b4c6b4d.jpg',
'http://farm1.static.flickr.com/7/11202867_c3833d758e.jpg',
'http://farm4.static.flickr.com/3154/3041795338_f9eb506368.jpg',
'http://static.flickr.com/2052/1547494237_4135d64fb1.jpg',
'http://farm1.static.flickr.com/121/290007835_015d1eea4f.jpg',
'http://farm1.static.flickr.com/201/476746478_49dac3bf4b.jpg',
'http://farm3.static.flickr.com/2283/2499880901_a5df4ef867.jpg',
'http://www.natures-desktop.com/images/wallpapers/800x600/landscapes/stone-wall-valley.jpg',
'http://static.flickr.com/145/370604481_3f4ebfde15.jpg',
'http://www.cottagecatalog.com/images/cottage/stonewall.jpg',
'http://farm4.static.flickr.com/3087/2580797907_e28859be95.jpg',
'http://farm3.static.flickr.com/2287/2944334392_6b21776aca.jpg',
'http://farm2.static.flickr.com/1118/1414880730_f40acca42d.jpg',
'http://static.flickr.com/27/45885641_3f421eca11.jpg',
'http://static.flickr.com/3142/3066137591_12ae1bea0e.jpg',
'http://www.chivalry.com/songfest/photos/mary-adi-stonewall.jpg',
'http://farm2.static.flickr.com/1421/1218755515_81f4a603b3.jpg',
'http://www.earthproducts.net/images/stonewall.jpg',
'http://static.flickr.com/2329/2050591096_1c4d78fa60.jpg',
'http://farm4.static.flickr.com/3026/2618945681_fd7519c121.jpg',
'http://farm4.static.flickr.com/3083/3130777027_7cb1d6a6de.jpg',
'http://static.flickr.com/157/329508488_ffe1af9ab0.jpg',
'http://www.trueelena.org/arcr/wallpapers/img/stone_wall_1-800x480.jpg',
'http://farm4.static.flickr.com/3023/2919866515_990401b96c.jpg',
'http://www.ravelgrane.com/pix/places/200310_Lechworth/087_stone_wall.jpg',
'http://farm1.static.flickr.com/176/441616466_54ef1527ee.jpg',
'http://farm1.static.flickr.com/202/510010947_0b8563a369.jpg',
'http://farm4.static.flickr.com/3219/2939899060_af5f0c4fba.jpg',
'http://lazykitty.net/irelandBN/original/08.%20Close%20up%20of%20a%20stone%20wall.jpg',
'http://farm4.static.flickr.com/3194/2581434695_cd8d46e8d2.jpg',
'http://ts11.brightqube.com/images/l/000/657/000657159.jpg',
'http://farm1.static.flickr.com/41/77752066_94fd0f8838.jpg',
'http://farm2.static.flickr.com/1228/758266377_e8afad1c9e.jpg',
'http://farm4.static.flickr.com/3113/2760382905_d13b4dbaf3.jpg',
'http://static.flickr.com/1156/1466297536_2e805003ca.jpg',
'http://farm3.static.flickr.com/2177/2252911907_70b98cdacc.jpg',
'http://farm4.static.flickr.com/3022/2633843071_e41c29bc49.jpg',
'http://static.flickr.com/2242/2413140109_9f7218f2a7.jpg',
'http://farm1.static.flickr.com/4/9262278_338e7f2bc8.jpg',
'http://farm1.static.flickr.com/30/53328264_eab6256e01.jpg',
'http://farm1.static.flickr.com/28/100934807_02e4d7a8a5.jpg',
'http://farm4.static.flickr.com/3027/3138359327_974d1959e9.jpg',
'http://farm4.static.flickr.com/3264/2634647776_7e856c3ef0.jpg',
'http://farm3.static.flickr.com/2306/2372746612_4d4207bfdf.jpg',
'http://farm1.static.flickr.com/202/513244698_429e5257ce.jpg',
'http://farm2.static.flickr.com/1086/1192125566_e1c4cfbf88.jpg']

dataset = []
for img_url in links:
  try:
    image = io.imread(img_url) 
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    dataset.append(image)
  except:
    print('Warning: ',img_url)
    

plt.subplots(figsize=(30,50)) 
columns = 5
for i, image in enumerate(dataset):
    plt.subplot(len(dataset) / columns + 1, columns, i + 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    

#Pré-processando via Descritores de Texturas

X = []
for image in dataset:
  textures = extract_features(image)
  X.append(textures)
  

X

#Agrupamento com K-Means

from sklearn.cluster import KMeans
import numpy as np

kmeans = KMeans(n_clusters=15).fit(X)
kmeans.labels_


# visualizando o cluster k
cluster_k = 12
for image_id, cluster in enumerate(kmeans.labels_):
    if cluster == cluster_k:
      print(image_id)
      cv2_imshow(dataset[image_id])