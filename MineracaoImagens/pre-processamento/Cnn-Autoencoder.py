# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 16:37:47 2021

@author: gabriel
"""
#DATASET
#!wget https://people.csail.mit.edu/torralba/code/spatialenvelope/spatial_envelope_256x256_static_8outdoorcategories.zip

#!unzip spatial_envelope_256x256_static_8outdoorcategories.zip

from os import listdir
from os.path import isfile, join

dataset = 'spatial_envelope_256x256_static_8outdoorcategories/'

imagens = [f for f in listdir(dataset) if isfile(join(dataset, f)) and '.jpg' in f]
imagens

#Conjunto de Treino e Teste

#!mkdir -p data/train
#!mkdir -p data/test

from sklearn.model_selection import train_test_split
from shutil import copyfile
import os

x_train ,x_test = train_test_split(imagens,test_size=0.3)

for img in x_train:
  img_class = img.split('_')[0]

  if not os.path.exists('data/train/'+img_class):
      os.makedirs('data/train/'+img_class)

  copyfile(dataset+'/'+img, 'data/train/'+img_class+'/'+img)

for img in x_test:
  img_class = img.split('_')[0]

  if not os.path.exists('data/test/'+img_class):
      os.makedirs('data/test/'+img_class)

  copyfile(dataset+'/'+img, 'data/test/'+img_class+'/'+img)
  

from keras.preprocessing.image import ImageDataGenerator

img_width = 256
img_height = 256

gen = ImageDataGenerator()
train_im = ImageDataGenerator(
               rescale=1./255,
               shear_range=0.2,
               horizontal_flip=False)
def train_images():
    train_generator = train_im.flow_from_directory (
            'data/train', 
             target_size=(img_width, img_height),
             color_mode='rgb',
             batch_size=len(x_train),
             shuffle = True,
             class_mode='categorical')
    x =  train_generator
    return x[0][0], x[0][1]

X_train, Y_train = train_images()


#CNN Autoencoder

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K


if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

input_img = Input(shape=input_shape) 

x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(4, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(4, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)


autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.summary()


autoencoder.fit(X_train, X_train, epochs=15, verbose=1)


autoencoder.save_weights('autoencoder.h5')

autoencoder.load_weights('autoencoder.h5')

#Analisando os resultados

import matplotlib.pyplot as plt
plt.imshow(X_train[0])


import matplotlib.pyplot as plt

prediction = autoencoder.predict(X_train)
x =prediction[0].reshape(256,256,3) # imagem reconstruida pelo autoencoder
plt.imshow(x)


encoder = Model(input_img, encoded)
img_train_features = encoder.predict(X_train)


img_train_features[0].shape

import numpy as np
np.array(list(img_train_features[0].reshape(1,4096)[0])).shape
    # test_codes = test_codes.reshape(test_codes.shape[0], test_codes.shape[1] * test_codes.shape[2] * test_codes.shape[3])


from keras.preprocessing.image import ImageDataGenerator

img_width = 256
img_height = 256

gen = ImageDataGenerator()
train_im = ImageDataGenerator(
               rescale=1./255,
               shear_range=0.2,
               horizontal_flip=False)
def test_images():
    train_generator = train_im.flow_from_directory (
            'data/test', 
             target_size=(img_width, img_height),
             color_mode='rgb',
             batch_size=len(x_test),
             shuffle = True,
             class_mode='categorical')
    x =  train_generator
    return x[0][0], x[0][1]

X_test, Y_test = test_images()

Y_test

import matplotlib.pyplot as plt

prediction = autoencoder.predict(X_test)
x =prediction[0].reshape(256,256,3)
plt.imshow(x)

plt.imshow(X_test[0])

img_test_features = encoder.predict(X_test)
list(img_test_features[0].reshape(1,4096)[0])


X_features_train = []
for v in img_train_features:
  X_features_train.append( list(v.reshape(1,4096)[0]) )

X_features_test = []
for v in img_test_features:
  X_features_test.append( list(v.reshape(1,4096)[0]) )
  

import numpy as np
np.array(X_features_train).shape


np.array(X_features_test).shape

import pandas as pd
pd.DataFrame(np.array(X_features_test))

#Busca por Similaridade


img_id_test = 178
plt.imshow(X_test[img_id_test])

img_id = 0
img_sel = -1
min_dist = 1000000

for v in X_features_train:
  dist = np.linalg.norm(np.array(X_features_test[img_id_test])-np.array(v)) # distancia euclidiana entre vetores de caracteristicas
  if dist < min_dist:
    img_sel = img_id
    min_dist = dist
  img_id += 1

print('img_sel=',img_sel,' min_dist=',min_dist)
plt.imshow(X_train[img_sel])


