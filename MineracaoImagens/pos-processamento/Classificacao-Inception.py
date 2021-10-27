# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 14:07:30 2021

@author: gabriel
"""

#Baixando o dataset

#!wget https://people.csail.mit.edu/torralba/code/spatialenvelope/spatial_envelope_256x256_static_8outdoorcategories.zip

#!unzip spatial_envelope_256x256_static_8outdoorcategories.zip


#!rm spatial_envelope_256x256_static_8outdoorcategories/Thumbs.db 


#Instalando o Ktrain

#!pip install ktrain

"""
Pro Colab

%reload_ext autoreload
%autoreload 2
%matplotlib inline

"""
import os

os.environ['DISABLE_V2_BEHAVIOR'] = '1'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"]="0" 
import ktrain
from ktrain import vision as vis

from os import listdir
from os.path import isfile, join

dataset = 'spatial_envelope_256x256_static_8outdoorcategories/'

imagens = [f for f in listdir(dataset) if isfile(join(dataset, f)) and '.jpg' in f]
imagens

#Gerando treino e teste

"""
!mkdir -p data/train
!mkdir -p data/test
"""

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
  

(train_data, val_data, preproc) = vis.images_from_folder('data/')

#Treinando o modelo

vis.print_image_classifiers()


model = vis.image_classifier('pretrained_inception', train_data, val_data)


# wrap model and data in Learner object
learner = ktrain.get_learner(model=model, train_data=train_data, val_data=val_data, 
                             workers=8, use_multiprocessing=False, batch_size=64)

learner.fit_onecycle(1e-4, 5)

#Explorando o Modelo


predictor = ktrain.get_predictor(learner.model, preproc)

import random
random.shuffle(val_data.filenames)
val_data.filenames


def show_prediction(fname):
    
    predicted = predictor.predict_filename(fname)[0]
    vis.show_image(fname)
    print('predicted:%s | actual: %s' % (predicted, fname))
    
    
    
show_prediction( 'data/test/street/street_boston373.jpg')



#!pip install git+https://github.com/amaiya/eli5@tfkeras_0_10_1


predictor.explain( 'data/test/forest/forest_text105.jpg')

