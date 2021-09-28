# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 16:12:20 2021

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

#Carregando dataset de imagens

#This dataset contains 8 outdoor scene categories: 
#coast, mountain, forest, open country, street, inside city, tall buildings and highways.
#There are 2600 color images, 256x256 pixels

#!wget https://people.csail.mit.edu/torralba/code/spatialenvelope/spatial_envelope_256x256_static_8outdoorcategories.zip

#!unzip spatial_envelope_256x256_static_8outdoorcategories.zip

#Extraindo descritores SIFT de cada imagem

from os import listdir
from os.path import isfile, join

dataset = 'spatial_envelope_256x256_static_8outdoorcategories/'

imagens = [f for f in listdir(dataset) if isfile(join(dataset, f))]
imagens

from tqdm.notebook import tqdm



feature_extractor = cv2.SIFT_create()

L_imagem = []
L_classe = []
L_sift = []

for imagem in tqdm(imagens):
  if '.jpg' in imagem:
    imagem_data = io.imread(dataset+'/'+imagem) 
    imagem_data = cv2.cvtColor(imagem_data, cv2.COLOR_BGR2RGB)
    _, sift_vectors = feature_extractor.detectAndCompute(imagem_data, None)
    L_imagem.append(imagem)
    L_classe.append(imagem.split("_")[0])
    L_sift.append(sift_vectors)


import pandas as pd
df_data = pd.DataFrame()
df_data['imagem'] = L_imagem
df_data['sift'] = L_sift
df_data['classe'] = L_classe
df_data


for index,row in df_data.iterrows():
  print(row['imagem'])
  for sift in row['sift']:
    print(sift)
    print('----')

  break



df_data.classe.hist(figsize=(10,5),bins=8)

df_data = df_data.sample(frac=1) # shuffle
df_data


#Agrupando descritores SIFT para gerar as Visual Words

#Gerando dataset de descritores SIFT

X_sift = []
for index,row in df_data.iterrows():
  for sift_vector in row['sift']:
    X_sift.append(sift_vector)
    

X_sift = np.array(X_sift)
X_sift.shape


X_sift



'''
Agrupamento de descritores
Em cenários reais, utilizar um grande número de clusters 
(acima de 10 mil)
Além disso, o K-means deve ser configurado para mais inicializações 
e mais iterações.
'''

from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
import numpy as np
from sklearn.model_selection import train_test_split

num_visual_words = 100
kmeans = MiniBatchKMeans(n_clusters=num_visual_words, n_init=1, max_iter=3, init='random', verbose=1)

kmeans.fit(X_sift)


kmeans.labels_

df_visual_words = pd.DataFrame(X_sift)
df_visual_words


df_visual_words['visual_word'] = kmeans.labels_
df_visual_words

df_visual_words.visual_word.hist()


df_visual_words[df_visual_words.visual_word==12]

#Calculando histogramas das visual words

L_histogram = []
for index,row in tqdm(df_data.iterrows(),total=len(df_data)):

  histogram = kmeans.predict(np.array(row['sift']))
  histogram = np.histogram(histogram,num_visual_words)[0]
  L_histogram.append(histogram)
  

df_data['visual_words_histogram'] = L_histogram
df_data

'''
Treinando classificador de imagens
Idealmente, usar o mesmo conjunto de treino que foi 
definido pra gerar as visual words!
'''

X = []
Y = []
for index,row in df_data.iterrows():
  X.append(row['visual_words_histogram'])
  Y.append(row['classe'])

X = np.array(X)
X.shape


X

Y

X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.20, random_state=42)

from sklearn.naive_bayes import GaussianNB

clf = GaussianNB().fit(X_train, y_train)
clf.score(X_test, y_test)


from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)
clf.score(X_test, y_test)


from sklearn.svm import SVC

clf = SVC().fit(X_train, y_train)
clf.score(X_test, y_test)

'''
Vamos comparar com um classificador simples baseado 
em histograma de cores
'''
#Extraindo histograma de cores

color = ('b','g','r')

L_histo_colors = []

for index,row in df_data.iterrows():
  imagem_data = io.imread(dataset+'/'+row['imagem']) 
  imagem_data = cv2.cvtColor(imagem_data, cv2.COLOR_BGR2RGB)

  hist_r = []
  hist_g = []
  hist_b = []
  for i,col in enumerate(color):
    histr = cv2.calcHist([imagem_data],[i],None,[256],[0,256])
    if col == 'r': hist_r = histr
    if col == 'g': hist_g = histr
    if col == 'b': hist_b = histr
  hist = np.concatenate((hist_r,hist_g,hist_b)).T[0]
  L_histo_colors.append(hist)
  

df_data['histogram_colors'] = L_histo_colors
df_data


X = []
Y = []
for index,row in df_data.iterrows():
  X.append(row['histogram_colors'])
  Y.append(row['classe'])

X = np.array(X)
X.shape


X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.20, random_state=42)

from sklearn.svm import SVC

clf = SVC().fit(X_train, y_train)
clf.score(X_test, y_test)


#Busca por proximidade usando Bag-of-Visual Words

def show_image(image_src):
    image = io.imread(image_src) 
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2_imshow(image)


max = 10
counter = 0
min_dist = 50
for index1,row1 in df_data.iterrows():
  for index2,row2 in df_data.iterrows():
    if index1==index2: continue
    img1 = row1['visual_words_histogram']  
    img2 = row2['visual_words_histogram']  
    dist = np.linalg.norm(img1-img2)
    if dist < min_dist:
      print(row1['imagem'],row2['imagem'],'dist=',dist)
      show_image(dataset+'/'+row1['imagem'])
      show_image(dataset+'/'+row2['imagem'])
      print('-------------------')
    counter += 1
  if counter > max: break

df_data

