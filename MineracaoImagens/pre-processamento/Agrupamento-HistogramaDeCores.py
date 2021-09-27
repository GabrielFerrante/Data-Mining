# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 15:00:57 2021

@author: gabriel
"""

import numpy as np
import pandas as pd
import cv2 as cv
#from google.colab.patches import cv2_imshow # for image display
from skimage import io
from PIL import Image 
import matplotlib.pylab as plt

#Carregar imagens
image = io.imread('http://farm4.static.flickr.com/3071/2762284293_0d6cef0979.jpg') 
image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
cv2_imshow(image)

image = io.imread('http://farm2.static.flickr.com/1353/1230897342_2bd7c7569f.jpg') 
image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
cv2_imshow(image)

print('Altura: ',image.shape[0])
print('Largura: ',image.shape[1])
print('#Canais: ', image.shape[2])


#Gerando o histograma de cor para cada canal

color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv.calcHist([image],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])

plt.show()

color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv.calcHist([image],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])

plt.show()


#Gerando o histograma de core para escala de cinza

gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
cv2_imshow(gray_image)

plt.hist(gray_image.ravel(),bins = 256, range = [0, 256])
plt.show()

#Vamos construir nosso dataset

links = ['http://farm2.static.flickr.com/1353/1230897342_2bd7c7569f.jpg',
'http://farm2.static.flickr.com/1016/1306604645_3993048c41.jpg',
'http://farm1.static.flickr.com/81/207747002_37e2540d6c.jpg',
'http://farm2.static.flickr.com/1435/1200081464_30fae875f0.jpg',
'http://farm4.static.flickr.com/3043/2597925441_0864d85fd1.jpg',
'http://farm1.static.flickr.com/86/263170989_6abf355855.jpg',
'http://farm1.static.flickr.com/54/177164374_e6c248e694.jpg',
'http://farm3.static.flickr.com/2406/2260615381_e9812d45a9.jpg',
'http://farm3.static.flickr.com/2388/2174456425_861222cd58.jpg',
'http://farm1.static.flickr.com/105/309908091_0ad0a44c9e.jpg',
'http://farm1.static.flickr.com/137/320656678_97406f5b17.jpg',
'http://farm3.static.flickr.com/2031/2050579839_c8c35580bd.jpg',
'http://farm3.static.flickr.com/2359/2223801603_4ba2a35128.jpg',
'http://farm4.static.flickr.com/3071/2762284293_0d6cef0979.jpg',
'http://farm1.static.flickr.com/190/455149905_9cec4da2e5.jpg',
'http://farm1.static.flickr.com/116/272861209_8b3a480f0f.jpg',
'http://farm1.static.flickr.com/128/406639889_aa7d37fc76.jpg',
'http://farm3.static.flickr.com/2110/2104917897_2aaa85bd4d.jpg',
'http://farm3.static.flickr.com/2225/2295004064_dc0392e64f.jpg',
'http://farm4.static.flickr.com/3246/2360351827_2f70df00f5.jpg',
'http://farm4.static.flickr.com/3106/2338086144_607c18479e.jpg',
'http://farm1.static.flickr.com/86/274560239_9bcc6e8fb5.jpg',
'http://farm2.static.flickr.com/1336/1377795483_5bf4c5a62e.jpg',
'http://farm1.static.flickr.com/29/53089209_7eaa21ab1c.jpg',
'http://farm3.static.flickr.com/2282/1801906722_7f90b20741.jpg',
'http://farm3.static.flickr.com/2122/2533944531_6e88004ab4.jpg',
'http://farm1.static.flickr.com/142/350135966_6cfe2dfc9d.jpg',
'http://farm1.static.flickr.com/84/237236358_e030ccd9ef.jpg',
'http://farm1.static.flickr.com/58/171100724_f0212e55b5.jpg',
'http://farm3.static.flickr.com/2103/2141928076_9ecacb58cc.jpg',
'http://farm1.static.flickr.com/171/470375092_6679a97792.jpg',
'http://farm3.static.flickr.com/2253/2459473980_21b897beba.jpg',
'http://farm1.static.flickr.com/70/197585850_a283f8c84b.jpg',
'http://farm2.static.flickr.com/1099/888435770_428ff2f867.jpg',
'http://farm1.static.flickr.com/69/153992025_b7a81b477b.jpg',
'http://farm2.static.flickr.com/1132/898243666_58834f6686.jpg',
'http://farm1.static.flickr.com/168/406941658_d18e6868bd.jpg',
'http://farm1.static.flickr.com/167/421756870_584a8844a9.jpg',
'http://farm1.static.flickr.com/33/60267188_afb2f63d6b.jpg',
'http://farm1.static.flickr.com/6/69239736_8e50418703.jpg',
'http://static.flickr.com/67/201636953_83ebaf707e.jpg',
'http://farm2.static.flickr.com/1039/1159729328_3d35191139.jpg',
'http://farm1.static.flickr.com/209/467273152_79a0590b94.jpg',
'http://farm1.static.flickr.com/46/139890893_7ad5cacb27.jpg',
'http://farm1.static.flickr.com/185/428657905_0c3230d167.jpg',
'http://farm1.static.flickr.com/39/85260161_fa6f30757e.jpg',
'http://farm1.static.flickr.com/66/185059444_b9564085af.jpg',
'http://farm1.static.flickr.com/51/115789599_2f251b75da.jpg',
'http://farm1.static.flickr.com/154/404375520_4784b1f5b8.jpg',
'http://farm1.static.flickr.com/207/471466323_c2401cafd5.jpg',
'http://farm2.static.flickr.com/1365/863259458_f97c0f0332.jpg',
'http://static.flickr.com/2085/1804684289_b7c3699c03.jpg',
'http://farm1.static.flickr.com/163/429988777_f73a2665e8.jpg',
'http://farm1.static.flickr.com/136/390075985_f48cab9e8b.jpg',
'http://farm1.static.flickr.com/44/146461282_fec5e91344.jpg',
'http://farm4.static.flickr.com/3109/2320255944_498e371aa4.jpg',
'http://farm1.static.flickr.com/65/215333205_70036372e9.jpg',
'http://farm2.static.flickr.com/1080/1179106100_aa9f2c98d9.jpg',
'http://farm1.static.flickr.com/125/350602385_85f68393a6.jpg',
'http://farm3.static.flickr.com/2245/2415172467_3341938774.jpg',
'http://farm4.static.flickr.com/3199/2703379801_d30c64c4bc.jpg',
'http://farm3.static.flickr.com/2153/2432722585_f81524e391.jpg',
'http://farm1.static.flickr.com/26/45058883_0ba5a99080.jpg',
'http://farm3.static.flickr.com/2293/2382982837_54be5cc4e4.jpg',
'http://farm4.static.flickr.com/3184/2295157330_f7a8093646.jpg',
'http://farm3.static.flickr.com/2097/2262391575_4b7f825c53.jpg',
'http://farm4.static.flickr.com/3201/2291925076_940d6215d2.jpg',
'http://farm1.static.flickr.com/79/237236334_224c8c20f8.jpg',
'http://farm1.static.flickr.com/134/366412559_af2f42f17f.jpg',
'http://farm4.static.flickr.com/3080/2291530231_7c917eb6b3.jpg',
'http://farm3.static.flickr.com/2188/1555133897_10287c9fbe.jpg',
'http://farm1.static.flickr.com/134/327524420_dc5c486e18.jpg',
'http://farm2.static.flickr.com/1222/1182963574_83ad2319da.jpg',
'http://farm4.static.flickr.com/3212/2288379817_b720d131fa.jpg',
'http://farm2.static.flickr.com/1335/1428968915_98086f2dfa.jpg',
'http://farm4.static.flickr.com/3045/2637419284_1d36095cdc.jpg',
'http://farm1.static.flickr.com/82/237165134_72e9489aab.jpg',
'http://farm4.static.flickr.com/3083/2564902777_bdb0a1c0bd.jpg',
'http://farm4.static.flickr.com/3290/2308017381_5a8b2999a6.jpg',
'http://farm1.static.flickr.com/15/21280319_1d418b6652.jpg',
'http://farm1.static.flickr.com/82/278754831_c8776cc5f5.jpg',
'http://farm2.static.flickr.com/1291/867490321_2010f3563b.jpg',
'http://farm2.static.flickr.com/1268/1328783278_169a0592ef.jpg',
'http://farm2.static.flickr.com/1220/1017270995_cc43a603bb.jpg',
'http://farm1.static.flickr.com/131/356141168_997d785cb0.jpg',
'http://farm4.static.flickr.com/3243/2302274471_55df316d4d.jpg',
'http://farm3.static.flickr.com/2225/2062534589_7e473b108c.jpg',
'http://farm3.static.flickr.com/2331/1896877543_41042098a3.jpg',
'http://farm1.static.flickr.com/27/42144533_934946d5e7.jpg',
'http://farm1.static.flickr.com/66/201734928_767e867dce.jpg',
'http://farm3.static.flickr.com/2168/2247351256_873e8a75e0.jpg',
'http://farm3.static.flickr.com/2153/2324724148_69ba45c3af.jpg',
'http://farm3.static.flickr.com/2045/2274162936_75d24ba910.jpg',
'http://farm3.static.flickr.com/2330/1590630895_c3f421ca8f.jpg',
'http://farm1.static.flickr.com/171/446222646_77edc5ccc6.jpg',
'http://farm4.static.flickr.com/3261/2613115669_5925a48d2b.jpg',
'http://farm1.static.flickr.com/218/505899199_e3cbd9aab5.jpg',
'http://farm1.static.flickr.com/211/526360274_e033600b43.jpg',
'http://farm3.static.flickr.com/2026/1556007496_d9fbfbd747.jpg',
'http://farm1.static.flickr.com/110/316454688_9251b60a8a.jpg',
'http://farm4.static.flickr.com/3063/2316156998_20ffe25145.jpg',
'http://farm3.static.flickr.com/2106/2142131954_e421fee877.jpg',
'http://farm1.static.flickr.com/22/32062548_c1cc0e34e1.jpg',
'http://farm2.static.flickr.com/1043/1351649886_2090307781.jpg',
'http://farm1.static.flickr.com/175/367205493_c6448d297e.jpg',
'http://farm1.static.flickr.com/46/139316010_9a30abcf9b.jpg',
'http://farm4.static.flickr.com/3017/2639544912_4aacc0f3bb.jpg',
'http://farm1.static.flickr.com/231/500059788_889487a67e.jpg',
'http://farm1.static.flickr.com/145/403078018_39fb515be6.jpg',
'http://farm3.static.flickr.com/2202/1926711075_82e2cc7b05.jpg',
'http://farm1.static.flickr.com/6/7338482_e344e0dbcd.jpg',
'http://farm2.static.flickr.com/1065/1393401918_619a60b006.jpg',
'http://farm3.static.flickr.com/2399/1501994672_46b0ef7993.jpg',
'http://farm1.static.flickr.com/28/59260298_60126c75c3.jpg',
'http://farm1.static.flickr.com/9/14057620_1596e48da1.jpg',
'http://farm3.static.flickr.com/2026/1544063360_7ec3529266.jpg',
'http://farm2.static.flickr.com/1308/660599591_dcfe11774a.jpg',
'http://farm1.static.flickr.com/151/336624970_7254c3e279.jpg',
'http://farm1.static.flickr.com/18/69740649_6075d36e8e.jpg',
'http://farm1.static.flickr.com/212/535753244_658439afdf.jpg',
'http://farm1.static.flickr.com/128/360815568_b9f9e1bb29.jpg']

dataset = []
for img_url in links:
  try:
    image = io.imread(img_url) 
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    dataset.append(image)
  except:
    print('Warning: ',img_url)
    

plt.subplots(figsize=(30,50)) 
columns = 5
for i, image in enumerate(dataset):
    plt.subplot(len(dataset) / columns + 1, columns, i + 1)
    plt.imshow(cv.cvtColor(image, cv.COLOR_RGB2BGR))  

#Extraindo características do dataset usando Histograma de Cor

color = ('b','g','r')

dataset_hist_r = []
dataset_hist_g = []
dataset_hist_b = []

counter = 0
for image in dataset:
  hists = {}
  for i,col in enumerate(color):
    histr = cv.calcHist([image],[i],None,[256],[0,256])
    if col == 'r': dataset_hist_r.append(histr)
    if col == 'g': dataset_hist_g.append(histr)
    if col == 'b': dataset_hist_b.append(histr)


X_r = np.array(dataset_hist_r)
length = np.sqrt((X_r**2).sum(axis=1))[:,None]
X_r = X_r / length

X_g = np.array(dataset_hist_g)
length = np.sqrt((X_g**2).sum(axis=1))[:,None]
X_g = X_g / length


X_b = np.array(dataset_hist_b)
length = np.sqrt((X_b**2).sum(axis=1))[:,None]
X_b = X_b / length

X = np.concatenate((X_r,X_g,X_g),axis=1)
X.shape


X = X.reshape(X.shape[0],X.shape[1])
X.shape


#Agrupamento de Imagens

from sklearn.cluster import KMeans
import numpy as np

kmeans = KMeans(n_clusters=15).fit(X)
kmeans.labels_

for image_id, cluster in enumerate(kmeans.labels_):
    if cluster == 9:
      print(image_id)
      cv2_imshow(dataset[image_id])
      

