# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 13:42:03 2021

@author: gabriel
"""

#Extração de Embeddings a partir de Imagens - Modelos DNN Pré-Treinados

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image

#Carregando modelo pré-treinado (ResNet18)

"""
He, Kaiming, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. "Deep residual learning for image recognition." In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 770-778. 2016.

Estou usando a ResNet18. Outras opções de modelos pré-treinados estão listadas abaixo.

OBS: Se for trocar o modelo, precisa pesquisar a dimensão da imagem em que o modelo foi treinado.

https://pytorch.org/vision/stable/models.html

* resnet18 = models.resnet18(pretrained=True)
* alexnet = models.alexnet(pretrained=True)
* squeezenet = models.squeezenet1_0(pretrained=True)
* vgg16 = models.vgg16(pretrained=True)
* densenet = models.densenet161(pretrained=True)
* inception = models.inception_v3(pretrained=True)
* googlenet = models.googlenet(pretrained=True)
* shufflenet = models.shufflenet_v2_x1_0(pretrained=True)
* mobilenet_v2 = models.mobilenet_v2(pretrained=True)
* mobilenet_v3_large = models.mobilenet_v3_large(pretrained=True)
* mobilenet_v3_small = models.mobilenet_v3_small(pretrained=True)
* resnext50_32x4d = models.resnext50_32x4d(pretrained=True)
* wide_resnet50_2 = models.wide_resnet50_2(pretrained=True)
* mnasnet = models.mnasnet1_0(pretrained=True)

"""

model = models.resnet18(pretrained=True)

layer = model._modules.get('avgpool')
model.eval() # coloca a rede em model eval() para desativar as dropout layers (sao uteis durante o treinamento apenas)


#Transformação das Imagens
#ResNet utiliza imagens com 224x224 pixels. Imagens fora dessa dimensão são automaticamente ajustadas.

scaler = transforms.Scale((224, 224))
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
to_tensor = transforms.ToTensor()

#Gerando Embeddings

import numpy as np

def gerar_embeddings(arquivo_imagem):
    
    img = Image.open(arquivo_imagem)
    img_transformada = Variable(normalize(to_tensor(scaler(img))).unsqueeze(0))

    embedding = torch.zeros(512) # armazenado espaço

    def capturar_embedding(m, i, o):
        embedding.copy_(o.data.reshape(o.data.size(1)))
    
    # capturando embeddings
    h = layer.register_forward_hook(capturar_embedding) 
    model(img_transformada)
    
    h.remove()
    
    return np.array(embedding)


"""
Testes
Testando similaridade entre gatos e cachorro.

Usamos a dissimilaridade de cosseno (quanto menor, mais próximas são as imagens)
"""


#Baixando 

"""
!wget https://upload.wikimedia.org/wikipedia/commons/thumb/1/1d/1-month-old_kittens_32.jpg/1200px-1-month-old_kittens_32.jpg -O gato1.jpg
!wget https://upload.wikimedia.org/wikipedia/commons/thumb/0/07/A_focused_kitten_%28Flickr%29.jpg/1200px-A_focused_kitten_%28Flickr%29.jpg -O gato2.jpg
!wget https://upload.wikimedia.org/wikipedia/commons/thumb/2/22/An_Indian_Spitz_Dog_with_pointy_ears_and_conical_snout_2021.jpg/800px-An_Indian_Spitz_Dog_with_pointy_ears_and_conical_snout_2021.jpg -O cachorro1.jpg
!wget https://upload.wikimedia.org/wikipedia/commons/thumb/b/b3/Female_house_sparrow_at_Kodai.jpg/1280px-Female_house_sparrow_at_Kodai.jpg -O passarinho.jpg

"""

gato1_embedding = gerar_embeddings('gato1.jpg')
gato2_embedding = gerar_embeddings('gato2.jpg')
cachorro1_embedding = gerar_embeddings('cachorro1.jpg')
passarinho_embedding = gerar_embeddings('passarinho.jpg')


gato1_embedding.shape

#Testando a dissimilariedade entre imagens
from IPython.display import Image
from IPython.display import display
from scipy.spatial import distance

#Do mesmo gato
x = Image(filename='gato1.jpg',width=224) 
y = Image(filename='gato1.jpg',width=224) 
display(x, y)

d_cos = distance.cosine(gato1_embedding, gato1_embedding) # dissimilaridade de cosseno
print('Dissimilaridade de Cosseno: ',d_cos)

#Dois gatos diferentes
x = Image(filename='gato1.jpg',width=224) 
y = Image(filename='gato2.jpg',width=224) 
display(x, y)

d_cos = distance.cosine(gato1_embedding, gato2_embedding) # dissimilaridade de cosseno
print('Dissimilaridade de Cosseno: ',d_cos)

#Entre um gato e um cachorro
x = Image(filename='gato1.jpg',width=224) 
y = Image(filename='cachorro1.jpg',width=224) 
display(x, y)

d_cos = distance.cosine(gato1_embedding, cachorro1_embedding) # dissimilaridade de cosseno
print('Dissimilaridade de Cosseno: ',d_cos)

#Entre um gato e um passaro

x = Image(filename='gato2.jpg',width=224) 
y = Image(filename='passarinho.jpg',width=224) 
display(x, y)

d_cos = distance.cosine(gato2_embedding, passarinho_embedding) # dissimilaridade de cosseno
print('Dissimilaridade de Cosseno: ',d_cos)


