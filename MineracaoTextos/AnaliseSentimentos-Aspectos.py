# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 14:50:36 2021

@author: gabriel
"""

import pandas as pd
df_train = pd.read_excel('restaurants_train.xls');
df_train


df_test = pd.read_excel('restaurants_test.xls');
df_test

#treinando o modelo

import pandas as pd

# treino
x_train = df_train[['review', 'aspect']].values
y_train = df_train['sentiment'].apply(str).values

# teste
x_test = df_test[['review', 'aspect']].values
y_test = df_test['sentiment'].apply(str).values


# IMPORTANT: data format for sentence pair classification is list of tuples of form (str, str)
x_train = list(map(tuple, x_train))
x_test = list(map(tuple, x_test))


#Selecionando um modelo, treinando e avaliando


import ktrain
from ktrain import text
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import numpy as np



# modelo selecionado (# https://huggingface.co/transformers/pretrained_models.html)

# pré-processamento
t = text.Transformer('bert-base-cased', maxlen=128)
trn = t.preprocess_train(x_train, y_train)
val = t.preprocess_test(x_test, y_test)

# treinamento
model = t.get_classifier()
learner = ktrain.get_learner(model, train_data=trn, val_data=val, batch_size=32)
learner.fit_onecycle(0.00005, 5)


#Avaliando com outras métricas

p = ktrain.get_predictor(model, preproc)