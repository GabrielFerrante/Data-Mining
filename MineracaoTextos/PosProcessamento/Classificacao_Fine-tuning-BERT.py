# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 14:22:52 2021

@author: gabriel
"""

import pandas as pd
dataset = pd.read_csv('Corona_NLP_train.csv',encoding='iso8859')
dataset


dataset = dataset[(dataset.Sentiment=='Extremely Negative') | (dataset.Sentiment=='Extremely Positive')][['OriginalTweet','Sentiment']].dropna().sample(2000,random_state=42)
dataset.reset_index(inplace=True,drop=True)
dataset

dataset.Sentiment.hist()

import keras
import ktrain
from ktrain import text

#Pré-processando (tente alterar os parâmetros)
(x_train, y_train), (x_test, y_test), preproc = text.texts_from_df(dataset, 
                                                                   'OriginalTweet',
                                                                   label_columns='Sentiment',
                                                                   maxlen=64, 
                                                                   max_features=10000,
                                                                   preprocess_mode='bert',
                                                                   lang=None,
                                                                   val_pct = 0.3,
                                                                   random_state=42
                                                                   )

model = text.text_classifier('bert', (x_train, y_train) , preproc=preproc)
classifier = ktrain.get_learner(model, 
                             train_data=(x_train, y_train), 
                             val_data=(x_test, y_test),
                             batch_size=64
                             )


classifier.fit_onecycle(0.00002,5)

classifier.validate()

#Testando para novos textos

predictor = ktrain.get_predictor(classifier.model, preproc)

predictor.get_classes()

predictor.predict('"We also have family but we can\'t stay home...Be responsible and STAY HOME')


#Explicando predição

predictor.explain('"We also have family but we can\'t stay home...Be responsible and STAY HOME')