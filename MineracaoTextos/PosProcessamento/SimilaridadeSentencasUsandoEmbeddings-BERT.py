# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 13:57:48 2021

@author: gabriel
"""

from sentence_transformers import SentenceTransformer
import numpy as np
import logging

# Load Sentence model (based on BERT) from URL
model = SentenceTransformer('distiluse-base-multilingual-cased')

list(model.encode(['a polícia atacou o ladrão']))

model.encode(['o ladrão atacou a polícia'])

model.encode(['Der Dieb griff die Polizei an'])

import pandas as pd
dataset = pd.read_csv('Corona_NLP_train.csv',encoding='iso8859')
dataset


dataset = dataset[(dataset.Sentiment=='Extremely Negative') | (dataset.Sentiment=='Extremely Positive')][['OriginalTweet','Sentiment']].dropna().sample(2000,random_state=42)
dataset.reset_index(inplace=True,drop=True)
dataset


dataset['bert_embedding'] = list(model.encode(dataset.OriginalTweet.to_list()))

dataset