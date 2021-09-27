# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 14:37:50 2021

@author: gabriel
"""

import keras
import ktrain
from ktrain import text

from sklearn.datasets import fetch_20newsgroups
remove = ('headers', 'footers', 'quotes')
newsgroups_train = fetch_20newsgroups(subset='train', remove=remove)
newsgroups_test = fetch_20newsgroups(subset='test', remove=remove)
docs = newsgroups_train.data +  newsgroups_test.data

#Indexando os textos
text.SimpleQA.initialize_index("meu_indice")
text.SimpleQA.index_from_list(docs, "meu_indice")


#Carregando um modelo BERT treinado para QA

qa = text.SimpleQA("meu_indice")

#Fazendo perguntas

answers = qa.ask('What causes computer images to be too dark?')
qa.display_answers(answers[:5])

answers = qa.ask('What is the best operating system?')
qa.display_answers(answers[:5])

answers = qa.ask('What is the best modem?')
qa.display_answers(answers[:5])