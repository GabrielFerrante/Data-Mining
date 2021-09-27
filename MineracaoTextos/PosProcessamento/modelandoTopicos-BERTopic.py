# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 14:42:41 2021

@author: gabriel
"""

from sklearn.datasets import fetch_20newsgroups
docs = fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))['data']

#Modelagem de tópicos

from bertopic import BERTopic

lang = "english"  # pode ser multilingual

topic_model = BERTopic(language="english", calculate_probabilities=True, verbose=True)
topics, probs = topic_model.fit_transform(docs)


#Visualizando

topic_model.visualize_barchart(top_n_topics=5)

topic_model.visualize_barchart(top_n_topics=10)

topic_model.visualize_topics()

