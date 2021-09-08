#Carregando base de textos (Twitter)

# https://www.kaggle.com/datatattle/covid-19-nlp-text-classification
!unzip tweets_corona_virus.zip


import pandas as pd
dataset = pd.read_csv('Corona_NLP_train.csv',encoding='iso8859')
dataset

#Treinando sua Word Embeddings

import gensim 

# percorre cada linha do dataset, preprocessa e armazena na lista documents

documents = []
for index,row in dataset.iterrows(): # para cada linha do dataset
  if len(row['OriginalTweet']) > 30: # se a linha possui mais do que 50 caracteres
    tokens = gensim.utils.simple_preprocess (row['OriginalTweet']) # preprocessamento simples de cada texto
    documents.append(tokens)

len(dataset)

import numpy as np
len(documents)

# treinando o Word2Vec a partir dos documents
model = gensim.models.Word2Vec(
        documents, # lista com documents
        size=300, # tamanho da dimensao de cada palavra
        window=2, # tamanho da janela de contexto
        min_count=1, # numero minimo de ocorrencia de uma palavra no texto
        workers=4, # paralelizacao/cpu
        iter=10) # numero maximo de iteracoes


# testando o modelo..

palavra = 'covid'
model.wv.most_similar(palavra, topn=7) # identificando as 7 palavra mais similares (e.g. cosseno)


# testando o modelo..

palavra = 'crisis'
model.wv.most_similar(palavra, topn=7) # identificando as 7 palavra mais similares (e.g. cosseno)

word2vec = model

#Repetindo a classificação do exemplo anterior

dataset = dataset[(dataset.Sentiment=='Extremely Negative') | (dataset.Sentiment=='Extremely Positive')][['OriginalTweet','Sentiment']].dropna().sample(2000,random_state=42)
dataset.reset_index(inplace=True,drop=True)
dataset


dataset.Sentiment.hist()

#Separando treino e teste

import numpy as np
from sklearn.model_selection import train_test_split


df_train, df_test = train_test_split(dataset, test_size=0.30, random_state=42)

#Classificação usando Média de Word Vectors

from nltk.corpus import stopwords
from nltk import download
download('stopwords')  # Download stopwords list.
stop_words = stopwords.words('english')

def preprocess(sentence):
    return [w for w in sentence.lower().split() if w not in stop_words]

doc_embeddings = []
for index,row in df_train.iterrows():
  sentence = preprocess(row['OriginalTweet'])
  L = []
  for token in sentence:
    try:
      #print(token,word2vec[token])
      L.append(word2vec.wv[token])
    except:
      1
  if len(L) > 0: tweet_vec = np.mean(np.array(L),axis=0)
  else: tweet_vec = np.zeros(300)
  doc_embeddings.append(tweet_vec)


X_train = np.array(doc_embeddings)
y_train = df_train.Sentiment.to_list()
X_train.shape


doc_embeddings = []
for index,row in df_test.iterrows():
  sentence = preprocess(row['OriginalTweet'])
  L = []
  for token in sentence:
    try:
      #print(token,word2vec[token])
      L.append(word2vec[token])
    except:
      1
  if len(L) > 0: tweet_vec = np.mean(np.array(L),axis=0)
  else: tweet_vec = np.zeros(300)
  doc_embeddings.append(tweet_vec)


X_test = np.array(doc_embeddings)
y_test = df_test.Sentiment.to_list()
X_test.shape


from sklearn.neighbors import KNeighborsClassifier

neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)

neigh.score(X_test,y_test)