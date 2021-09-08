# keras (redes neurais)
import keras
from keras.preprocessing.text import one_hot,Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense , Flatten ,Embedding,Input
from keras.models import Model

#nltk (tokenizacao e remocao de stopwords)
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk import word_tokenize
stop_words=set(nltk.corpus.stopwords.words('english'))

#Carregando corpus

doc1="o cachorro gosta de osso"
doc2="o gato gosta de peixe"
doc3="o cachorro brincou com o gato"
doc4="o dia está quente"
doc5='Eu sentei no banco da praça'
doc6='Fui no banco conferir o meu saldo'

dataset=[doc1,doc2,doc3,doc4,doc5,doc6]
num_docs=len(dataset)

#Gerando código inteiro único para cada palavra

max_vocab=30
dataset_cod=[]
for i,doc in enumerate(dataset):
  onehot_enc = one_hot(doc,max_vocab)
  dataset_cod.append(onehot_enc)
  print("Documento ",i+1," : ",onehot_enc)


#Armazenando código de cada palavra

word_cods = {}
for i,doc in enumerate(dataset):
  tokens = word_tokenize(doc)
  for t in range(0,len(tokens)):
    word_cods[dataset_cod[i][t]] = tokens[t]

word_cods

#Padding / preenchimento
#Manter cada documento com o mesmo tamanho para apresentar à rede neural.

max_words_doc = 15  # número máximo de tokens para um documento, preenche o restante com zeros
dataset_pad=pad_sequences(dataset_cod,maxlen=max_words_doc)

for i,doc in enumerate(dataset_pad):
     print("Documento",i+1," : ",doc)

    
#Rede Neural (Embedding Layer)


dimensao = 2 # dimensao da nossa embedding

model = Sequential()
model.add(Embedding(max_vocab, dimensao, input_length=max_words_doc))
model.add(Flatten())
model.add(Dense(max_words_doc, activation='relu'))
model.compile(optimizer='adam', loss='mse')
model.summary()

model.fit(dataset_pad,dataset_pad,epochs=100)


# camadas da rede neural
model.layers
# verificando a primeira camada (embeddings)  (#vocabulario vs #dimensao)
embeddings = model.layers[0].get_weights()[0]
embeddings.shape


import pandas as pd

matrix_embeddings = {}
for cod in word_cods:
  matrix_embeddings[word_cods[cod]] = embeddings[cod]

word_embeddings = pd.DataFrame(matrix_embeddings).transpose()  

word_embeddings

import matplotlib.pyplot as plt

x = word_embeddings[0].to_list()
y = word_embeddings[1].to_list()
n = word_embeddings.index.to_list()

fig, ax = plt.subplots()
ax.scatter(x, y)

for i, txt in enumerate(n):
    ax.annotate(txt, (x[i], y[i]))