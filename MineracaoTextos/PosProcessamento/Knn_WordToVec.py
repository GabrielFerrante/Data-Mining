#Download do modelo pré-treinado (~1.5 gb)

import numpy as np

!wget -P /root/input/ -c "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"

#Instalando a biblioteca "gensim" para manipular word vectors


from gensim.models import KeyedVectors

#Carregando o modelo pré-treinado

EMBEDDING_FILE = '/root/input/GoogleNews-vectors-negative300.bin.gz' # from above
word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)

word2vec["smartphone"].shape

word2vec["corona"]

#Vamos usar a similaridade de cosseno para calcular a proximidade entre palavras
#VISUALIZAR IMAGE2

import numpy as np

def cos(x1, x2):
  return np.dot(x1, x2)/(np.linalg.norm(x1)*np.linalg.norm(x2))

#Testando similaridades..

cos(word2vec["smartphone"], word2vec["telephone"])

#Similaridade de Textos com Word Mover Distance

from nltk.corpus import stopwords
from nltk import download
download('stopwords')  # Download stopwords list.
stop_words = stopwords.words('english')

def preprocess(sentence):
    return [w for w in sentence.lower().split() if w not in stop_words]


sentence_obama = 'Obama speaks to the media in Illinois'
sentence_president = 'The president greets the press in Chicago'

sentence_obama = preprocess(sentence_obama)
sentence_president = preprocess(sentence_president)


distance = word2vec.wmdistance(sentence_obama, sentence_president)
print('distance = %.4f' % distance)

sentence_orange = preprocess('Oranges are my favorite fruit')
distance = word2vec.wmdistance(sentence_obama, sentence_orange)
print('distance = %.4f' % distance)


#Carregando base de textos (Twitter)

# https://www.kaggle.com/datatattle/covid-19-nlp-text-classification
!unzip tweets_corona_virus.zip

import pandas as pd
dataset = pd.read_csv('Corona_NLP_train.csv',encoding='iso8859')
dataset

dataset = dataset[(dataset.Sentiment=='Extremely Negative') | (dataset.Sentiment=='Extremely Positive')][['OriginalTweet','Sentiment']].dropna().sample(2000,random_state=42)
dataset.reset_index(inplace=True,drop=True)
dataset

dataset.Sentiment.hist()


#Separando treino e teste

import numpy as np
from sklearn.model_selection import train_test_split


df_train, df_test = train_test_split(dataset, test_size=0.30, random_state=42)

#Classificação usando Média de Word Vectors

doc_embeddings = []
for index,row in df_train.iterrows():
  #print(row['OriginalTweet'])
  sentence = preprocess(row['OriginalTweet'])
  #print(sentence)
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

X_train = np.array(doc_embeddings)
y_train = df_train.Sentiment.to_list()
X_train.shape

pd.DataFrame(X_train)


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

#Exportando para análise visual

df_ = pd.concat([pd.DataFrame(X_train),pd.DataFrame(X_test)])
df_.reset_index(inplace=True,drop=True)
df_
df_.to_csv('projector2.tsv',sep="\t",index=False,header=None)