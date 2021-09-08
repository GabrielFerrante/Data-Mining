# Importando bibliotecas
import pandas as pd
import string
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('rslp')
from nltk.tokenize import word_tokenize
from nltk.stem.porter import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cosine
import numpy as np
import networkx as nx
from plotly import graph_objs as go


# remoção de pontuacao e stopwords

def remove_stopwords(text,lang,domain_stopwords=[]):
  
  stop_words = nltk.corpus.stopwords.words(lang) # lang='portuguese' or lang='english'
  
  s = str(text).lower() # tudo para caixa baixa
  table = str.maketrans({key: None for key in string.punctuation})
  s = s.translate(table) # remove pontuacao
  tokens = word_tokenize(s) #obtem tokens
  v = [i for i in tokens if not i in stop_words and not i in domain_stopwords and not i.isdigit()] # remove stopwords
  s = ""
  for token in v:
    s += token+" "
  return s.strip()


# exemplos de uso
text = "O estudante de Inteligência Artificial foi na livraria comprar  livros para estudar."
text2 = remove_stopwords(text, 'portuguese')
print('Antes: '+text)
print('Depois: '+text2)

# stemming
def stemming(text,lang):
  
  stemmer = PorterStemmer() # stemming para ingles
  
  if lang=='portuguese':
    stemmer = nltk.stem.RSLPStemmer() # stemming para portuguese
    
  tokens = word_tokenize(text) #obtem tokens
  
  sentence_stem = ''
  doc_text_stems = [stemmer.stem(i) for i in tokens]
  for stem in doc_text_stems:
    sentence_stem += stem+" "
    
  return sentence_stem.strip()


# exemplos de uso
text = "O estudante de Inteligência Artificial foi na livraria comprar livros para estudar."
text2 = remove_stopwords(text, 'portuguese')
text3 = stemming(text2, 'portuguese')
print('Antes: '+text)
print('Depois: '+text3)


#Coletando uma Base de Textos para Testar

# https://www.kaggle.com/datatattle/covid-19-nlp-text-classification
!unzip tweets_corona_virus.zip

dataset = pd.read_csv('Corona_NLP_train.csv',encoding='iso8859')
dataset

dataset = dataset[(dataset.Sentiment=='Extremely Negative') | (dataset.Sentiment=='Extremely Positive')][['OriginalTweet','Sentiment']].dropna().sample(2000,random_state=42)
dataset.reset_index(inplace=True,drop=True)
dataset

dataset.Sentiment.hist()

#Computando uma Bag-of-Words e aplicando TFIDF


# obtendo a VSM com TFIDF
def compute_vsm_tfidf(dataset,lang,domain_stopwords=[]):
  
  d = []
  for index,row in dataset.iterrows():
    text = row['OriginalTweet'] 
    text2 = remove_stopwords(text, lang,domain_stopwords)
    text3 = stemming(text2, lang)
    d.append(text3)
  
  matrix = TfidfVectorizer(max_features=500)
  X = matrix.fit_transform(d)
  
  tfidf_vect_df = pd.DataFrame(X.todense(), columns=matrix.get_feature_names())

  return tfidf_vect_df


vsm = compute_vsm_tfidf(dataset,'english')
vsm


#Preparando dados para classificador kNN
X = np.array(vsm)
length = np.sqrt((X**2).sum(axis=1))[:,None]
X = X / (length+0.00001)

X.shape

X

Y = dataset['Sentiment'].to_list()


#Ao normalizar os vetores, podemos distância euclidiana. Prova: IMAGE 1

#Vamos usar 70% (treino) e 30% (teste)

import numpy as np
from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=42)

from sklearn.neighbors import KNeighborsClassifier

neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)

neigh.score(X_test,y_test)


#Exportando para análise em outras ferramentas


dataset2 = dataset.copy()
dataset2

vsm.to_csv('projector.tsv',sep="\t",index=False,header=None)

L = []
for index,row in dataset2.iterrows():
  L.append(row['OriginalTweet'].replace('\n', ' ').replace('\r', ''))
dataset2['TweetLines'] = L

dataset2[['TweetLines','Sentiment']].to_csv('labels.tsv',sep="\t",index=True)