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
!pip install plotly.express
from plotly import graph_objs as go


#Removendo StopWords
nltk.corpus.stopwords.words('portuguese')

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



#Exemplo de Stemming/Radicalização de Termos


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

import urllib.parse
# obtendo dataset com uma amostra de textos

query = "febre amarela" # query para consultar 
query = urllib.parse.quote(query)

pd.set_option('display.max_colwidth', -1)
dataset = pd.read_csv('http://websensors.net.br/minicurso/2019/eventos-br-2017.php?q='+query,sep='\t', lineterminator='\n')
dataset


# filtrando apenas o título
dataset = dataset[['title']]
dataset


#Computando uma Bag-of-Words

# obtendo a bag-of-words
def compute_bag_of_words(dataset,lang,domain_stopwords=[]):
  
  d = []
  for index,row in dataset.iterrows():
    text = row['title'] #texto do evento
    text2 = remove_stopwords(text, lang,domain_stopwords)
    text3 = stemming(text2, lang)
    d.append(text3)
  
  matrix = CountVectorizer(max_features=1000)
  X = matrix.fit_transform(d)
  
  count_vect_df = pd.DataFrame(X.todense(), columns=matrix.get_feature_names())

  return count_vect_df


bow = compute_bag_of_words(dataset,'portuguese')
bow


#Ponderação de Termos com TFIDF

# obtendo a VSM com TFIDF
def compute_vsm_tfidf(dataset,lang,domain_stopwords=[]):
  
  d = []
  for index,row in dataset.iterrows():
    text = row['title'] #texto do evento
    text2 = remove_stopwords(text, lang,domain_stopwords)
    text3 = stemming(text2, lang)
    d.append(text3)
  
  matrix = TfidfVectorizer()
  X = matrix.fit_transform(d)
  
  tfidf_vect_df = pd.DataFrame(X.todense(), columns=matrix.get_feature_names())

  return tfidf_vect_df


vsm = compute_vsm_tfidf(dataset,'portuguese')
vsm


#Dissimilaridade de cosseno (1-cos) entre dois textos

# computando dissimilaridade de cosseno

def dis_cosine(matrix, e1, e2):
  dcos = cosine(matrix.iloc[e1,:], matrix.iloc[e2,:])
  return dcos


# exemplo: dissimilaride entre o primeiro (id=0) e o segundo evento (id=1) do vsm-tfidf
dis_cosine(vsm,1,2)

# similaridade entre documento 1 e os outros 1 até 10

for i in range(1,11):
  print('doc=',i,dis_cosine(vsm,1,i))