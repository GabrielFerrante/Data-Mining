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


from keras.layers import Input, Dense
from keras.models import Model

# qual a dimensao dos dados de entrada
input_dim = 500

# qual a dimensao da embedding que a rede neural vai aprender
encoding_dim = 10

# Definindo a camada de entrada da rede
input_data = Input(shape=(input_dim,))

# Definindo a camada intermediária da rede (e passa a camada de entrada como parametro)
encoded = Dense(encoding_dim, activation='relu')(input_data)

# Definindo a camada de saída da rede (e passa a camada intermediaria como parametro)
decoded = Dense(input_dim, activation='sigmoid')(encoded)

# Modelando o autoencoder (entrada e saida)
autoencoder = Model(input_data, decoded)

# Criando o modelo APENAS para o encoder (entrada = input_data e saida a layer intermediaria encoded)
encoder = Model(input_data, encoded)

# Gera de fato a rede neural
autoencoder.compile(optimizer='adam', loss='mse')


x_train = np.array(vsm)
x_train

autoencoder.fit(x_train, x_train,
                epochs=30,
                batch_size=8,
                shuffle=True)

encoded_data = encoder.predict(x_train)
encoded_data

import pandas as pd


df = pd.DataFrame(encoded_data)
df['sentiment'] = dataset['Sentiment']
df

import seaborn as sns

sns.scatterplot(data=df, x=0, y=1, hue="sentiment")