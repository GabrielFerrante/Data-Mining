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

#Exemplo de Remoção de Stopwords

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

dataset.Sentiment.unique()

dataset = dataset[dataset.Sentiment=='Extremely Negative'][['OriginalTweet']].dropna()
dataset.reset_index(inplace=True,drop=True)
dataset

#Computando uma Bag-of-Words e aplicando TFIDF

# obtendo a VSM com TFIDF
def compute_vsm_tfidf(dataset,lang,domain_stopwords=[]):
  
  d = []
  for index,row in dataset.iterrows():
    text = row['OriginalTweet'] 
    text2 = remove_stopwords(text, lang,domain_stopwords)
    text3 = stemming(text2, lang)
    d.append(text3)
  
  matrix = TfidfVectorizer(max_features=1000,)
  X = matrix.fit_transform(d)
  
  tfidf_vect_df = pd.DataFrame(X.todense(), columns=matrix.get_feature_names())

  return tfidf_vect_df


vsm = compute_vsm_tfidf(dataset,'english')
vsm

#Agrupamento Particional

# preparando os dados para o k-means

X = np.array(vsm)
length = np.sqrt((X**2).sum(axis=1))[:,None]
X = X / length


#Ao normalizar os vetores, podemos usar k-means com distância euclidiana. Prova: VER IMAGE1

from sklearn.cluster import KMeans
import numpy as np
from sklearn import metrics

for k in range(2,21):

  kmeans = KMeans(n_clusters=k).fit(X)
  sil = metrics.silhouette_score(X, kmeans.labels_)
  print('k=',k,'silhouette=',sil)


kmeans = KMeans(n_clusters=10, random_state=42).fit(X)
df_ = vsm
df_['grupo'] = kmeans.labels_ 
df_['silhouette'] = metrics.silhouette_samples(X, kmeans.labels_)
df_


df_[['grupo']].hist()


#Olhando representantes de cada grupo via Silhouette

df_[df_.grupo==0].sort_values('silhouette',ascending=False)

#Olhando top-5 em cada cluster

pd.set_option('display.max_colwidth', None) # mostrar sem truncar as linhas

dataset.iloc[df_[df_.grupo==0].sort_values('silhouette',ascending=False).head(5).index]

dataset.iloc[df_[df_.grupo==1].sort_values('silhouette',ascending=False).head(5).index]

dataset.iloc[df_[df_.grupo==2].sort_values('silhouette',ascending=False).head(5).index]

dataset.iloc[df_[df_.grupo==3].sort_values('silhouette',ascending=False).head(5).index]

dataset.iloc[df_[df_.grupo==4].sort_values('silhouette',ascending=False).head(5).index]

dataset.iloc[df_[df_.grupo==5].sort_values('silhouette',ascending=False).head(5).index]

dataset.iloc[df_[df_.grupo==6].sort_values('silhouette',ascending=False).head(5).index]

dataset.iloc[df_[df_.grupo==7].sort_values('silhouette',ascending=False).head(5).index]

dataset.iloc[df_[df_.grupo==8].sort_values('silhouette',ascending=False).head(5).index]

dataset.iloc[df_[df_.grupo==9].sort_values('silhouette',ascending=False).head(5).index]

#Exportando para análise em outras ferramentas

dataset2 = dataset.copy()
dataset2['cluster'] = 'cluster_'
dataset2['cluster'] += df_.grupo.astype(str)
dataset2

dataset2.to_excel('analise.xls')

df_.to_csv('projector.tsv',sep="\t",index=False,header=None)

L = []
for index,row in dataset2.iterrows():
  L.append(row['OriginalTweet'].replace('\n', ' ').replace('\r', ''))
dataset2['TweetLines'] = L

dataset2[['TweetLines','cluster']].to_csv('labels.tsv',sep="\t",index=True)