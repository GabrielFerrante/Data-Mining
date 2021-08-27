import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt


#LENDO O CSV

dados1 = pd.read_csv('kmeans_exemplo1.csv')
print(dados1)

#Plotando usando a função de espalhamento ("Scatter")

dados1.plot(kind='scatter',x=0,y=1)

"""
Executando o k-means com a seguinte configuração:

n_clusters=5 (número de clusters)
n_init=10 (quantidade de inicializações)
init='random' (inicialização aleatória dos centroides)
max_iter=300 (maximo de iterações)

"""

kmeans = KMeans(n_clusters=5,n_init=10,init='random',max_iter=300)
kmeans.fit(dados1)

"""
Verificando os clusters obtidos
"""

dados1['cluster'] = kmeans.labels_
print(dados1)

#Plotando usanfo a função de espalhamento
sns.scatterplot(data=dados1, x='0', y='1', hue="cluster")

#Centroides

centroids = kmeans.cluster_centers_
print(centroids)

#Erro quadrático, o que tiver menor valor, é o melhor agrupamento dado
#varias iterações
E = kmeans.inertia_
print(E)

"""
Vamos testar uma execução do k-means com finalização prematura, ou seja, o algoritmo finaliza antes de convergir.
"""
dados1 = pd.read_csv('kmeans_exemplo1.csv')

kmeans = KMeans(n_clusters=5,n_init=1,init='random',max_iter=2)
kmeans.fit(dados1)

dados1['cluster'] = kmeans.labels_

sns.scatterplot(data=dados1, x='0', y='1', hue="cluster")


"""
Exemplo 2, PROBLEMA DE CONJUNTO DE DADOS NÃO GLOBULARES
"""

dados2 = pd.read_csv('kmeans_exemplo2.csv')
print(dados2)

sns.scatterplot(data=dados2, x='x', y='y')

kmeans = KMeans(n_clusters=2,n_init=10,init='random',max_iter=300)
kmeans.fit(dados2)

dados2['cluster'] = kmeans.labels_
sns.scatterplot(data=dados2, x='x', y='y', hue="cluster")

print(kmeans.cluster_centers_)


"""
EXEMPLO 3, PROBLEMA DE CONJUNTO DE DADOS COM CLUSTERS COM DENSIDADES ALTAS
"""

dados3 = pd.read_csv('kmeans_exemplo3.csv')
print(dados3)

sns.scatterplot(data=dados3, x='x', y='y')

kmeans = KMeans(n_clusters=4,n_init=10,init='random',max_iter=300)
kmeans.fit(dados3)

dados3['cluster'] = kmeans.labels_

sns.scatterplot(data=dados3, x='x', y='y', hue='cluster')

print(kmeans.cluster_centers_)

"""
Vamos tentar mitigar esse problema, aumentando o número de clusters!
"""

dados3 = pd.read_csv('kmeans_exemplo3.csv')

kmeans = KMeans(n_clusters=16,n_init=10,init='random',max_iter=300)
kmeans.fit(dados3)

dados3['cluster'] = kmeans.labels_

sns.scatterplot(data=dados3, x='x', y='y', hue="cluster", legend=None)

"""
Exemplo 4
Conjunto de dados com sete tipos diferentes de grãos de feijão secos, levando em consideração as características como forma, formato, tipo e estrutura. Um sistema de visão computacional foi desenvolvido para distinguir sete diferentes variedades registradas de feijão seco com características semelhantes. Imagens de 13.611 grãos de 7 diferentes grãos secos registrados foram tiradas com uma câmera de alta resolução.

Koklu, Murat, and Ilker Ali Ozkan. "Multiclass classification of dry beans using computer vision and machine learning techniques." Computers and Electronics in Agriculture 174 (2020): 105507.

"""

df = pd.read_csv('kmeans_exemplo4_drybean.csv')
print(df)

#Plotando um histograma da frequencia

#argumento bins: Número de compartimentos do histograma a serem usados
#argumento xrot: Rotação dos labels do eixo X, em graus
df.Class.hist(bins=7,xrot=90)

"""
Vamos verificar se o k-means consegue capturar, 
de forma não supervisionada, grupos de sementes com 
características similares.
"""

"""
Excluindo a coluna classe, que classifica. 
"""
dados4 = df.copy()
del dados4['Class']

kmeans = KMeans(n_clusters=7,n_init=10,init='random',max_iter=300)
kmeans.fit(dados4)
dados4['cluster'] = kmeans.labels_

sns.scatterplot(data=dados4, x='Extent', y='Compactness', hue="cluster")

"""Por padrão, esta função criará uma grade de eixos 
de forma que cada variável numérica nos dados seja compartilhada 
entre os eixos y em uma única linha e os eixos x em uma única coluna. 
Os gráficos diagonais são tratados de forma diferente: um gráfico de 
distribuição univariada é desenhado para mostrar a distribuição marginal
 dos dados em cada coluna."""

sns.pairplot(dados4,hue='cluster')

df['cluster'] = kmeans.labels_
"""
Lendo os 30 primeiros
"""
df.head(30)

"""
Lendo os 30 ultimos
"""
df.tail(30)