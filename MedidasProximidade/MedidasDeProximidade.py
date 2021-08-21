import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn import preprocessing


"""
Medidas de Proximidade
Para realizarmos experimentos com medidas de proximidade, usaremos uma amostra de uma base 
de dados sobre campanhas de marketing direto de uma instituição bancária portuguesa. 
As campanhas de marketing foram baseadas em ligações telefônicas. Mais informações sobre o 
conjunto de dados estão disponíveis em Moro et al. (2014).

[1] [Moro et al., 2014] S. Moro, P. Cortez and P. Rita. A Data-Driven Approach to Predict the Success of Bank Telemarketing. Decision Support Systems, Elsevier, 62:22-31, June 2014
"""

"""
Leitura dos dados bank_marketing.csv
Usamos o pandas para leitura do arquivo CSV. Em seguida, as 30 primeiras linhas são exibidas.
 Os dados já estão pré-processados.

Cada linha representa um cliente do banco. Cada coluna representa um atributo do cliente.

Atributos:

age: Idade do cliente
balance: saldo em conta do cliente
duration: duração (em segundos) do último contato do banco com o cliente
"""
#LENDO AS COLUNAS DO CSV
dados = pd.read_csv('bank.csv')
print(dados)
#LENDO OS 30 PRIMEIROS REGISTROS
print(dados.head(30))

#Uma breve análise dos atributos
#EXECUTAR UM POR VEZ COMENTANDO OS OUTROS
#Histograma da frequência das idades
dados.age.hist()

#Histograma do balanço
dados.balance.hist()

#Histograma da duracao
dados.duration.hist()

"""
Testando medidas de proximidade
Medidas: Euclidiana, Manhattan, Suprema (chebyshev)

Usaremos o método NearestNeighbors do Scikit-Learn para testar nossas medidas de proximidade.

metric = define a medida de proximidade desejada

n_neighbors = determina quantos vizinhos mais próximos iremos localizar considerando a medida selecionada.
"""

nbrs = NearestNeighbors(metric='euclidean',n_neighbors=7).fit(dados)

"""
Vamos localizar 7 objetos mais próximos de acordo com atributos definidos manualmente.
"""
age = 30
balance = 200
duration = 30

distances, indices = nbrs.kneighbors([[age,balance,duration]])

"""
A variável indices armazena os IDs (pandas) dos objetos mais próximos encontrados.

A variável distances armazena os valores das distâncias.
"""

print(indices[0])
print(distances)

"""
Vamos usar essas variáveis para filtrar os objetos mais próximos em um dataframe do pandas.
"""

resultado = dados.iloc[indices[0]]
print(resultado)

"""
Vamos adicionar os valores de distância nos nossos resultados.
"""

resultado['distance'] = distances[0]
print(resultado)

"""
Agora, vamos repetir esse procedimento para outras medidas de proximidade e comparar os resultados.
"""
"""
USANDO DISTANCIA MANHANTTAN
"""
nbrs = NearestNeighbors(metric='manhattan',n_neighbors=7).fit(dados)

age = 30
balance = 200
duration = 30

distances, indices = nbrs.kneighbors([[age,balance,duration]])
resultado = dados.iloc[indices[0]]
resultado['distance'] = distances[0]

print(resultado)

"""
USANDO DISTÂNCIA CHEBYSHEV
"""
nbrs = NearestNeighbors(metric='chebyshev',n_neighbors=7).fit(dados)

age = 30
balance = 200
duration = 30

distances, indices = nbrs.kneighbors([[age,balance,duration]])
resultado = dados.iloc[indices[0]]
resultado['distance'] = distances[0]

print(resultado)

"""
USANDO DISTÂNCIA MINKOWSKI
"""
nbrs = NearestNeighbors(metric='minkowski',n_neighbors=7, p=2).fit(dados)

age = 30
balance = 200
duration = 30

distances, indices = nbrs.kneighbors([[age,balance,duration]])
resultado = dados.iloc[indices[0]]
resultado['distance'] = distances[0]

print(resultado)

"""
Testando o efeito da normalização dos dados na medida de proximidade
"""
#Calculando o desvio padrão para cada coluna
dados.var()

"""
Calculando a distância euclidiana dos 7 vizinhos proximos do objeto 5
"""
nbrs = NearestNeighbors(metric='euclidean',n_neighbors=7).fit(dados)

#Pegando o objeto
objeto_id = 5
print(dados.loc[objeto_id],"\n")
age = dados.loc[objeto_id].age
balance = dados.loc[objeto_id].balance
duration = dados.loc[objeto_id].duration

#Calculando as distancias passando as informações do objeto 5
distances, indices = nbrs.kneighbors([[age,balance,duration]])
resultado = dados.iloc[indices[0]]
resultado['distance'] = distances[0]

print(resultado)

#Preprocessamento para a normalização, alterando os dados para
#uma mesma escala.
min_max_scaler = preprocessing.MinMaxScaler() # normalização min-max
#transformando o dataframe em normalizado
dados_norm = pd.DataFrame(min_max_scaler.fit_transform(dados))

#rotulando as colunas
dados_norm.columns = dados.columns
print(dados_norm)

#Desvio padrão para cada coluna
dados_norm.var()

print(dados_norm)

"""
Calculando a distância euclidiana dos 7 vizinhos próximos 
do objeto 5 normalizado na mesma escala

"""

nbrs = NearestNeighbors(metric='euclidean',n_neighbors=7).fit(dados_norm)

objeto_id = 5
print(dados_norm.loc[objeto_id],"\n")
age = dados_norm.loc[objeto_id].age
balance = dados_norm.loc[objeto_id].balance
duration = dados_norm.loc[objeto_id].duration

distances, indices = nbrs.kneighbors([[age,balance,duration]])
resultado = dados_norm.iloc[indices[0]]
resultado['distance'] = distances[0]

print(resultado)

"""
Calculando a distância manhattan dos 7 vizinhos próximos 
do objeto 5 normalizado na mesma escala

"""
nbrs = NearestNeighbors(metric='manhattan',n_neighbors=7).fit(dados_norm)

objeto_id = 5
print(dados_norm.loc[objeto_id],"\n")
age = dados_norm.loc[objeto_id].age
balance = dados_norm.loc[objeto_id].balance
duration = dados_norm.loc[objeto_id].duration

distances, indices = nbrs.kneighbors([[age,balance,duration]])
resultado = dados_norm.iloc[indices[0]]
resultado['distance'] = distances[0]

print(resultado)

"""
Leitura dos dados bank_marketing_binary.csv
Usamos o pandas para leitura do arquivo CSV. Em seguida, as 30 primeiras linhas são exibidas. Os dados já estão pré-processados.

Cada linha representa um cliente do banco. Cada coluna representa um atributo binário do cliente.
"""

dados = pd.read_csv('bank_marketing_binary.csv')
dados.head(30)

"""
Calculando a distância de Jaccard para dados binários
"""
nbrs = NearestNeighbors(metric='jaccard',n_neighbors=70).fit(dados)

marital_divorced=1
marital_married=0
marital_single=0

education_primary=0
education_secondary=0
education_tertiary=1

contact_cellular=1
loan=0
housing=1

distances, indices = nbrs.kneighbors([[marital_divorced,
                                       marital_married,
                                       marital_single,
                                       education_primary,
                                       education_secondary,
                                       education_tertiary,
                                       contact_cellular,
                                       loan,
                                       housing]])
resultado = dados.iloc[indices[0]]
resultado['distance'] = distances[0]

print(resultado)