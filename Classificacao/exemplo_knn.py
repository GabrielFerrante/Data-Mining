import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,plot_confusion_matrix,confusion_matrix
from sklearn import preprocessing
import numpy as np
from sklearn.dummy import DummyClassifier




"""
Leitura da Base de Dados
O banco de dados contém 14 atributos com base em testes físicos de um paciente. Amostras de sangue foram coletadas e o paciente também realizou um breve teste de esforço. O atributo "target" refere-se à presença de doença cardíaca no paciente (classe). É um número inteiro (0 para nenhuma presença, 1 para presença).

Em geral, confirmar 100% se um paciente tem doença cardíaca pode ser um processo bastante invasivo, portanto, se pudermos criar um modelo que preveja com precisão a probabilidade de doença cardíaca, podemos ajudar a evitar procedimentos caros e invasivos.

age
sex
chest pain type (4 values)
resting blood pressure
serum cholestoral in mg/dl
fasting blood sugar > 120 mg/dl
resting electrocardiographic results (values 0,1,2)
maximum heart rate achieved
exercise induced angina
oldpeak = ST depression induced by exercise * relative to rest
the slope of the peak exercise ST segment
number of major vessels (0-3) colored by flourosopy
thal: 3 = normal; 6 = fixed defect; 7 = reversable defect
target:0 for no presence of heart disease, 1 for presence of heart disease

Referência: Detrano, R., Janosi, A., Steinbrunn, W., Pfisterer, M., Schmid, J., Sandhu, S., Guppy, K., Lee, S., & Froelicher, V. (1989). International application of a new probability algorithm for the diagnosis of coronary artery disease. American Journal of Cardiology, 64,304--310.
"""
#Lendo o CSV
df = pd.read_csv('heart.csv')
df

"""
Analisando atributos
Verificar relações simples entre atributos e a classe.
"""
#Criando o histograma
df.target.hist() # verificando balanceamento da base de dados

sns.pairplot(df[['age','trestbps','chol','oldpeak','target']], hue='target')
plt.show()

"""
Classificação kNN
Dividindo o conjunto de dados em treino (70%) e teste (30%).
"""
#random_state: Argumento para ramdomizar os conjuntos de treinamento e test
df_train, df_test = train_test_split(df, test_size=0.3, random_state=42)

df_train

#Retorna as estatísticas do dataframe
df_train.describe()

df_test

"""
Preparando o classificador kNN

k=7
Distância Euclidiana
Votação majoritária
"""
# estamos explicitamente selecionando os atributos de interesse
knn = KNeighborsClassifier(n_neighbors=7)
#Treinando o modelo com o dataframe de treinamento
#X_train é os dados de treino
X_train = df_train[['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']]
#y_train é os valores alvo dos dados de treino
y_train = df_train[['target']]
knn.fit(X_train, y_train.values.ravel())

#X_test é os dados para testar o treinamento
X_test = df_test[['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']]
#y_train é os valores alvo que o modelo deve atingir, que são usados para avaliação
y_test = df_test[['target']]

#Predição somente com os dados de teste, sem o target
y_pred = knn.predict(X_test)

#Avaliando em comparação do esperado
print(classification_report(y_test, y_pred))


# Classificador nulo que apenas chuta uma resposta considerando a distribuições das classes
#ESTRATÉGIA PIOR
dummy_clf = DummyClassifier(strategy='stratified')
dummy_clf.fit(X_train,y_train)
print(classification_report(y_test, dummy_clf.predict(X_test)))


#USANDO VOTO PONDERADO
knn_ponderado = KNeighborsClassifier(n_neighbors=7,weights='distance')
X_train = df_train[['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']]
y_train = df_train[['target']]
knn_ponderado.fit(X_train, y_train.values.ravel())

y_pred = knn_ponderado.predict(X_test)

print(classification_report(y_test, y_pred))

"""
Comparando com padronização de dados
"""


# preparando scaler para padronizar escala dos dados
# usaremos o conjunto de treino
min_max_scaler = preprocessing.MinMaxScaler()
min_max_scaler.fit(df_train[['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']])

# transformar os dados para a nova escala (treino e teste)
df_train_norm = min_max_scaler.transform(df_train[['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']])
df_test_norm = min_max_scaler.transform(df_test[['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']])

# preparando o knn com voto ponderado
knn_norm = KNeighborsClassifier(n_neighbors=7,weights='distance')

# conjunto de treino
X_train = df_train_norm
y_train = df_train[['target']]
knn_norm.fit(X_train, y_train.values.ravel())

# conjunto de teste
X_test = df_test_norm
y_test = df_test[['target']]

y_pred = knn_norm.predict(X_test)

# comparando desempenho
print(classification_report(y_test, y_pred))

pd.DataFrame(df_train_norm).describe()