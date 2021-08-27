import numpy as np
import pandas as pd
import scipy
from scipy.spatial import distance_matrix 
from scipy.cluster import hierarchy 
import seaborn as sns
from scipy.spatial.distance import pdist
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn import datasets
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.cluster import KMeans



"""
AVALIANDO UMA CLASSIFICAÇÃO
"""
"""
ACURÁCIA

"""

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

y_true = [0, 0, 0, 0, 1, 1, 1]
y_pred = [0, 0, 0, 0, 1, 1, 0]

#Calcula a acurácia com o que a saida verdadeira, com a predição
acc = accuracy_score(y_true, y_pred)
print('accuracy = ',acc)

#Atribuindo os verdadeiros negativos, falsos positivos, falsos negativos,
#e verdadeiros positivos
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

print('TP = ',tp)
print('FP = ',fp)
print('TN = ',tn)
print('FN = ',fn)

"""
PRECISÃO
Relação entre predições corretas e o número de vezes que a
classe foi predita
"""

from sklearn.metrics import precision_score

y_true = [1, 0, 1, 0, 0, 1, 1]
y_pred = [0, 0, 1, 0, 1, 0, 1]

p = precision_score(y_true, y_pred)
print('precision = ',p)

tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
print('TP = ',tp)
print('FP = ',fp)
print('TN = ',tn)
print('FN = ',fn)

"""
Outro teste
"""

from sklearn.metrics import precision_score
y_true = [1, 1, 1, 1, 1, 1, 1]
y_pred = [0, 0, 0, 0, 0, 0, 1] 

p = precision_score(y_true, y_pred)
print('precision = ',p)

tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
print('TP = ',tp)
print('FP = ',fp)
print('TN = ',tn)
print('FN = ',fn)

"""
RECALL
"""

from sklearn.metrics import recall_score
y_true = [0, 1, 0, 0, 1, 1, 1]
y_pred = [0, 0, 1, 0, 1, 1, 1]

r = recall_score(y_true, y_pred)

print('recall = ',r)

tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
print('TP = ',tp)
print('FP = ',fp)
print('TN = ',tn)
print('FN = ',fn)

"""
Outro teste
"""
from sklearn.metrics import recall_score
y_true = [0, 0, 0, 0, 1, 1, 1]
y_pred = [1, 1, 1, 1, 1, 1, 1]

r = recall_score(y_true, y_pred)

print('recall = ',r)

tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
print('TP = ',tp)
print('FP = ',fp)
print('TN = ',tn)
print('FN = ',fn)

#Mostrando a matriz de confusão
confusion_matrix(y_true, y_pred)


"""
F1-Score, Média harmônica entre precision e recall
"""
from sklearn.metrics import f1_score
y_true = [0, 0, 0, 0, 1, 1, 1]
y_pred = [0, 0, 0, 0, 1, 1, 1] 

f1_score(y_true, y_pred) 

"""
Outro teste
"""
from sklearn.metrics import f1_score
y_true = [0, 0, 0, 0, 1, 1, 1]
y_pred = [1, 1, 1, 1, 1, 1, 1]

f1_score(y_true, y_pred)

"""
F1-Score por classe e dados desbalanceados
"""

from sklearn.metrics import f1_score
y_true = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
y_pred = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

#Argumento 'micro': Calcule as métricas globalmente contando
# o total de verdadeiros positivos, falsos negativos e falsos positivos.
f1_score(y_true, y_pred, average='micro')

from sklearn.metrics import f1_score
y_true = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
y_pred = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] 
#Argumento 'macro': Calcule as métricas para cada rótulo e 
#encontre sua média não ponderada. Isso não leva em consideração
# o desequilíbrio do rótulo.
f1_score(y_true, y_pred, average='macro')


"""
ROC e AUC
"""

import numpy as np
from sklearn.metrics import roc_auc_score
y_true = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
y_pred = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1]
roc_auc_score(y_true, y_pred)

"""
Outro exemplo

"""
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.metrics import roc_auc_score

# Import some data to play with
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Binarize the output
y = label_binarize(y, classes=[0, 1, 2])
n_classes = y.shape[1]

# Add noisy features to make the problem harder
random_state = np.random.RandomState(0)
n_samples, n_features = X.shape
X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

# shuffle and split training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5,
                                                    random_state=0)

# Learn to predict each class against the other
classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,
                                 random_state=random_state))
y_score = classifier.fit(X_train, y_train).decision_function(X_test)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


classe = 2
plt.figure()
lw = 2
plt.plot(fpr[classe], tpr[classe], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[classe])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

"""
Outros algoritmos de classificação
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

h = .02  # step size in the mesh

names = ["Nearest Neighbors", "SVM", "Decision Tree", "MLP"]

classifiers = [
    KNeighborsClassifier(),
    SVC(),
    DecisionTreeClassifier(),
    MLPClassifier()]

X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                           random_state=1, n_clusters_per_class=1)
rng = np.random.RandomState(2)
X += 2 * rng.uniform(size=X.shape)
linearly_separable = (X, y)

datasets = [make_moons(noise=0.1, random_state=0),
            make_moons(noise=0.2, random_state=0),
            make_moons(noise=0.5, random_state=0)]

figure = plt.figure(figsize=(27, 12))
i = 1

L_dataset = []
L_classifier = []
L_accuracy = []
L_precision = []
L_recall = []
L_f1 = []
L_f1_macro = []
L_roc_auc = []

# iterate over datasets
for ds_cnt, ds in enumerate(datasets):
    # preprocess dataset, split into training and test part
    X, y = ds
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.4, random_state=42)

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # just plot the dataset first
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
    if ds_cnt == 0:
        ax.set_title("Input data")
    # Plot the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
               edgecolors='k')
    # Plot the testing points
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6,
               edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    i += 1

    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        y_pred = clf.predict(X_test)

        L_dataset.append(ds_cnt)
        L_classifier.append(name)
        L_accuracy.append(accuracy_score(y_test,y_pred))
        L_precision.append(precision_score(y_test,y_pred))
        L_recall.append(recall_score(y_test,y_pred))
        L_f1.append(f1_score(y_test,y_pred,average="micro"))
        L_f1_macro.append(f1_score(y_test,y_pred,average="macro"))

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        else:
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

        # Plot the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
                   edgecolors='k')
        # Plot the testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
                   edgecolors='k', alpha=0.6)

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        if ds_cnt == 0:
            ax.set_title(name)
        ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                size=15, horizontalalignment='right')
        i += 1

plt.tight_layout()
plt.show()


import pandas as pd

df = pd.DataFrame()
df['dataset'] = L_dataset
df['classifier'] = L_classifier
df['accuracy'] = L_accuracy
df['precision'] = L_precision
df['recall'] = L_recall
df['f1'] = L_f1
df['f1-macro'] = L_f1_macro

df


"""
VALIDAÇÃO DE AGRUPAMENTOS
"""

"""
Validação por Inspeção Visual, indice de validade interna
Funciona bem quando os clusters são bem definidos
"""

from sklearn.datasets import make_blobs

dados1, y = make_blobs(n_samples=100,
                  n_features=2,
                  centers=3,
                  shuffle=True,
                  random_state=1)  # For reproducibility

dados1 = pd.DataFrame(dados1)
dados1.columns = ['x','y']
dados1.head(10)

dados1.plot(kind='scatter',x='x',y='y')

def calcular_matriz_dissimilaridade(dados):
    #np.zeros retorna uma matriz de zeros
  M = np.zeros((len(dados),len(dados)))
  for i,row_i in dados.iterrows(): # para cada objeto i
      features_i = np.array(row_i) # atributos do objeto i
      for j,row_j in dados.iterrows(): # para cada objeto j
          features_j = np.array(row_j) # atributos do objeto j
          # calcula distância euclidiana
          euc = scipy.spatial.distance.euclidean(features_i, features_j)
          # armazena na posição M[i,j]
          M[i,j] = euc
  return M


M = calcular_matriz_dissimilaridade(dados1)
#Apresentando 
sns.clustermap(M)  # average-link com distância euclidiana

"""
Outro Exemplo, com clusters não bem definidos
"""
dados_random = pd.DataFrame(np.random.rand(300,2))
dados_random

dados_random.plot(kind='scatter',x=0,y=1)

M = calcular_matriz_dissimilaridade(dados_random)
sns.clustermap(M)  # average-link com distância euclidiana

"""
Índice de Validade Relativa (Silhueta)

"""

dados1.plot(kind='scatter',x='x',y='y')
kmeans = KMeans(n_clusters=3,n_init=10,init='random',max_iter=300)
kmeans.fit(dados1) # agrupando

dados1['cluster'] = kmeans.labels_

"""
Calculando o índice de silhueta para cada objeto"""

dados1['silhueta'] = silhouette_samples(dados1, kmeans.labels_)
dados1

# calculando o valor de silhueta para todo o agrupamento
dados1.silhueta.mean()

"""Identificando k conforme silhueta"""

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

print(__doc__)

# Generating the sample data from make_blobs
# This particular setting has one distinct cluster and 3 clusters placed close
# together.
X, y = make_blobs(n_samples=500,
                  n_features=2,
                  centers=4,
                  cluster_std=1,
                  center_box=(-10.0, 10.0),
                  shuffle=True,
                  random_state=1)  # For reproducibility

range_n_clusters = [2, 3, 4, 5, 6]

for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(X)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                c=colors, edgecolor='k')

    # Labeling the clusters
    centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                c="white", alpha=1, s=200, edgecolor='k')

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                    s=50, edgecolor='k')

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')

plt.show()

"""Índice de Validade Externa
Vamos usar o dataset Iris, que já possui informação de rótulo para cada objeto (tipo de flor).

Esse dataset já foi utilizado em outras partes do curso de ciência de dados!"""

from sklearn import datasets

iris = datasets.load_iris()
X = iris.data[:, :2]  # vamos usar apenas dois atributos
y = iris.target

df_iris = pd.DataFrame(X)
df_iris['label']=y
df_iris

df_iris.plot(kind='scatter',x=0,y=1)


kmeans = KMeans(n_clusters=3,n_init=10,init='random',max_iter=300)
kmeans.fit(X) # agrupando

C = kmeans.labels_  # resultado do agrupamento
R = y  # organização de referência

RAND = adjusted_rand_score(C,R)
print('RAND=',RAND)

