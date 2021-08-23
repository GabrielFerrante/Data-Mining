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