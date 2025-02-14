# clustering_methods.py 

from tslearn.clustering import TimeSeriesKMeans, KShape, KernelKMeans
from data_loader import load_dataset, working_datasets
from preprocessor import preprocess_data
import numpy as np

#######################################################################

def kmeans_dtw(X, n_clusters):
    
    model = TimeSeriesKMeans(n_clusters=n_clusters,
                             max_iter = 100, 
                             n_init = 10,
                             metric="dtw",
                             max_iter_barycenter=10,
                             init='k-means++')
    return model.fit_predict(X)

#########################################################################

def kmeans_soft_dtw(X, n_clusters):
    
    model = TimeSeriesKMeans(n_clusters=n_clusters,
                             max_iter = 100, 
                             n_init = 10,
                             metric="softdtw",
                             max_iter_barycenter=10,
                             init='k-means++'
                             )
    return model.fit_predict(X)

###########################################################################

def kernel_kmeans_gak(X, n_clusters):
    
    model = KernelKMeans(n_clusters=n_clusters, kernel="gak", max_iter=100, n_init=10)
    
    return model.fit_predict(X)


############################################################################

def kshape(X, n_clusters):

    model = KShape(n_clusters=n_clusters, max_iter=100, init="random")
    
    return model.fit_predict(X)


