#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Time series kernel kmeans
mplementation according (more or less) to AEON toolbox API, specifically:
https://github.com/aeon-toolkit/aeon/blob/main/aeon/clustering/_k_means.py

Created on Sat Feb 15 09:49:08 2025

@author: Pierre-François Marteau

References:
[i] P. F. Marteau and S. Gibet, "On Recursive Edit Distance Kernels With Application to Time Series Classification," 
in IEEE Transactions on Neural Networks and Learning Systems, vol. 26, no. 6, pp. 1121-1133, June 2015. 
doi: 10.1109/TNNLS.2014.2333876
[ii] Pierre-François Marteau, Times series averaging and denoising from a probabilistic perspective on time-elastic kernels, 
International Journal of Applied Mathematics and Computer Science, De Gruyter, 2019, 29 (2), pp.375-392. ⟨hal-01401072v4⟩
"""

from typing import Optional, Union
import math, random
import numpy as np
from numpy.random import RandomState
from sklearn.cluster import SpectralClustering

from aeon.clustering.base import BaseClusterer

import KDTWKMedoids

from teka import PyTEKA
TEKA = PyTEKA()

def get_kdtw_inertia(ts, ds, sigma, epsilon):
    inertia = 0
    for i in range(np.shape(ds)[0]):
        inertia = inertia + TEKA.kdtw(ts, ds[i], sigma, epsilon)
    return inertia

def get_iTEKACentroid(ds, kmed, sigma, epsilon, npass=5):
    ii = 0
    inertiap = 0
    Cp = kmed
    Y=TEKA.iTEKA_stdev(kmed, ds, sigma, epsilon)
    TT=Y[1]
    TTp=TT
    X=np.array(list(Y[0]))
    T=X[:,len(X[0])-1]
    Tp=T
    X=X[:,0:len(X[0])-1]
    C = TEKA.interpolate(X)
    dim=len(ds[0][0])
    C0=C[:,0:dim]
    inertia = get_kdtw_inertia(C0, ds, sigma, epsilon)
    #print("inertia: ", inertia)
    while (not math.isnan(inertia)) and (ii < npass) and (inertia > inertiap):
        inertiap = inertia
        Cp = C
        Tp=T
        TTp=TT
        Y=TEKA.iTEKA_stdev(Cp[:,0:dim], ds, sigma, epsilon)
        TT=Y[1]
        X=np.array(list(Y[0]))
        T=X[:,len(X[0])-1]
        X=X[:,0:len(X[0])-1]
        C = TEKA.interpolate(X)
        C0=C[:,0:dim]
        inertia = get_kdtw_inertia(C0, ds, sigma, epsilon)
        #if not math.isnan(inertia):
        #   print("inertia: ", inertia)
        ii = ii + 1
    return Cp, Tp, inertiap, TTp

# Partition X according to label y. 
# Returns a dictionnary key = l, a label : value = subset of instances of X with label l.
def split_on_y(X,y):
   out = dict()
   for i in range(len(y)):
      if y[i] not in out.keys():
         out[y[i]] = [X[i]]
      else:
         l = out[y[i]]
         l.append(X[i])
         out[y[i]] = l  
   return out


def calculate_kdtw_similarities(x: np.ndarray, c: np.ndarray, sigma: float = 1., epsilon: float = 1e-300) -> np.ndarray:
    """Calculates KDTW similarities between elements of 2 arrays"""
    out = np.zeros((len(x),len(c)))
    lnc = []
    for j in range(len(c)):
        u = TEKA.kdtw(c[j], c[j], sigma, epsilon) + 1e-300
        if math.isnan(u):
           u = 1e-300
        lnc.append(u)
    for i in range(len(x)):
        for j in range(len(c)):
            u = TEKA.kdtw(x[i], c[j], sigma, epsilon)/(lnc[j] + 1e-300)
            if math.isnan(u):
               u = 1e-300
            out[i,j] = u
    return out

def calculate_kdtw_pairwise_similarities(x: np.ndarray, sigma: float = 1., epsilon: float = 1e-300) -> np.ndarray:
    """Calculates KDTW similarities between elements of 2 arrays"""
    out = np.zeros((len(x),len(x)))
    for i in range(len(x)):
        for j in range(i,len(x)):
            u = TEKA.kdtw(x[i], x[j], sigma, epsilon) + 1e-300
            if math.isnan(u):
               u = 1e-300

            out[i,j] = u
            out[j,i] = u
    return out

def center_equals(x: np.ndarray, y: np.ndarray, tol: float=1e-10) -> bool:
    """Compares 2 centroid matrices"""
    if len(x) != len(y):
        return False
    distances = np.sqrt(np.sum((x - y) ** 2, 1))
    if np.any(distances > tol):  # Accuracy to be tested
        return False
    return True

def assign_labels_teka(data: np.ndarray, centroids: np.ndarray, sigma: float = 1., epsilon: float = 1e-300) -> np.ndarray:
    """
    Calculates similarities between data points and centroids
    and finds the most similar for each one.
    """
    similarities = calculate_kdtw_similarities(data, centroids, sigma, epsilon)
    most_similar = np.argmax(similarities, axis=1)
    return most_similar

def spectral_clustering_initialization(data: np.ndarray, nclust: int, sigma: float, epsilon: float) -> np.ndarray:
    lC = []
    dim = np.shape(data)[2]
    A = calculate_kdtw_pairwise_similarities(data, sigma, epsilon)
    sc = SpectralClustering(nclust, affinity='precomputed', n_init=400, assign_labels='discretize', eigen_tol=1e-4)
    sc.fit_predict(A)
    y0 = sc.labels_               
    #print('spectral clustering done!',flush=True)
    dsC = split_on_y(data,y0)
    for k in dsC.keys():
        X0 = dsC[k]
        C_TEKA, Tstd_c, C_inertia, TTp_c = get_iTEKACentroid(X0, X0[0], sigma, epsilon, npass=10)
        lC.append(C_TEKA[:,0:dim])
    return np.array(lC)

def kdtw_kmedoids_initialization(data: np.ndarray, nclust: int, sigma: float, epsilon: float) -> np.ndarray:
    kdtw_kmd = KDTWKMedoids.KDTWKMedoids(n_clusters=nclust, method="pam", 
    distance_params={"sigma": sigma, "epsilon":epsilon}, n_init=20, max_iter=200)
    kdtw_kmd.fit(data)
    return kdtw_kmd.cluster_centers_

def random_initialization(data: np.ndarray, k: int) -> np.ndarray:
    """Choose random points"""
    indices = np.random.choice(data.shape[0], k, replace=False)
    return data[indices]

def plusplus_initialization(data: np.ndarray, k: int) -> np.ndarray:
    """
    Kmeans++ initialization
    First point is random
    Then each new point is chosen based on distance to already chosen centroids
    """
    centroids = data[np.random.randint(0, data.shape[0])]
    centroids = np.expand_dims(centroids, axis=0)
    #expanded_data = np.expand_dims(data, axis=1)

    for _ in range(1, k):
        similarities = calculate_kdtw_similarities(data, centroids)
        similarities = np.max(similarities, axis=1)
        probabilities = -2*np.log(similarities+1e-300)
        probabilities = probabilities / probabilities.sum()

        new_index = np.random.choice(data.shape[0], p=probabilities)
        centroids = np.append(centroids, data[new_index : new_index + 1, :], axis=0)

    return centroids


def cluster_teka(data: np.ndarray, centroids: np.ndarray, sigma: float=1., epsilon: float=1e-300, max_iters: int=100) -> np.ndarray:
    """Performs the kmeans clustering iteration"""
    current_centroids = centroids.copy()
    for i in range(max_iters):
        new_centroids, inertia = _cluster_teka(data, current_centroids, sigma, epsilon)
        if center_equals(current_centroids, new_centroids):
            break
        current_centroids = new_centroids

    return current_centroids, i + 1, inertia

def _cluster_teka(data: np.ndarray, current_centroids: np.ndarray, sigma: float=1., epsilon: float=1e-300) -> np.ndarray:
    """Performs single iteration of kmeans clustering"""
    k = len(current_centroids)

    # Find assignement of each point
    assignement = assign_labels_teka(data, current_centroids, sigma, epsilon)

    # Update centroids by taking a mean of every point assigned to it
    new_centroids = current_centroids.copy()
    inertia = 0
    for cid in range(k):
        assigned = data[assignement == cid, :]
        # Update mean if points assigned
        if assigned.any():
            """ TEKA averaging """
            dim = len(new_centroids[cid, 0])
            #C_TEKA, Tstd_c, inertia, TTp_c = get_iTEKACentroid(assigned, current_centroids[cid], sigma, epsilon, npass=5)
            C_TEKA, Tstd_c, C_inertia, TTp_c = get_iTEKACentroid(assigned, assigned[0], sigma, epsilon, npass=10)
            new_centroids[cid, :] = C_TEKA[:,0:dim]
            inertia += C_inertia
    return new_centroids, -np.log(inertia+1e-300)

class TekaKernelKMeans():
    """TEKA Kernel K Means [1]_: close to ``tslearn`` implementation.

    Parameters
    ----------
    n_clusters: int, default=8
        The number of clusters to form as well as the number of
        centroids to generate.
    kernel : string, or callable (default: "teka")
        The kernel should be "teka", in which case the Kernelized DTW (KDTW) [2]_ is used in conjunction with 
        the Time Elastic Kernel Averaging (TEKA) procedure. Otherwise, nothing is done.
    n_init: int, default=10
        Number of times the k-means algorithm will be run with different
        centroid seeds. The final result will be the best output of ``n_init``
        consecutive runs in terms of inertia.
    kernel_params : dict (default: {'sigma': 1.0, 'epsilon':1e-300})
        Kernel parameters to be passed to the kernel function.
        For KDTW Kernel, the only parameters of interest is ``sigma`` and ``epsilon``.
    max_iter: int, default=100
        Maximum number of iterations of the k-means algorithm for a single
        run.
    tol: float, default=1e-4
        Relative tolerance with regards to Frobenius norm of the difference
        in the cluster centers of two consecutive iterations to declare
        convergence.
    verbose: bool, default=False
        Verbosity mode.
    n_jobs : int or None, default=None
        Not used
    random_state: int or np.random.RandomState instance or None, default=None
        Determines random number generation for centroid initialization.

    Attributes
    ----------
    labels_: np.ndarray (1d array of shape (n_case,))
        Labels that is the index each time series belongs to.
    inertia_: float
        Sum of squared distances of samples to their closest cluster center, weighted by
        the sample weights if provided.
    n_iter_: int
        Number of iterations run.

    References
    ----------
    .. [1] 
           Dhillon, Yuqiang Guan, Brian Kulis. KDD 2004.
    .. [2] Fast Global Alignment Kernels. Marco Cuturi. ICML 2011.

    Examples
    --------
    >>> from aeon.clustering import TimeSeriesKernelKMeans
    >>> from aeon.datasets import load_basic_motions
    >>> # Load data
    >>> X_train, y_train = load_basic_motions(split="TRAIN")[0:10]
    >>> X_test, y_test = load_basic_motions(split="TEST")[0:10]
    >>> # Example of KernelKMeans Clustering
    >>> kkm = TimeSeriesKernelKMeans(n_clusters=3, kernel='rbf')  # doctest: +SKIP
    >>> kkm.fit(X_train)  # doctest: +SKIP
    TimeSeriesKernelKMeans(kernel='rbf', n_clusters=3)
    >>> preds = kkm.predict(X_test)  # doctest: +SKIP
    """

    _tags = {
        "capability:multivariate": True,
        "capability:multithreading": True,
        #"python_dependencies": "tslearn",
    }

    def __init__(
        self,
        n_clusters: int = 8,
        kernel: str = "teka",
        n_init: int = 10,
        init_type: str = "random",
        initial_centroids: np.ndarray = None,
        max_iter: int = 300,
        tol: float = 1e-4,
        kernel_params: dict = {"sigma": 1.0, "epsilon":1e-300},
        verbose: bool = False,
        n_jobs: int = None,
        random_state: Optional[Union[int, RandomState]] = None,
    ):
        self.kernel = kernel
        self.n_init = n_init
        self.init_type = init_type
        self.initial_centroids = initial_centroids
        self.max_iter = max_iter
        self.tol = tol
        self.kernel_params = kernel_params
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.n_clusters = n_clusters

        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = 0
     
        self._tslearn_kernel_k_means = None

        if self.init_type == "kmedoids":
            self.n_init = 1

        super().__init__()

    def initialize_centroids(self, data: np.ndarray, k: int, init_type: str, initial_centroids: np.ndarray=None) -> np.ndarray:
        """
        Initialize centroids
        """
        if init_type == "random":
            return random_initialization(data, k)
        elif init_type == "plusplus":
            return plusplus_initialization(data, k)
        elif init_type == "spectral_clustering":
            return spectral_clustering_initialization(data, k, self.kernel_params["sigma"], self.kernel_params["epsilon"])
        elif init_type == "kmedoids":
            return kdtw_kmedoids_initialization(data, k, self.kernel_params["sigma"], self.kernel_params["epsilon"])
        else:
            raise ValueError(f"No initialization for init_type={init_type!r}")



    def fit(self, X, y=None):
        """Fit time series clusterer to training data.

        Parameters
        ----------
        X: np.ndarray, of shape (n_cases, n_channels, n_timepoints) or
                (n_cases, n_timepoints)
            A collection of time series instances.
        y: ignored, exists for API consistency reasons.

        Returns
        -------
        self:
            Fitted estimator.
        """
        self.inertia = 1e300
        _X = X.copy()

        if self.n_clusters > X.shape[0]:
            raise ValueError("parameter n_clusters can't be bigger than sample size")
        for run in range(self.n_init):
            random.shuffle(list(_X))
            centroids = self.initialize_centroids(_X, self.n_clusters, self.init_type)
            centroids, n_iter, inertia = cluster_teka(X, centroids, self.kernel_params["sigma"], 
                                                            self.kernel_params["epsilon"], self.max_iter)  
            if inertia<self.inertia:
                self.cluster_centers_ = centroids
                self.n_iter_ = n_iter
                self.inertia = inertia
                if True:#self.verbose:
                    print("INERTIA", inertia, run)


    def predict(self, X, y=None) -> np.ndarray:
        """Predict the closest cluster each sample in X belongs to.

        Parameters
        ----------
        X: np.ndarray, of shape (n_cases, n_channels, n_timepoints) or
                (n_cases, n_timepoints)
            A collection of time series instances.
        y: ignored, exists for API consistency reasons.

        Returns
        -------
        np.ndarray (1d array of shape (n_cases,))
            Index of the cluster each time series in X belongs to.
        """

        return assign_labels_teka(X, self.cluster_centers_, 
                                      self.kernel_params["sigma"], 
                                      self.kernel_params["epsilon"])
         

    @classmethod
    def _get_test_params(cls, parameter_set="default") -> dict:
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.


        Returns
        -------
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
        """
        return {
            "n_clusters": 2,
            "kernel": "teka",
            "n_init": 1,
            "max_iter": 1,
            "tol": 0.0001,
        }

