import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tslearn.barycenters import dtw_barycenter_averaging, softdtw_barycenter
from aeon.clustering.averaging import elastic_barycenter_average
from teka import PyTEKA
from tslearn.metrics import dtw, soft_dtw
from aeon.distances import msm_distance, shape_dtw_distance
from aeon.distances import sbd_distance
from aeon.datasets import load_classification
from aeon.transformations.collection import Normalizer

#################################################################################################################

# created by peirre-françois marteau

# ckeck: https://github.com/pfmarteau/KDTW/blob/master/kdtw.py

TEKA = PyTEKA()

def get_kdtw_inertia(ts, ds, sigma, epsilon):
    inertia = 0
    for i in range(np.shape(ds)[0]):
        inertia = inertia + TEKA.kdtw(ts, ds[i], sigma, epsilon)
    return inertia

from numpy import isnan
def get_iTEKACentroid(ds, kmed, sigma, epsilon, npass=5):
    ii = 0
    inertiap = 0
    Cp = kmed
    Y=TEKA.iTEKA_stdev(kmed, ds, sigma, epsilon)
    TT=Y[1]
    X=np.array(list(Y[0]))
    T=X[:,len(X[0])-1]
    X=X[:,0:len(X[0])-1]
    C = TEKA.interpolate(X)
    dim=len(ds[0][0])
    #print('DIM=',dim)
    C0=C[:,0:dim]
    inertia = get_kdtw_inertia(C0, ds, sigma, epsilon)
    #print("inertia: ", inertia)
    while (not isnan(inertia)) and (ii < npass) and (inertia > inertiap):
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
        if not isnan(inertia):
           #print("inertia: ", inertia)
            ii = ii + 1
    return Cp# , Tp, inertiap, TTp

#####################################################################################################################

# created by peirre-françois marteau

# ckeck: https://github.com/pfmarteau/KDTW/blob/master/kdtw.py

def kdtw(A, B, sigma = 1, epsilon = 1e-3):
    d=np.shape(A)[1]
    Z=[np.zeros(d)]
    A = np.concatenate((Z,A), axis = 0)
    B = np.concatenate((Z,B), axis = 0)
    [la,d] = np.shape(A)
    [lb,d] = np.shape(B)

    DP = np.zeros((la,lb))
    DP1 = np.zeros((la,lb));
    DP2 = np.zeros(max(la,lb));
    l=min(la,lb);
    DP2[1] = 1.0;
    for i in range(1,l):
        DP2[i] = Dlpr(A[i],B[i], sigma, epsilon);
    if la<lb:
        for i in range(la,lb):
            DP2[i] = Dlpr(A[la-1],B[i], sigma, epsilon);
    elif lb<la:
        for i in range(lb,la):
            DP2[i] = Dlpr(A[i],B[lb-i], sigma, epsilon);

    DP[0,0] = 1;
    DP1[0,0] = 1;
    n = len(A);
    m = len(B);

    for i in range(1,n):
        DP[i,1] = DP[i-1,1]*Dlpr(A[i], B[2], sigma, epsilon);
        DP1[i,1] = DP1[i-1,1]*DP2[i];

    for j in range(1,m):
        DP[1,j] = DP[1,j-1]*Dlpr(A[2], B[j], sigma, epsilon);
        DP1[1,j] = DP1[1,j-1]*DP2[j];

    for i in range(1,n):
        for j in range(1,m): 
            lcost = Dlpr(A[i], B[j], sigma, epsilon);
            DP[i,j] = (DP[i-1,j] + DP[i,j-1] + DP[i-1,j-1])*lcost;
            if i == j:
                DP1[i,j] = DP1[i-1,j-1]*lcost + DP1[i-1,j]*DP2[i] + DP1[i,j-1]*DP2[j]
            else:
                DP1[i,j] = DP1[i-1,j]*DP2[i] + DP1[i,j-1]*DP2[j];
    DP = DP + DP1;
    return DP[n-1,m-1]

def Dlpr(a, b, sigma = 1, epsilon = 1e-3):
    return (np.exp(-np.sum((a - b)**2) / sigma) + epsilon)/(3*(1+epsilon))


#######################################################################################################

def calculate_centroids_and_compare(X, X_ts, y, class_label):
    # Select all samples for the given class
    class_samples_aeon = X[y == class_label]
    class_samples_tslearn = X_ts[y == class_label]
    
    # Calculate centroids using different methods
    centroids = {}
    dba_c = dtw_barycenter_averaging(class_samples_tslearn, max_iter=10, tol=1e-5)
    centroids['DBA'] = dba_c.transpose()
    softdba_c = softdtw_barycenter(class_samples_tslearn, gamma=1., max_iter=10, tol=1e-5)
    centroids['Soft-DBA'] = softdba_c.transpose()
    centroids['Shape-DBA'] = elastic_barycenter_average(class_samples_aeon, distance="shape_dtw", 
                                                                                max_iters=10, tol=1e-5)
    
    # Using TEKA
    sigma = 1
    epsilon = 1e-3
    initial_centroid = class_samples_tslearn[0]  # First time series as initial centroid
    dim = len(class_samples_tslearn[0][0])
    
    
    #centroid_teka, Tstd, inertia, TTp 
    centroid_teka = get_iTEKACentroid(class_samples_tslearn, initial_centroid, sigma, epsilon, npass=10)
    centroid_teka = centroid_teka[:,0:dim]  
    centroids['TEKA'] = centroid_teka.transpose()
    
    # Compare methods by calculating average distance to other time series
    methods = ['DBA', 'Shape-DBA', 'Soft-DBA', 'TEKA']
    distance_metrics = ['DTW', 'ShapeDTW', 'SoftDTW', 'KDTW', 'MSM', 'SBD']
    results_df = pd.DataFrame(index=methods, columns=distance_metrics)
    
    
    for method in methods:
        centroid = centroids[method]
        distances = {metric: [] for metric in distance_metrics}

        for ts in class_samples_aeon:
            centroid_tslearn = centroid.T 
            ts_tslearn = ts.T 
            
            # Use transposed data for tslearn distances
            distances['DTW'].append(dtw(centroid_tslearn, ts_tslearn))
            distances['SoftDTW'].append(soft_dtw(centroid_tslearn, ts_tslearn, gamma=1.0))
            distances['KDTW'].append(kdtw(centroid_tslearn, ts_tslearn))


            # Use original data for other distances
            distances['MSM'].append(msm_distance(centroid, ts))
            distances['SBD'].append(sbd_distance(centroid, ts))
            distances['ShapeDTW'].append(shape_dtw_distance(centroid, ts))

        for metric in distance_metrics:
            results_df.loc[method, metric] = np.mean(distances[metric])

    return class_samples_aeon, class_samples_tslearn, centroids, results_df

#########################################################################################################################

def process_datasets(datasets):
    all_results = []
    normalizer = Normalizer()
    
    for dataset in datasets:
        print(f"Processing dataset: {dataset}")
        X, y = load_classification(name=dataset)
        Xn = normalizer.fit_transform(X)
        X_ts = Xn.transpose(0, 2, 1)
        
        for class_label in np.unique(y):
            print(f"  Processing class: {class_label}")
            _, _, _, comparison = calculate_centroids_and_compare(Xn, X_ts, y, class_label)
            
            for method, distances in comparison.iterrows():
                result = {
                    'Dataset': dataset,
                    'Class': class_label,
                    'Method': method,
                    **distances.to_dict()
                }
                all_results.append(result)
    
    return pd.DataFrame(all_results)

##########################################################################################################################

if __name__ == "__main__":
    datasets = [
        'ArticularyWordRecognition', 'BasicMotions', 'Cricket', 'EigenWorms', 
        'Epilepsy', 'EthanolConcentration', 'FaceDetection', 'FingerMovements', 
        'HandMovementDirection', 'Handwriting', 'Heartbeat', 'Libras', 'LSST', 
        'MotorImagery', 'NATOPS', 'PenDigits', 'PEMS-SF', 'Phoneme', 
        'RacketSports', 'SelfRegulationSCP1', 'SelfRegulationSCP2', 
        'StandWalkJump', 'UWaveGestureLibrary'
    ]
    
    df_results = process_datasets(datasets)
    print(df_results)
