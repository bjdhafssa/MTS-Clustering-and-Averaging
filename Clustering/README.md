
This project uses both Aeon and Tslearn libraries for a comparative analysis of clustering methods on multivariate time series (MTS) datasets. You can run the benchmark script with either library to evaluate and compare their respective methods. 

### Why Use Both Libraries? 

Aeon: Features TimeSeriesKMedoids clustering and supports distance metrics like Shape-DTW and MSM, which are not available in Tslearn.
Tslearn: Offers KShape clustering and supports Soft-DTW, a distance metric not found in Aeon.

### Example Usage : 

Running with Aeon: 

       $ cd aeon_implementations
       $ python benchmark.py BasicMotions

Running with Tslearn:

       
       $ cd tslearn_implementations
       $ python benchmark.py <dataset_name>


###### Explanation of Arguments

<dataset_name>: Replace BasicMotions with the name of the dataset you want to process.

## Results :

**Evaluation Metrics**

Below are the [CD-diagrams](https://github.com/hfawaz/cd-diagram) evaluating the clustering results using the Silhouette Score and Adjusted Rand Index (ARI) Score:

* Silhouette Score CD-Diagram:

![Silhouette Score CD-Diagram](figures/cd-diagram-silhouette.png)


* ARI Score CD-Diagram:

![ARI Score CD-Diagram](figures/cd-diagram-ari.png)


**Computational Runtime**

The following CD-diagram compares the computational runtime (in seconds) for one iteration across nine clustering schemes:

![Computational Runtime CD-Diagram](figures/cd-diagram-runtime.png)


###### Clustering Schemes


'Kmedoids-dtw' : 'Kmed-dtw'
'Kmedoids-msm' : 'Kmed-msm'
'Kmedoids-shapedtw' : 'Kmed-shapedtw'
'Kmedoids-kdtw' : 'kmed-kdtw'
'Kmeans-teka' : 'KM-teka'
'Kmeans-msm' : 'KM-msm'
'Kmeans-shapedtw' : 'KM-shapedtw'
'Kmeans-dtw' : 'KM-dtw'
'Kmeans-softdtw' : 'KM-softdtw'
'KernelKmeans-gak' : 'KKMe-gak'
'Kshape' : 'Kshape'
