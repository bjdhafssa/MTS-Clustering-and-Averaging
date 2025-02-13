
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
       $ python benchmark.py BasicMotions


###### Explanation of Arguments

<dataset_name>: Replace BasicMotions with the name of the dataset you want to process.

## Results :

**Evaluation Metrics**

Below are the [CD-diagrams](https://github.com/hfawaz/cd-diagram) evaluating the clustering results using the Silhouette Score and Adjusted Rand Index (ARI) Score:

* Silhouette Score CD-Diagram:

<img src="https://github.com/user-attachments/assets/8c7d9dce-8fcc-4d19-9abf-a22b2b0051c8" alt="Silhouette Score CD-Diagram" width="700">

* ARI Score CD-Diagram:

<img src="https://github.com/user-attachments/assets/4ed1206d-8ff2-4733-a4c0-30e5b87f20cc" alt="ARI Score CD-Diagram" width="700">



**Computational Runtime**

The following CD-diagram compares the computational runtime (in seconds) for one iteration across nine clustering schemes:

<img src="https://github.com/user-attachments/assets/c02ea9b4-2780-4100-a6bf-24b5ab91d3ff" alt="Computational Runtime" width="700">


###### Clustering Schemes


'Kmedoids-dtw' : 'Kmed-dtw'
'Kmedoids-msm' : 'Kmed-msm'
'Kmedoids-shapedtw' : 'Kmed-shapedtw'
'Kmeans-msm' : 'KM-msm'
'Kmeans-shapedtw' : 'KM-shapedtw'
'Kmeans-dtw' : 'KM-dtw'
'Kmeans-softdtw' : 'KM-softdtw'
'KernelKmeans-gak' : 'KKMe-gak'
'Kshape' : 'Kshape'
