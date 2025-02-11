
This project uses both Aeon and Tslearn libraries for a comparative analysis of clustering methods on multivariate time series (MTS) datasets. You can run the benchmark script with either library to evaluate and compare their respective methods. 

### Why Use Both Libraries? 

Aeon: Features TimeSeriesKMedoids clustering and supports distance metrics like Shape-DTW and MSM, which are not available in Tslearn.
Tslearn: Offers KShape clustering and supports Soft-DTW, a distance metric not found in Aeon.

### Example Usage : 

Running with Aeon: 

       python main_benchmark.py BasicMotions --library aeon


Running with Tslearn:

       python main_benchmark.py BasicMotions --library tslearn


###### Explanation of Arguments

    <dataset_name>:
        Replace BasicMotions with the name of the dataset you want to process.
        The dataset must be one of the predefined datasets (see Available Datasets).
    --library:
        Specify which library to use for clustering. Options are:
            aeon: Use clustering methods implemented with the Aeon library.
            tslearn: Use clustering methods implemented with the Tslearn library.
