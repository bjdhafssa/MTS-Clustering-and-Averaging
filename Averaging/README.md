#### Prerequisites

Ensure you have installed the necessary dependencies, as specified in the main project's requirements.txt file. Key dependencies include:

    numpy
    pandas
    tslearn
    aeon
    teka

Usage
To run the centroid calculation and comparison script, use the following command:

        python barycenter_averaging_comparison.py

This script will:

 1- Load several datasets.
 
 2- Calculate centroids for each class in each dataset using:
 
 DTW Barycenter Averaging (DBA)
 
 Soft-DTW Barycenter
 
 Shape-DBA (Elastic Barycenter Average with Shape-DTW)
 
 Time Elastic Kernel Alignment (TEKA)
         
 3- Evaluate the quality of each centroid by computing its average distance to all time series within its respective class, using different distance metrics.
 
 4- Display the results as pandas DataFrames. The values in each column represent the average distance between the centroid obtained and all time series in that class for a specific distance metric.
                    
