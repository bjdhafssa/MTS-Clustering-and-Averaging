### Prerequisites

Ensure you have installed the necessary dependencies, as specified in the main project's requirements.txt file. Key dependencies include:

    numpy
    pandas
    tslearn
    aeon
    teka

### Usage

### To run the centroid calculation and comparison script, use the following command:

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
                    
### To visualize the centroids of a specific class from a dataset, you can use the following command:

        python plot_centroids.py

This script allows you to interactively choose a dataset, a class, and a dimension (or all dimensions) to visualize.

-- Example Interaction:

(WP2-MTS) boujida-h@po-boujida-2022:~/MTS-Clustering-and-Averaging/Averaging$ python plot_centroids.py 
Available datasets:
1. ArticularyWordRecognition
2. BasicMotions
3. Cricket
4. EigenWorms
5. Epilepsy
6. EthanolConcentration
7. FaceDetection
8. FingerMovements
9. HandMovementDirection
10. Handwriting
11. Heartbeat
12. Libras
13. LSST
14. MotorImagery
15. NATOPS
16. PenDigits
17. PEMS-SF
18. Phoneme
19. RacketSports
20. SelfRegulationSCP1
21. SelfRegulationSCP2
22. StandWalkJump
23. UWaveGestureLibrary
Enter the number corresponding to the dataset you want to use: 15

Available classes:
- 1.0
- 2.0
- 3.0
- 4.0
- 5.0
- 6.0
Enter the label of the class you want to analyze: 4.0

The dataset has 51 dimensions.
Enter the dimension you want to visualize (or type 'all' to visualize all dimensions): 0
Centroid comparison plot saved successfully.


-- The script will generate a plot like the one below, showing the original time series (in light gray) and the calculated centroids using different methods (in various colors).

![image](https://github.com/user-attachments/assets/d222c018-e2f0-44f4-8747-045c40b18942)





