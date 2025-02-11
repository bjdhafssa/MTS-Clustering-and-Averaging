# MTS Clustering and Averaging

This repository contains the code that supports the results presented in our paper, 'Recent Advances in Time Series Averaging: Approaches, Comparative Analysis, and Future Directions'. Submitted at The 34th International Joint Conference on Artificial Intelligence (IJCAI-25).

The purpose of this repository is to provide researchers with the means to reproduce our experiments.

Here is an overview of the main aspects covered:

## 1- Assessment of Clustering Performance

* K-means-based clustering: Comparison of different dissimilarity measures: Elastic Measures, Sliding Measure, Kernel Measures.
    
* K-medoids-based clustering: Comparison with k-means results.
    
* Nine clustering schemes in total: An in-depth analysis of various combinations, evaluated using two intrinsic and seven extrinsic criteria, and runtime for each. 
  
      In aeon: Kmedoids-DTW | Kmedoids-MSM | Kmedoids-ShapeDTW | Kmeans-MSM | Kmeans-ShapeDTW |
      In tslearn: Kmeans-DTW | Kmeans-SoftDTW | KernelKmeans-GAK | Kshape
  
## 2- Comparison of Averaging Methods

<table style="border-collapse: collapse; border: none;">
  <tr style="border: none;">
    <td style="vertical-align: top; border: none; padding-right: 20px;">
      Four methods evaluated:
      <ul>
        <li><a href="https://tslearn.readthedocs.io/en/stable/gen_modules/barycenters/tslearn.barycenters.dtw_barycenter_averaging.html#tslearn.barycenters.dtw_barycenter_averaging">DBA</a></li>
        <li><a href="https://github.com/MSD-IRIMAS/ShapeDBA">Shape-DBA</a></li>
        <li><a href="https://tslearn.readthedocs.io/en/stable/gen_modules/barycenters/tslearn.barycenters.softdtw_barycenter.html#tslearn.barycenters.softdtw_barycenter">Soft-DBA</a></li>
        <li><a href="https://github.com/pfmarteau/py-TEKA/blob/main/README.md">TEKA</a></li>
      </ul>
    </td>
    <td style="vertical-align: top; text-align: right; border: none;">
      <img src="https://github.com/user-attachments/assets/9be2e9c6-cb9d-40d4-a847-c3c5b7d49f4a" alt="Description of image" width="540">
    </td>
  </tr>
</table>


# Data, Protocol and Implementation
- Data Source: The data used is from [UCR UEA](https://www.timeseriesclassification.com/dataset.php) and was Z-normalized to zero mean and unit standard deviation.
  
- Evaluation Protocol: The evaluation of clustering methods relied on extrinsic criteria available in sklearn and two intrinsic criteria:

    Silhouette Score: Implemented using the [tslearn library](https://tslearn.readthedocs.io/en/latest/gen_modules/clustering/tslearn.clustering.silhouette_score.html).
  
    Davies-Bouldin Score: As there was no implementation handling multivariate time series, we created one to adapt it to the multivariate case. The code is available [here](https://github.com/bjdhafssa/MTS-Clustering-and-Averaging/blob/main/Clustering/tslearn_implementations/evaluator.py).
  
- Main Frameworks: Our primary frameworks were tslearn and the aeon toolkit. These libraries support different formats for multivariate time series: tslearn uses the format (n_ts, sz, d), aeon uses the format (n_ts, d, sz).

- All of our code adapts the data and methods to be compatible with these formats for each combination in our experiments.
  
- Since the libraries do not provide dependent versions for all time series dissimilarity measures, a fair comparison is achieved by employing the independent versions.
  
  

