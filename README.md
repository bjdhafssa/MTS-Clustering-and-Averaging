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

  

# Bibliographic References 
Summary of surveys and benchmarks focused on time series dissimilarity measures and averaging.

| Reference                       | # Meas. | Survey  | Data | # Datasets | UTS/MTS | I,D | Task  | Aut/Reu/Lib   |
| [Giusti and Batista, 2013](https://dl.acm.org/doi/10.1109/BRACIS.2013.22)      | 4 EM / 48 | No   | UCR (2006) | 42      | NS      | NS      | TSCla (1-NN)      | NS   |
| [Wang et al., 2013](https://link.springer.com/article/10.1007/s10618-012-0250-5)             | 7 EM     | Yes   | UCR (2006) | 38      | NS     | NS       | TSCla (1-NN)      | Aut (Java), Reu                   |
| [Serrà and Arcos, 2014](https://dl.acm.org/doi/10.1016/j.knosys.2014.04.035)        | 3 EM / 7  | Yes    | UCR (2013) | 45      | NS        | NS        | TSCla (1-NN)     | Reu, NS                           |
| [Salarpour and Khotanlou, 2018](https://www.sciencedirect.com/science/article/abs/pii/S0031320317303163) | 14 EM    | No        | NS         | 23      | MTS     | NS        | HAC     | Aut (Matlab, Mex)                 |
| [Paparrizos et al., 2020](https://dl.acm.org/doi/10.1145/3318464.3389760)       | 7 EM, 4 KM | Yes     | UCR (2018) | 128     | UTS       | —         | TSCla (1-NN)    | Matlab, Reu                       |
| [Parmezan et al., 2022](https://www.researchgate.net/publication/362170345_Time_Series_Prediction_via_Similarity_Search_Exploring_Invariances_Distance_Measures_and_Ensemble_Functions)| 4 EM / 25| No|ICMC-USP | 55| UTS  | —   |kNN-TSPI| NS  |
| [Shifaz et al., 2023](https://link.springer.com/article/10.1007/s10115-023-01835-4)          | 11 EM    | Yes   | UEA (2018) | 23      | MTS   | I,D    | TSCla (1-NN)    | Aut (Java)     |
| [Holder et al., 2024](https://link.springer.com/article/10.1007/s10115-023-01952-0)          | 10 EM    | Yes         | UCR (2018) | 112     | UTS     | —    | TSClu     | Lib (AEON)           |
| [Górecki et al., 2024](https://www.sciencedirect.com/science/article/abs/pii/S1877750324000280)  | 27 EM / 56 | Yes   | UCR (2019) | 128     | UTS    | —   | TSCla (1-NN)  | Aut (C++), Lib (CRAN)             |
| [Paparrizos et al., 2024](https://arxiv.org/abs/2412.20574v1)       | 10 EM, 4 KM / 100+ | Yes        | —          | —       | —     | —      | —       | —                                 |
| **Ours**                | 5 EM & 2 KM | Yes          | UCR (2024) | 22      | MTS      | I        | TSClu      | AEON, tslearn                     |


- **# Meas.:** Number of measures considered.
- **EM:** Elastic measure.
- **KM:** Kernel measure.
- **x EM / y:** x EM over a total of y measures.
- **UTS / MTS:** Univariate / multivariate time series.
- **I,D:** Independent and Dependent versions of MTS dissimilarity measure.
- **TSCla:** Time series classification.
- **TSClu:** Time series clustering.
- **NN:** Nearest-neighbor.
- **Rep.:** Repository.
- **HAC:** Hierarchical agglomerative clustering.
- **NS:** Not specified.
- **Aut/Reu/Lib:** Authors’ implementation of code for dissimilarity measures / Reuse of other authors’ code / Library used.
- **TSP:** Time series prediction.

  
  
  

