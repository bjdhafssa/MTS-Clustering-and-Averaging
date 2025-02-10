# benchmark.py
import pickle
import os
import pandas as pd
import numpy as np
import sys
import timeit
from data_loader import load_dataset, working_datasets
from preprocessor import preprocess_data
from clustering_methods import kmeans_dtw_dba, kmeans_soft_dtw_soft_dba, kernel_kmeans_gak_teka, kshape_method
from evaluator import evaluate_clustering

def run_benchmark(dataset_name):
    clustering_methods = [
        ("K-means DTW-DBA", kmeans_dtw_dba),
        ("K-means Soft-DTW Soft-DBA", kmeans_soft_dtw_soft_dba),
        ("Kernel K-means GAK", kernel_kmeans_gak_teka),
        ("K-Shape", kshape_method)
    ]

        
    results = []
    clustering_outputs = {}

    X_train, y_train, X_test, y_test = load_dataset(dataset_name)
    X = np.concatenate((X_train, X_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)
    n_clusters = len(np.unique(y))
    X_processed = preprocess_data(X)
    
        
    for clustering_method in clustering_methods:
        try:
                
            # runtime
            runtime = timeit.timeit(lambda: clustering_method(X_processed, n_clusters), number=1)
                
            labels = clustering_method(X_processed, n_clusters)
            metrics = evaluate_clustering(X_processed, labels, true_labels=y)
                
            result = {
                    "Dataset": dataset_name,
                    "Clustering Method": method_name,
                    **metrics
                }
            results.append(result)
                
            print(f"Completed: {dataset_name}, {method_name}, Runtime: {runtime:.2f} seconds")
                
            # Save clustering output
            clustering_outputs[f"{method_name}"] = labels
                
            print(f"Completed: {dataset_name}, {method_name}, {norm_method}")
        except Exception as e:
            print(f"Error with {dataset_name}, {method_name}, {norm_method}: {str(e)}")

    return pd.DataFrame(results), clustering_outputs

####################################################################################################################

# Example Usage

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python benchmark.py <dataset_name>")
        sys.exit(1)
    
    dataset_name = sys.argv[1]
    results_df, clustering_outputs = run_benchmark(dataset_name)
    
    # Save the results to a CSV file
    results_df.to_csv(f"clustering_results_{dataset_name}.csv", index=False)
    print(f"Results saved to clustering_results_{dataset_name}.csv")

    
    # Save clustering outputs
    output_dir = f"clustering_outputs_{dataset_name}"
    os.makedirs(output_dir, exist_ok=True)
    for method_name, labels in clustering_outputs.items():
        with open(os.path.join(output_dir, f"{method_name}.pkl"), 'wb') as f:
            pickle.dump(labels, f)
    print(f"Clustering outputs saved to {output_dir}")
    