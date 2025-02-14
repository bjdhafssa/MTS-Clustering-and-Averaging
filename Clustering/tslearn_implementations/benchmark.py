# benchmark.py
import pickle
import os
import pandas as pd
import numpy as np
import sys
import timeit
from data_loader import load_dataset, working_datasets
from preprocessor import preprocess_data
from clustering_methods import kmeans_dtw, kmeans_soft_dtw, kernel_kmeans_gak, kshape
from evaluator import evaluate_clustering

def run_benchmark(dataset_name):
    clustering_methods = [
        ("K-means DTW", kmeans_dtw),
        ("K-means Soft-DTW", kmeans_soft_dtw),
        ("Kernel K-means GAK", kernel_kmeans_gak),
        ("K-Shape", kshape)
    ]

        
    results = []
    clustering_outputs = {}

    X_train, y_train, X_test, y_test = load_dataset(dataset_name)
    X = np.concatenate((X_train, X_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)
    n_clusters = len(np.unique(y))
    X_processed = preprocess_data(X)


    for method_name, clustering_function in clustering_methods:
        try:
            # runtime
            runtime = timeit.timeit(lambda: clustering_function(X_processed, n_clusters), number=1)
            
            # clustering function
            labels = clustering_function(X_processed, n_clusters)
            
            # evaluation
            metrics = evaluate_clustering(X_processed, labels, true_labels=y)
            
            # Ajouter les résultats au tableau
            result = {
                "Dataset": dataset_name,
                "Clustering Method": method_name,
                "Runtime": runtime,
                **metrics
            }
            results.append(result)
            
            print(f"Completed: {dataset_name}, {method_name}, Runtime: {runtime:.2f} seconds")
            
            # Sauvegarder les étiquettes de clustering
            clustering_outputs[method_name] = labels
            
        except Exception as e:
            print(f"Error with {dataset_name}, {method_name}: {str(e)}")


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
    
