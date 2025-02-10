# benchmark.py
import pickle
import os
import pandas as pd
import sys
import timeit
from data_loader import load_dataset, get_n_clusters, working_datasets
from preprocessor import preprocess_data
from clustering_methods import kmedoids_dtw, kmedoids_msm, kmedoids_shape_dtw, kmeans_msm, kmeans_shape_dtw
from evaluator import evaluate_clustering

def run_benchmark(dataset_name):
    clustering_methods = [
        ("KMedoids-DTW", kmedoids_dtw),
        ("KMedoids-MSM", kmedoids_msm),
        ("KMedoids-ShapeDTW", kmedoids_shape_dtw),
        ("KMeans-MSM", kmeans_msm),
        ("KMeans-ShapeDTW", kmeans_shape_dtw)
    ]
   
    results = []
    clustering_outputs = {}

    
    X, y = load_dataset(dataset_name)
    n_clusters = get_n_clusters(y)
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
                
            print(f"Completed: {dataset_name}, {method_name}")
        except Exception as e:
            print(f"Error with {dataset_name}, {method_name}: {str(e)}")

    return pd.DataFrame(results), clustering_outputs   
    
    

#####################################################################################################################


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
