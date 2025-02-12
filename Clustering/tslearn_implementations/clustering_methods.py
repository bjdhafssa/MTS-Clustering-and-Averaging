# clustering_methods.py 

from tslearn.clustering import TimeSeriesKMeans, KShape, KernelKMeans
from data_loader import load_dataset, working_datasets
from preprocessor import preprocess_data
import numpy as np

#######################################################################

def kmeans_dtw(X, n_clusters):
    
    model = TimeSeriesKMeans(n_clusters=n_clusters,
                             max_iter = 100, 
                             n_init = 10,
                             metric="dtw",
                             max_iter_barycenter=10,
                             init='k-means++')
    return model.fit_predict(X)

#########################################################################

def kmeans_soft_dtw(X, n_clusters):
    
    model = TimeSeriesKMeans(n_clusters=n_clusters,
                             max_iter = 100, 
                             n_init = 10,
                             metric="softdtw",
                             max_iter_barycenter=10,
                             init='k-means++'
                             )
    return model.fit_predict(X)

###########################################################################

def kernel_kmeans_gak(X, n_clusters):
    
    model = KernelKMeans(n_clusters=n_clusters, kernel="gak", max_iter=100, n_init=10)
    
    return model.fit_predict(X)


############################################################################

def kshape(X, n_clusters):

    model = KShape(n_clusters=n_clusters, max_iter=100, init="random")
    
    return model.fit_predict(X)


############################################################################

# Example Usage

if __name__ == "__main__":
    
    # Display available datasets to the user
    print("Available datasets:")
    for i, dataset in enumerate(working_datasets):
        print(f"{i + 1}. {dataset}")
    
    # Select a dataset by entering its index
    while True:
        try:
            user_input = int(input("\nEnter the number corresponding to your chosen dataset: "))
            if 1 <= user_input <= len(working_datasets):
                dataset_name = working_datasets[user_input - 1]
                break
            else:
                print(f"Please enter a number between 1 and {len(working_datasets)}.")
        except ValueError:
            print("Invalid input. Please enter a valid number.")

    print(f"\nYou selected: {dataset_name}")

    # Load and preprocess dataset
    dataset_name = "BasicMotions" 
    X_train, y_train, X_test, y_test = load_dataset(dataset_name)
    X = np.concatenate((X_train, X_test), axis=0)
    X_scaled = preprocess_data(X)


    # Determine the number of clusters based on unique labels in the dataset
    n_clusters = len(np.unique(np.concatenate((y_train, y_test))))

    # List of clustering methods to apply
    methods = [
        ("KMeans-DTW", kmeans_dtw),
        ("KMeans-Soft-DTW", kmeans_soft_dtw),
        ("KernelKMeans-GAK", kernel_kmeans_gak),
        ("KShape", kshape)
    ]

    # Apply each clustering method and display results
    for name, method in methods:
        print(f"\nPerforming clustering with {name}...")
        
        try:
            labels = method(X_scaled, n_clusters)
            print(f"Clustering completed with {name}")
            print(f"Number of clusters: {n_clusters}")
            print(f"Cluster labels: {labels[:10]}...")  # Display first 10 labels as a preview
        except Exception as e:
            print(f"Error during clustering with {name}: {e}")
