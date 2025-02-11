import numpy as np
import matplotlib.pyplot as plt
from aeon.datasets import load_classification
from aeon.transformations.collection import Normalizer
from tslearn.barycenters import dtw_barycenter_averaging, softdtw_barycenter
from aeon.clustering.averaging import elastic_barycenter_average
from barycenter_averaging_comparison import get_kdtw_inertia, get_iTEKACentroid, kdtw
from teka import PyTEKA
TEKA = PyTEKA()


def calculate_centroids_for_class(X, X_ts, y, class_label):
    # Select all samples for the given class
    class_samples_aeon = X[y == class_label]
    class_samples_tslearn = X_ts[y == class_label]
    
    # Calculate centroids using different methods
    centroids = {}
    dba_c = dtw_barycenter_averaging(class_samples_tslearn, max_iter=10, tol=1e-5)
    centroids['DBA'] = dba_c.transpose()
    softdba_c = softdtw_barycenter(class_samples_tslearn, gamma=1, max_iter=10, tol=1e-5)
    centroids['Soft-DBA'] = softdba_c.transpose()
    centroids['Shape-DBA'] = elastic_barycenter_average(class_samples_aeon, distance="shape_dtw", 
                                                                                max_iters=10, tol=1e-5)
    
    # Using TEKA
    sigma = 1
    epsilon = 1e-3
    initial_centroid = class_samples_tslearn[0]  # First time series as initial centroid
    dim = len(class_samples_tslearn[0][0])
    
    #centroid_teka, Tstd, inertia, TTp
    centroid_teka = get_iTEKACentroid(class_samples_tslearn, initial_centroid, sigma, epsilon, npass=10)
    centroid_teka = centroid_teka[:,0:dim]  
    centroids['TEKA'] = centroid_teka.transpose()
    
    return class_samples_aeon, class_samples_tslearn, centroids

def plot_centroids_for_class_n_dimension(class_samples, centroids, class_label, dataset_name, dimension=None):
    """
    Plot centroids for each class with the original time series in the background, 
    for a specific dimension or all dimensions.
    
    Args:
    class_samples (np.ndarray): Time series samples for the specific class of shape (n_samples, d, size).
    centroids (dict): A dictionary where keys are method names and values are centroids of shape (d, size).
    class_label (int): The label of the class being visualized.
    dataset_name (str): The name of the dataset being used.
    dimension (int, optional): The specific dimension to plot. If None, all dimensions are plotted.
    """
    num_dimensions = class_samples.shape[1]
    
    if dimension is not None:
        if dimension < 0 or dimension >= num_dimensions:
            raise ValueError(f"Dimension must be between 0 and {num_dimensions-1}")
        dimensions_to_plot = [dimension]
        fig, ax = plt.subplots(figsize=(10, 6))
        axes = [ax]
    else:
        dimensions_to_plot = range(num_dimensions)
        fig, axes = plt.subplots(3, 2, figsize=(15, 10))  # 3 rows, 2 columns layout for 6 dimensions
        axes = axes.flatten()  # Flatten the 2D array of axes to easily iterate through them

    # Define colors for different centroid methods
    method_colors = {'DBA': 'b', 'Soft-DBA': 'g', 'Shape-DBA': 'r', 'TEKA': 'm'}
    
    for i, dim in enumerate(dimensions_to_plot):
        ax = axes[i]
        
        # Plot all time series in the background in light gray for the specific dimension
        for sample in class_samples:
            ax.plot(sample[dim], color='lightgray', alpha=0.5)  # Background in light gray
        
        # Plot centroids for each method in different colors
        for method, centroid in centroids.items():
            ax.plot(centroid[dim], color=method_colors[method], label=f'{method} Centroid', linewidth=2)
        
        ax.set_title(f'Dimension {dim + 1}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Feature Value')
        #ax.legend(loc='upper right')
        ax.legend(loc='lower left')  

    # Adjust layout to avoid overlap
    plt.tight_layout()
    if dimension is None:
        plt.suptitle(f'Centroids Comparison for {dataset_name} - Class {class_label}', y=1.02)
    else:
        plt.title(f'Centroids Comparison for {dataset_name} - Class {class_label}, Dimension {dimension}')
    
    plt.savefig(f'{dataset_name}_class_{class_label}_dim_{dimension if dimension is not None else "all"}.png', bbox_inches='tight')
    #plt.show()

#######################################################################################################################################

def main():
    # Display available datasets and prompt user to choose one
    print("Available datasets:")
    for i, dataset in enumerate(working_datasets):
        print(f"{i + 1}. {dataset}")
    
    dataset_choice = int(input("Enter the number corresponding to the dataset you want to use: ")) - 1
    if not (0 <= dataset_choice < len(working_datasets)):
        raise ValueError("Invalid dataset choice.")
    
    dataset_name = working_datasets[dataset_choice]
    
    # Load the selected dataset and normalize it
    X, y = load_classification(name=dataset_name)
    normalizer = Normalizer()
    Xn = normalizer.fit_transform(X)
    
    # Reshape X to match expected shape (n_ts, size, d)
    X_ts = Xn.transpose(0, 2, 1)

    # Display available classes and prompt user to choose one
    unique_classes = np.unique(y)
    print("\nAvailable classes:")
    for cls in unique_classes:
        print(f"- {cls}")
    
    class_label = input("Enter the label of the class you want to analyze: ")
    
    if class_label not in unique_classes.astype(str):
        raise ValueError("Invalid class label.")
    
    # Display number of dimensions and prompt user to choose one or all
    num_dimensions = Xn.shape[2]
    print(f"\nThe dataset has {num_dimensions} dimensions.")
    
    dimension_choice = input("Enter the dimension you want to visualize (or type 'all' to visualize all dimensions): ")
    
    if dimension_choice.lower() == 'all':
        dimension_choice = None
    else:
        dimension_choice = int(dimension_choice)
        if not (0 <= dimension_choice < num_dimensions):
            raise ValueError("Invalid dimension choice.")
    
    # Calculate centroids and plot results
    class_samples_aeon, _, centroids = calculate_centroids_for_class(Xn, X_ts, y.astype(str), class_label)
    
    plot_centroids_for_class_n_dimension(
        class_samples_aeon,
        centroids,
        class_label,
        dataset_name,
        dimension=dimension_choice
    )
    
    print("Centroid comparison plot saved successfully.")

if __name__ == "__main__":
    main()
