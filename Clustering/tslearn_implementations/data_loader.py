# data_loader.py

import numpy as np
from tslearn.datasets import UCR_UEA_datasets

# List of datasets that are known to work successfully
working_datasets = [
    'ArticularyWordRecognition',
    'BasicMotions',
    'Cricket',
    'EigenWorms',
    'Epilepsy',
    'EthanolConcentration',
    'FaceDetection',
    'FingerMovements',
    'HandMovementDirection',
    'Handwriting',
    'Heartbeat',
    'Libras',
    'LSST',
    'MotorImagery',
    'NATOPS',
    'PenDigits',
    'PEMS-SF',
    'RacketSports',
    'SelfRegulationSCP1',
    'SelfRegulationSCP2',
    'StandWalkJump',
    'UWaveGestureLibrary'
]


def load_dataset(dataset_name):
    
    """
    Loads a specific dataset from the UCR/UEA collection.

    """
    X_train, y_train, X_test, y_test = UCR_UEA_datasets().load_dataset(dataset_name)
    return X_train, y_train, X_test, y_test



def get_working_datasets(multivariate_datasets):
    """
    Identifies datasets that can be successfully loaded and processed.

    Args:
        multivariate_datasets (list): List of multivariate dataset names to check.

    Returns:
        list: A list of dataset names that can be successfully loaded and processed.
    
    Notes:
        This function attempts to load each dataset and validates its structure.
        Datasets that fail to load are skipped with an error message.
    """
    
    # List to store the names of datasets that can be successfully processed
    working_datasets = []

    # Iterate through each dataset name
    for dataset_name in multivariate_datasets:
        try:
            # Load the dataset
            X_train, y_train, X_test, y_test = load_dataset(dataset_name)

            # Concatenate training and test data
            X = np.concatenate((X_train, X_test), axis=0)
            Y = np.concatenate((y_train, y_test), axis=0)

            # Validate the data (optional: add more checks if needed)
            if X.shape[0] > 0 and len(np.unique(Y)) > 1:
                working_datasets.append(dataset_name)
        
        except Exception as e:
            # Print an error message if loading fails
            print(f"Error processing dataset {dataset_name}: {e}")

    return working_datasets


