# data_loader.py

from aeon.datasets import load_classification
import numpy as np

def load_dataset(dataset_name):
    X, y = load_classification(name=dataset_name)
    return X, y

def get_n_clusters(y):
    return len(np.unique(y))

working_datasets = ['ArticularyWordRecognition',
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
 'UWaveGestureLibrary']
