# preprocessor.py

from aeon.transformations.collection import Normalizer

def preprocess_data(X):
    
    normaliser = Normalizer()
    return normaliser.fit_transform(X)

