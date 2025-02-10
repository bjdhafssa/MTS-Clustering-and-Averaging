# preprocessor.py

from tslearn.preprocessing import TimeSeriesScalerMeanVariance

def preprocess_data(X, mu=0., std=1.):
    
    scaler = TimeSeriesScalerMeanVariance(mu=mu, std=std)
    return scaler.fit_transform(X)

