from aeon.clustering import TimeSeriesKMeans, TimeSeriesKMedoids


def kmedoids_dtw(X, n_clusters):
    return TimeSeriesKMedoids(n_clusters=n_clusters, distance="dtw", method="pam",
                              n_init=10, max_iter=100).fit_predict(X)

def kmedoids_msm(X, n_clusters):
    return TimeSeriesKMedoids(n_clusters=n_clusters, distance="msm", method="pam",
                              n_init=10, max_iter=100).fit_predict(X)

def kmedoids_shape_dtw(X, n_clusters):
    return TimeSeriesKMedoids(n_clusters=n_clusters, distance="shape_dtw", method="pam",
                              n_init=10, max_iter=100).fit_predict(X)

def kmeans_msm(X, n_clusters):
    return TimeSeriesKMeans(n_clusters=n_clusters, distance="msm", 
                            averaging_method="ba",n_init=10, max_iter=100).fit_predict(X)

def kmeans_shape_dtw(X, n_clusters):
    return TimeSeriesKMeans(n_clusters=n_clusters, distance="shape_dtw", 
                            averaging_method="ba", n_init=10, max_iter=100).fit_predict(X)
