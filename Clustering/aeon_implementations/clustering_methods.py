from aeon.clustering import TimeSeriesKMeans, TimeSeriesKMedoids
import TekaKernelKMeans
import KDTWKMedoids

# Config parameters
n_init = 10
verbose = False

def kmedoids_dtw(X, args: dict ={"n_clusters":3}):
    return TimeSeriesKMedoids(n_clusters=args["n_clusters"], distance="dtw", method="pam",
                              n_init=args["n_init"], max_iter=100).fit_predict(X)

def kmedoids_msm(X, args: dict ={"n_clusters":3}):
    return TimeSeriesKMedoids(n_clusters=args["n_clusters"], distance="msm", method="pam",
                              n_init=args["n_init"], max_iter=100).fit_predict(X)

def kmedoids_shape_dtw(X, args: dict ={"n_clusters":3}):
    return TimeSeriesKMedoids(n_clusters=args["n_clusters"], distance="shape_dtw", method="pam",
                              n_init=args["n_init"], max_iter=100).fit_predict(X)

def kmeans_msm(X, args: dict ={"n_clusters":3}):
    return TimeSeriesKMeans(n_clusters=args["n_clusters"], distance="msm", 
                            #averaging_method="ba", init_algorithm="kmeans++", n_init=10, max_iter=100).fit_predict(X)
                            averaging_method="ba", init="kmeans++", n_init=n_init, max_iter=100).fit_predict(X) ##pfm

def kmeans_shape_dtw(X, args: dict ={"n_clusters":3}):
    return TimeSeriesKMeans(n_clusters=args["n_clusters"], distance="shape_dtw", 
                            #averaging_method="ba", init_algorithm="kmeans++", n_init=10, max_iter=100).fit_predict(X)
                            averaging_method="ba", init="kmeans++", n_init=args["n_init"], max_iter=100).fit_predict(X) ##pfm

##pfm: 
def kmedoids_kdtw(X, args: dict ={"n_clusters":3, "sigma":1., "epsilon":1e-3}):
    _X = X.swapaxes(1, 2)
    kdtw_kmd = KDTWKMedoids.KDTWKMedoids(n_clusters=args["n_clusters"], method="pam", init='random',
        distance_params={"sigma": args["sigma"], "epsilon":args["epsilon"]}, n_init=args["n_init"], max_iter=100, verbose=verbose)
    kdtw_kmd.fit(_X)
    return kdtw_kmd.predict(_X) 

##pfm: 
def kmeans_teka(X, args: dict ={"n_clusters":3, "sigma":1., "epsilon":1e-300}):
    print("sigma:",args["sigma"], "epsilon:", args["epsilon"])
    _X = X.swapaxes(1, 2)
    init_type = "kmedoids"
    teka_km = TekaKernelKMeans.TekaKernelKMeans(n_clusters=args["n_clusters"], init_type=init_type, 
            kernel_params={"sigma": args["sigma"], "epsilon":args["epsilon"]}, n_init=args["n_init"], max_iter=100, verbose=verbose)
    teka_km.fit(_X)
    return teka_km.predict(_X) 

