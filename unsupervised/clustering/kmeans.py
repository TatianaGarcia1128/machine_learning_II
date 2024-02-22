#import modules
import numpy as np

class KMEANS:
    #Constructor
    def __init__(self,  K, max_iter):
        self.K = K
        self.max_iter = max_iter

    # randomly initializing K centroid by picking K samples from X
    def _initialize_random_centroids(self, X):
        """Initializes and returns k random centroids"""
        m, n = np.shape(X)
        # a centroid should be of shape (1, n), so the centroids array will be of shape (K, n)
        centroids = np.empty((self.K, n))
        for i in range(self.K):
            # pick a random data point from X as the centroid
            centroids[i] =  X[np.random.choice(range(m))] 
        return centroids
    
    #Calculate euclidean distance between two vectors
    def _euclidean_distance(self, x1, x2):
        """Calculates and returns the euclidean distance between two vectors x1 and x2"""
        return np.linalg.norm(x1 - x2)


    #Finding the closest centroid to a given data point
    def _closest_centroid(self, x, centroids):
        """Finds and returns the index of the closest centroid for a given vector x"""
        distances = np.empty(self.K)
        for i in range(self.K):
            distances[i] = self._euclidean_distance(centroids[i], x)
        return np.argmin(distances) 
    
    #Create clusters
    def _create_clusters(self, centroids, X):
        """Returns an array of cluster indices for all the data samples"""
        m, _ = np.shape(X)
        cluster_idx = np.empty(m)
        for i in range(m):
            cluster_idx[i] = self._closest_centroid(X[i], centroids)
        return cluster_idx
    
    #Compute means
    def _compute_means(self, cluster_idx, X):
        """Computes and returns the new centroids of the clusters"""
        _, n = np.shape(X)
        centroids = np.empty((self.K, n))
        for i in range(self.K):
            points = X[cluster_idx == i] # gather points for the cluster i
            centroids[i] = np.mean(points, axis=0) # use axis=0 to compute means across points
        return centroids
    

    #Putting everything together
    def fit(self, X):
        """Runs the K-means algorithm and computes the final clusters"""
        # initialize random centroids
        centroids = self._initialize_random_centroids(X)
        # loop till max_iterations or convergance
        print(f"initial centroids: {centroids}")
        for _ in range(self.max_iter):
            # create clusters by assigning the samples to the closet centroids
            clusters = self._create_clusters(centroids, X)
            previous_centroids = centroids                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
            # compute means of the clusters and assign to centroids
            self.centroids_ = self._compute_means(clusters, X)
            # if the new_centroids are the same as the old centroids, return clusters
            diff = previous_centroids - centroids
            if not diff.any():
                break

    def predict(self, X):
        """Predict the closest cluster each sample in X belongs to."""
        clusters = self._create_clusters(self.centroids_, X) #convention to denote that this variable is established during model fitting and is available after that process.
        return clusters