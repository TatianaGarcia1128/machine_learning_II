#import modules
import numpy as np
from numpy.random import seed
from numpy.random import choice

class KMEDOIDS:
    #Constructor
    def __init__(self, k):
        self.k = k
        self.medoids = None
        self.labels = None

    #Medoid Initialization
    def _init_medoids(self, X) : 
        seed (1)
        samples = choice(len(X), size=self.k, replace=False)
        return X[samples, :]
    
    #Computing the distances
    def _compute_d_p(self, X, medoids, p):
        m = len (X)
        medoids_shape = medoids. shape
        # If a 1-D array is provided,
        # it will be reshaped to a single row 2-D array
        if len(medoids_shape) == 1:
            medoids = medoids.reshape(1, len (medoids))
        k = len (medoids)
        
        S = np.empty((m, k))
        
        for i in range(m) :
            d_i = np. linalg.norm (X[i, :] - medoids, ord=p, axis=1)
            S[i, :] = d_i**p

        return S
    
    #Cluster Assignment
    def _assign_labels(self, S):
        return np.argmin(S, axis=1)
    
    #Swap Test
    def _update_medoids(self, X, medoids, p) :
        S = self._compute_d_p(X, medoids, p)
        labels = self._assign_labels(S)
        
        out_medoids = medoids
        
        for i in set (labels):
            avg_dissimilarity = np.sum(self._compute_d_p(X, medoids[i], p)) 

            cluster_points = X[labels == i]

            for datap in cluster_points:
                new_medoid = datap
                new_dissimilarity= np.sum(self._compute_d_p(X, datap, p))

            if new_dissimilarity < avg_dissimilarity:
                avg_dissimilarity = new_dissimilarity

                out_medoids[i] = datap

        return out_medoids
    
    def _has_converged(self, old_medoids, medoids):
        return set([tuple(x) for x in old_medoids]) == set([tuple(x) for x in medoids])
    
    #Full algorithm
    def fit(self, X, k, p, starting_medoids=None, max_steps=np.inf): #p=2 euclidian distance, p=1 manhattan distance
        if starting_medoids is None:
            medoids = self._init_medoids(X)
        else:
            medoids = starting_medoids
            
        converged = False
        labels = np.zeros(len(X))
        i = 1
        while (not converged) and (i <= max_steps):
            old_medoids = medoids.copy()
            
            S = self._compute_d_p(X, medoids, p)
            
            labels = self._assign_labels(S)
            
            medoids = self._update_medoids(X, medoids, p)
            
            converged = self._has_converged(old_medoids, medoids)
            i += 1
        
        self.medoids = medoids

        return self
    
    def predict(self, X):
        if self.medoids is None:
            raise Exception("Model not fitted yet. Call 'fit' with appropriate data before 'predict'.")
        S = self._compute_d_p(X, self.medoids, p=2)
        return self._assign_labels(S)        