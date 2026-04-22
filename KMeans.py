import numpy as np
'''
K-Means clustering implementation from scratch.

Beginner intuition:
- Pick k centers (centroids).
- Assign each point to closest centroid.
- Recompute each centroid as mean of its assigned points.
- Repeat until centroids stabilize.
'''


class KMeansScratch:
    def __init__(self, n_clusters=3, max_iters=300, tol=1e-4, random_state=42):
        '''
        Parameters:
        n_clusters (int): Number of clusters.
        max_iters (int): Maximum optimization iterations.
        tol (float): Stop when centroid movement is smaller than this threshold.
        random_state (int): Seed for reproducible centroid initialization.
        '''
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.random_state = random_state
        self.centroids = None

    def fit(self, X):
        rng = np.random.default_rng(self.random_state)
        initial_indices = rng.choice(
            X.shape[0], size=self.n_clusters, replace=False)
        self.centroids = X[initial_indices].copy()

        for _ in range(self.max_iters):
            labels = self._assign_clusters(X)
            new_centroids = self._compute_centroids(X, labels)

            shift = np.linalg.norm(new_centroids - self.centroids)
            self.centroids = new_centroids

            if shift < self.tol:
                break

    def _assign_clusters(self, X):
        distances = np.linalg.norm(
            X[:, np.newaxis] - self.centroids[np.newaxis, :], axis=2)
        return np.argmin(distances, axis=1)

    def _compute_centroids(self, X, labels):
        centroids = np.zeros((self.n_clusters, X.shape[1]), dtype=float)
        for cluster_idx in range(self.n_clusters):
            cluster_points = X[labels == cluster_idx]
            if len(cluster_points) == 0:
                centroids[cluster_idx] = self.centroids[cluster_idx]
            else:
                centroids[cluster_idx] = np.mean(cluster_points, axis=0)
        return centroids

    def predict(self, X):
        return self._assign_clusters(X)

    def fit_predict(self, X):
        self.fit(X)
        return self.predict(X)
