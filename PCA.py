import numpy as np
'''
Principal Component Analysis (PCA) implementation from scratch.

Beginner intuition:
- Center data by removing feature means.
- Compute covariance matrix to capture joint variation.
- Find eigenvectors with largest eigenvalues.
- Project data onto top components to reduce dimensions.
'''


class PCAScratch:
    def __init__(self, n_components=2):
        '''
        Parameters:
        n_components (int): Number of principal components to keep.
        '''
        self.n_components = n_components
        self.mean = None
        self.components = None
        self.explained_variance = None
        self.explained_variance_ratio_ = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        cov_matrix = np.cov(X_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        self.components = eigenvectors[:, : self.n_components]
        self.explained_variance = eigenvalues[: self.n_components]
        total_variance = np.sum(eigenvalues)
        if total_variance == 0:
            self.explained_variance_ratio_ = np.zeros(self.n_components)
        else:
            self.explained_variance_ratio_ = self.explained_variance / total_variance

    def transform(self, X):
        X_centered = X - self.mean
        return np.dot(X_centered, self.components)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_reduced):
        return np.dot(X_reduced, self.components.T) + self.mean
