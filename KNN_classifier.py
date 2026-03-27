import numpy as np
'''
K-Nearest Neighbors (KNN) classification implementation from scratch.

Beginner intuition:
- KNN does not learn explicit weights.
- It stores training data and predicts using the nearest points.
- Nearby points are assumed to have similar labels.

Core formulas:
- Euclidean distance between two points x and x_i:
    d(x, x_i) = sqrt(sum((x_j - x_ij)^2))
- Prediction is majority vote among the k nearest labels.

This implementation is educational and can be optimized for speed on large data.
'''


class KNNClassifier:
    def __init__(self, k=5):
        '''
        Initializes the KNN classifier with number of neighbors.

        Parameters:
        k (int): Number of nearest neighbors to use for voting.
        '''
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        '''
        Stores training data for KNN prediction.

        KNN is a lazy learner, so "fit" mainly memorizes the dataset.

        Parameters:
        X (numpy.ndarray): Training feature matrix.
        y (numpy.ndarray): Training class labels.
        '''
        self.X_train = X
        self.y_train = y

    def _euclidean_distance(self, x1, x2):
        '''
        Computes Euclidean distance between two samples.

        Formula:
        d(x1, x2) = sqrt(sum((x1 - x2)^2))

        Parameters:
        x1 (numpy.ndarray): First sample.
        x2 (numpy.ndarray): Second sample.

        Returns:
        float: Euclidean distance.
        '''
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def _predict_single(self, x):
        '''
        Predicts class label for one sample using majority vote among nearest neighbors.

        Steps:
        1) Compute distance from x to every training sample.
        2) Select indices of k smallest distances.
        3) Return the most frequent label among those k neighbors.

        Parameters:
        x (numpy.ndarray): Input sample.

        Returns:
        int: Predicted class label.
        '''
        distances = [self._euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[: self.k]
        k_nearest_labels = self.y_train[k_indices]
        return np.bincount(k_nearest_labels).argmax()

    def predict(self, X):
        '''
        Predicts class labels for all input samples.

        Parameters:
        X (numpy.ndarray): Input feature matrix.

        Returns:
        numpy.ndarray: Predicted class labels.
        '''
        return np.array([self._predict_single(x) for x in X])
