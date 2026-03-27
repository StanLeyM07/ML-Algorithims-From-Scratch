import numpy as np
'''
Gaussian Naive Bayes implementation from scratch.

Beginner intuition:
- Use Bayes theorem to compute class probabilities.
- "Naive" means features are assumed conditionally independent given class.
- "Gaussian" means each feature is modeled with a normal distribution per class.

Core formulas:
- Bayes theorem (proportional form):
    P(c | x) proportional to P(c) * product(P(x_j | c))
- Gaussian likelihood for one feature:
    P(x_j | c) = (1 / sqrt(2*pi*var_cj)) * exp(-(x_j - mean_cj)^2 / (2*var_cj))
- We use log-space for stability:
    log P(c | x) = log P(c) + sum(log P(x_j | c))

This implementation is educational and can be extended with smoothing and priors tuning.
'''


class GaussianNaiveBayes:
    def __init__(self):
        '''
        Initializes Gaussian Naive Bayes model parameters.
        '''
        self.classes = None
        self.mean = None
        self.var = None
        self.priors = None

    def fit(self, X, y):
        '''
        Fits Gaussian Naive Bayes by computing class-wise mean, variance, and priors.

        For each class c and feature j, we estimate:
        - mean_cj
        - var_cj
        - prior P(c)

        Parameters:
        X (numpy.ndarray): Training feature matrix.
        y (numpy.ndarray): Training class labels.
        '''
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        self.mean = np.zeros((n_classes, n_features), dtype=float)
        self.var = np.zeros((n_classes, n_features), dtype=float)
        self.priors = np.zeros(n_classes, dtype=float)

        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            self.mean[idx, :] = np.mean(X_c, axis=0)
            self.var[idx, :] = np.var(X_c, axis=0) + 1e-9
            self.priors[idx] = X_c.shape[0] / n_samples

    def _log_gaussian_probability(self, class_idx, x):
        '''
        Computes log-probability of features for a class under Gaussian assumption.

        Log Gaussian formula per feature j:
        log P(x_j | c) = -0.5 * log(2*pi*var_cj) - (x_j - mean_cj)^2 / (2*var_cj)

        Parameters:
        class_idx (int): Class index in internal parameter arrays.
        x (numpy.ndarray): Input sample.

        Returns:
        numpy.ndarray: Log-probabilities per feature.
        '''
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        return -0.5 * np.log(2.0 * np.pi * var) - ((x - mean) ** 2) / (2.0 * var)

    def _predict_single(self, x):
        '''
        Predicts class label for one sample using maximum posterior probability.

        Decision rule:
        predict class = argmax_c [log P(c) + sum(log P(x_j | c))]

        Parameters:
        x (numpy.ndarray): Input sample.

        Returns:
        int: Predicted class label.
        '''
        posteriors = []

        for idx, c in enumerate(self.classes):
            prior = np.log(self.priors[idx])
            conditional = np.sum(self._log_gaussian_probability(idx, x))
            posterior = prior + conditional
            posteriors.append(posterior)

        return self.classes[np.argmax(posteriors)]

    def predict(self, X):
        '''
        Predicts class labels for all input samples.

        Parameters:
        X (numpy.ndarray): Input feature matrix.

        Returns:
        numpy.ndarray: Predicted class labels.
        '''
        return np.array([self._predict_single(x) for x in X])
