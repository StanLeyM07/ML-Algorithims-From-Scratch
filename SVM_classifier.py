import numpy as np
'''
Linear SVM (binary classification) implementation from scratch.

Beginner intuition:
- SVM finds a separating hyperplane with maximum margin.
- Soft-margin SVM allows some margin violations for noisy data.
- We optimize hinge-loss + L2 regularization.

Objective:
min_w,b 0.5 * ||w||^2 + C * mean(max(0, 1 - y * (w.x + b)))
where y is in {-1, +1}.
'''


class LinearSVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=2000, C=1.0):
        '''
        Parameters:
        learning_rate (float): Gradient descent step size.
        lambda_param (float): L2 regularization strength on weights.
        n_iters (int): Number of optimization passes over data.
        C (float): Hinge-loss penalty weight.
        '''
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.C = C
        self.weights = None
        self.bias = 0.0

    def fit(self, X, y):
        '''
        Fits linear SVM on binary labels encoded as 0/1 or -1/+1.
        '''
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features, dtype=float)
        self.bias = 0.0

        y_transformed = np.where(y <= 0, -1.0, 1.0)

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                margin = y_transformed[idx] * \
                    (np.dot(x_i, self.weights) + self.bias)

                if margin >= 1:
                    grad_w = self.lambda_param * self.weights
                    grad_b = 0.0
                else:
                    grad_w = self.lambda_param * self.weights - \
                        self.C * y_transformed[idx] * x_i
                    grad_b = -self.C * y_transformed[idx]

                self.weights -= self.learning_rate * grad_w
                self.bias -= self.learning_rate * grad_b

    def decision_function(self, X):
        '''
        Returns raw decision scores w.x + b.
        '''
        return np.dot(X, self.weights) + self.bias

    def predict(self, X):
        '''
        Predicts labels in {0, 1}.
        '''
        scores = self.decision_function(X)
        return np.where(scores >= 0, 1, 0)
