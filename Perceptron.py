import numpy as np
'''
Perceptron binary classification implementation from scratch.

Beginner intuition:
- Perceptron is a linear classifier.
- It updates weights only when it makes a mistake.
- Labels are treated as -1 and +1 internally.

Core formulas:
- Linear score: z = w . x + b
- Prediction sign: y_hat = sign(z)
- Update on mistake (when y * z <= 0):
    w = w + lr * y * x
    b = b + lr * y

This implementation is educational and can be extended with regularization,
averaged perceptron, and multiclass variants.
'''


class PerceptronModel:
    def __init__(self, learning_rate=0.01, n_iters=2000):
        '''
        Initializes Perceptron with learning rate and number of iterations.

        Parameters:
        learning_rate (float): Step size for weight updates.
        n_iters (int): Number of training passes over the dataset.
        '''
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = 0.0

    def fit(self, X, y):
        '''
        Fits Perceptron model on binary labels.

        Training procedure:
        1) Convert labels from {0,1} to {-1,+1}.
        2) For each sample, compute linear score z = w.x + b.
        3) If misclassified (y*z <= 0), apply perceptron update.

        Parameters:
        X (numpy.ndarray): Training feature matrix.
        y (numpy.ndarray): Binary labels encoded as 0 or 1.
        '''
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features, dtype=float)
        self.bias = 0.0

        y_transformed = np.where(y <= 0, -1, 1)

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                if y_transformed[idx] * linear_output <= 0:
                    self.weights += self.learning_rate * y_transformed[idx] * x_i
                    self.bias += self.learning_rate * y_transformed[idx]

    def predict(self, X):
        '''
        Predicts binary labels for input samples.

        Decision rule:
        - If w.x + b >= 0, predict 1.
        - Else, predict 0.

        Parameters:
        X (numpy.ndarray): Input feature matrix.

        Returns:
        numpy.ndarray: Predicted labels (0 or 1).
        '''
        linear_output = np.dot(X, self.weights) + self.bias
        return np.where(linear_output >= 0, 1, 0)
