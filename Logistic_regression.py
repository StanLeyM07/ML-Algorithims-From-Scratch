import numpy as np
'''
Logistic Regression implementation from scratch using Gradient Descent.

Beginner intuition:
- Start with a linear score z = Xw + b.
- Convert z into probability using sigmoid.
- Use a threshold (usually 0.5) to get class 0 or 1.

Core formulas:
- Linear score: z = Xw + b
- Sigmoid: sigma(z) = 1 / (1 + exp(-z))
- Probability: p = sigma(z)
- Gradient terms (binary cross-entropy form):
    dw = (1/n) * X^T * (p - y)
    db = (1/n) * sum(p - y)
- Update rule:
    w = w - lr * dw
    b = b - lr * db

This implementation is educational and can be improved with regularization,
early stopping, and multiclass extension.
'''


class LogisticRegressionModel:

    def __init__(self, learning_rate=0.01, n_iters=3000):
        '''
        Initializes the Logistic Regression model with a learning rate and number of iterations.

        Parameters:
        learning_rate (float): The step size for gradient descent updates.
        n_iters (int): Number of gradient descent iterations.
        weights (numpy.ndarray): Model weights initialized during fitting.
        bias (float): Model bias term initialized to 0.0.
        '''
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = 0.0

    def _sigmoid(self, z):
        '''
        Applies the sigmoid activation function.

        Formula:
        sigma(z) = 1 / (1 + exp(-z))

        Parameters:
        z (numpy.ndarray): Linear model output values.

        Returns:
        numpy.ndarray: Probability-like values in the range [0, 1].
        '''
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X, y):
        '''
        Fits Logistic Regression to training data using gradient descent.

        Training loop:
        1) Compute linear score z.
        2) Convert to probabilities p using sigmoid.
        3) Compute gradients from (p - y).
        4) Update weights and bias.

        Parameters:
        X (numpy.ndarray): Training feature matrix.
        y (numpy.ndarray): Binary training labels (0 or 1).
        '''
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features, dtype=float)
        self.bias = 0.0

        for _ in range(self.n_iters):
            linear_pred = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(linear_pred)

            dw = (1.0 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1.0 / n_samples) * np.sum(y_pred - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict_proba(self, X):
        '''
        Predicts class probabilities for input features.

        Formula:
        p = sigmoid(Xw + b)

        Parameters:
        X (numpy.ndarray): Input feature matrix.

        Returns:
        numpy.ndarray: Predicted probabilities for the positive class.
        '''
        linear_pred = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_pred)

    def predict(self, X, threshold=0.5):
        '''
        Predicts binary class labels from probabilities.

        Decision rule:
        - If probability >= threshold, predict 1.
        - Otherwise, predict 0.

        Parameters:
        X (numpy.ndarray): Input feature matrix.
        threshold (float): Cutoff for converting probabilities to class labels.

        Returns:
        numpy.ndarray: Predicted binary labels.
        '''
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)
