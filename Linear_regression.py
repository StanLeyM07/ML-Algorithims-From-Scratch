import numpy as np
'''
Linear Regression implementation from scratch using Gradient Descent.

Beginner intuition:
- The model draws a best-fit hyperplane through data.
- Prediction is a weighted sum of inputs plus bias.

Core formulas:
- Prediction: y_hat = Xw + b
- Loss (MSE): J = (1/n) * sum((y_hat - y)^2)
- Gradients:
    dJ/dw = (1/n) * X^T * (y_hat - y)
    dJ/db = (1/n) * sum(y_hat - y)
- Update rule:
    w = w - lr * dJ/dw
    b = b - lr * dJ/db

This implementation is educational and can be optimized further.
'''

class LinearRegressionModel:

    def __init__(self, learning_rate=0.01, n_iters=2000):
        '''
        Initializes the Linear Regression model.

        Parameters:
        learning_rate (float): The step size for updating the weights during gradient descent.
        n_iters (int): The number of iterations for the gradient descent optimization.
        weights (numpy.ndarray): The weights of the linear model, initialized to None.
        bias (float): The bias term of the linear model, initialized to 0.0.
        '''
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = 0.0

    def fit(self, X, y):
        '''
        Fits the Linear Regression model to the training data using gradient descent.

        The method repeatedly computes predictions, evaluates gradients of MSE,
        and updates parameters to reduce prediction error.

        Parameters:
        X (numpy.ndarray): The training data features.
        y (numpy.ndarray): The training data target values.
        '''
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features, dtype=float)
        self.bias = 0.0

        for _ in range(self.n_iters):
            y_pred = np.dot(X, self.weights) + self.bias

            dw = (1.0 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1.0 / n_samples) * np.sum(y_pred - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        '''
        Predicts the target values for the given input features using the learned weights and bias.

        Formula:
        y_hat = Xw + b

        Parameters:
        X (numpy.ndarray): The input data features for which to make predictions.

        Returns:
        numpy.ndarray: The predicted target values.
        '''
        return np.dot(X, self.weights) + self.bias
