import numpy as np
'''
Gradient Boosting Regressor implementation from scratch.

Beginner intuition:
- Start with a simple baseline prediction (mean of targets).
- Repeatedly fit weak learners to residual errors.
- Add each learner's prediction with a learning-rate shrinkage factor.
'''


class _RegressionTreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


class _RegressionTree:
    def __init__(self, max_depth=2, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y):
        self.root = self._grow_tree(X, y, depth=0)

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _grow_tree(self, X, y, depth):
        n_samples, n_features = X.shape

        if depth >= self.max_depth or n_samples < self.min_samples_split:
            return _RegressionTreeNode(value=float(np.mean(y)))

        best_feature, best_threshold = self._best_split(X, y)
        if best_feature is None:
            return _RegressionTreeNode(value=float(np.mean(y)))

        left_idxs = X[:, best_feature] < best_threshold
        right_idxs = ~left_idxs

        if np.sum(left_idxs) == 0 or np.sum(right_idxs) == 0:
            return _RegressionTreeNode(value=float(np.mean(y)))

        left = self._grow_tree(X[left_idxs], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs], y[right_idxs], depth + 1)

        return _RegressionTreeNode(feature=best_feature, threshold=best_threshold, left=left, right=right)

    def _best_split(self, X, y):
        best_mse = float('inf')
        split_feature, split_threshold = None, None

        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_idxs = X[:, feature] < threshold
                right_idxs = ~left_idxs

                if np.sum(left_idxs) == 0 or np.sum(right_idxs) == 0:
                    continue

                left_y = y[left_idxs]
                right_y = y[right_idxs]

                left_mse = np.mean((left_y - np.mean(left_y)) ** 2)
                right_mse = np.mean((right_y - np.mean(right_y)) ** 2)

                weighted_mse = (len(left_y) * left_mse +
                                len(right_y) * right_mse) / len(y)

                if weighted_mse < best_mse:
                    best_mse = weighted_mse
                    split_feature = feature
                    split_threshold = threshold

        return split_feature, split_threshold

    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature] < node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)


class GradientBoostingRegressorScratch:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=2):
        '''
        Parameters:
        n_estimators (int): Number of weak trees.
        learning_rate (float): Shrinkage applied to each tree contribution.
        max_depth (int): Depth of each weak tree.
        '''
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.base_prediction = 0.0
        self.trees = []

    def fit(self, X, y):
        self.base_prediction = float(np.mean(y))
        current_pred = np.full_like(y, self.base_prediction, dtype=float)
        self.trees = []

        for _ in range(self.n_estimators):
            residuals = y - current_pred
            tree = _RegressionTree(max_depth=self.max_depth)
            tree.fit(X, residuals)
            update = tree.predict(X)
            current_pred += self.learning_rate * update
            self.trees.append(tree)

    def predict(self, X):
        y_pred = np.full(X.shape[0], self.base_prediction, dtype=float)
        for tree in self.trees:
            y_pred += self.learning_rate * tree.predict(X)
        return y_pred
