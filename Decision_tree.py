import numpy as np
'''
Decision Tree (classification) implementation from scratch.

Beginner intuition:
- A tree asks a sequence of yes/no questions about features.
- Each split tries to make child groups as "pure" as possible.
- Purity here is measured with Gini impurity.

Key formulas used:
- Gini(node) = 1 - sum(p_k^2)
    where p_k is the fraction of class k in that node.
- Weighted split impurity:
    Gini_split = (n_left / n_total) * Gini(left) + (n_right / n_total) * Gini(right)

This implementation is for educational purposes and can be improved further
for speed and robustness.
'''
class DecisionTree:

    def __init__(self, max_depth=None):
        '''
        Initializes the Decision Tree.

        Parameters:
        max_depth (int): The maximum depth of the tree. If None, the tree will
                         grow until leaves become pure or no useful split remains.
        '''
        self.max_depth = max_depth 
        self.tree = None

    def fit(self, X, y):
        '''
        Fits the Decision Tree to the training data.

        Parameters:
        X (numpy.ndarray): The training data features.
        y (numpy.ndarray): The training data labels.
        '''
        self.tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        '''
        Recursively builds the Decision Tree.

        Stopping conditions:
        1) Maximum depth reached.
        2) Node is already pure (all labels are the same).
        3) No valid split can reduce impurity.

        Parameters: 
        X (numpy.ndarray): The training data features.
        y (numpy.ndarray): The training data labels.
        depth (int): The current depth of the tree.

        Returns:
        The root node of the built subtree.
        '''
        if depth == self.max_depth or len(set(y)) == 1:
            return self._most_common_label(y)
        best_feature, best_threshold = self._best_split(X, y)
        if best_feature is None:
            return self._most_common_label(y)
        left_indices = X[:, best_feature] < best_threshold
        right_indices = X[:, best_feature] >= best_threshold
        left_subtree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self._build_tree(X[right_indices], y[right_indices], depth + 1)
        return (best_feature, best_threshold, left_subtree, right_subtree)
    def _best_split(self, X, y):
        '''
        Finds the best feature and threshold to split the data.

        For each feature and candidate threshold, we compute:
        Gini_split = (n_left / n_total) * Gini(left) + (n_right / n_total) * Gini(right)
        and select the split with the smallest value.

        Parameters:
        X (numpy.ndarray): The training data features.
        y (numpy.ndarray): The training data labels.

        Returns:
        best_feature (int): The index of the best feature to split on.
        best_threshold (float): The threshold value for the best split.
        '''
        num_features = X.shape[1]
        best_gini = float('inf')
        best_feature, best_threshold = None, None
        for feature in range(num_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_indices = X[:, feature] < threshold
                right_indices = X[:, feature] >= threshold
                if len(y[left_indices]) == 0 or len(y[right_indices]) == 0:
                    continue
                gini = self._gini(y[left_indices], y[right_indices])
                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature
                    best_threshold = threshold
        return best_feature, best_threshold
    
    def _gini(self, left_labels, right_labels):
        '''
        Calculates the Gini impurity of a split.

        Formula per node:
        Gini(node) = 1 - sum(p_k^2)

        Formula for a split:
        Gini_split = (n_left * Gini(left) + n_right * Gini(right)) / n_total

        Parameters:
        left_labels (numpy.ndarray): The labels of the left split.
        right_labels (numpy.ndarray): The labels of the right split.

        Returns:
        gini (float): The Gini impurity of the split.
        '''
        total_samples = len(left_labels) + len(right_labels)
        if total_samples == 0:
            return 0
        left_gini = 1 - sum((np.sum(left_labels == c) / len(left_labels)) ** 2 for c in np.unique(left_labels))
        right_gini = 1 - sum((np.sum(right_labels == c) / len(right_labels)) ** 2 for c in np.unique(right_labels))
        return (len(left_labels) * left_gini + len(right_labels) * right_gini) / total_samples
    
    def _most_common_label(self, y):
        '''
        Finds the most common label in a set of labels.

        Parameters:
        y (numpy.ndarray): The labels to analyze.

        Returns:
        most_common (int): The most common label.
        '''
        return np.bincount(y).argmax()

    def predict(self, X):
        '''
        Predicts the labels for the given input data.

        Parameters:
        X (numpy.ndarray): The input data features.

        Returns:
        predictions (numpy.ndarray): The predicted labels.
        '''
        return np.array([self._predict_single(x, self.tree) for x in X])

    def _predict_single(self, x, node):
        '''
        Predicts one sample by traversing split rules from root to leaf.

        Traversal rule:
        - If x[feature] < threshold, go left.
        - Else, go right.
        - At leaf, return stored class label.

        Parameters:
        x (numpy.ndarray): The input sample.
        node: The current node in the tree.

        Returns:
        The predicted label.
        '''
        if not isinstance(node, tuple):
            return node
        feature, threshold, left_subtree, right_subtree = node
        if x[feature] < threshold:
            return self._predict_single(x, left_subtree)
        else:
            return self._predict_single(x, right_subtree)
        