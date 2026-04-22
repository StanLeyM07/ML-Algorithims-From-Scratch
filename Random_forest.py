import numpy as np
'''
Random Forest classification implementation from scratch.

Beginner intuition:
- Build many decision trees on bootstrap samples.
- Each split tests only a random subset of features.
- Final prediction is majority vote across trees.
'''


class _TreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


class _DecisionTreeClassifier:
    def __init__(self, max_depth=8, min_samples_split=2, n_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.root = None

    def fit(self, X, y):
        self.root = self._grow_tree(X, y, depth=0)

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _grow_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        if (
            depth >= self.max_depth
            or n_labels == 1
            or n_samples < self.min_samples_split
        ):
            return _TreeNode(value=self._most_common_label(y))

        feat_idxs = np.random.choice(
            n_features,
            self.n_features if self.n_features is not None else n_features,
            replace=False,
        )

        best_feature, best_threshold = self._best_split(X, y, feat_idxs)

        if best_feature is None:
            return _TreeNode(value=self._most_common_label(y))

        left_idxs = X[:, best_feature] < best_threshold
        right_idxs = ~left_idxs

        if np.sum(left_idxs) == 0 or np.sum(right_idxs) == 0:
            return _TreeNode(value=self._most_common_label(y))

        left = self._grow_tree(X[left_idxs], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs], y[right_idxs], depth + 1)

        return _TreeNode(feature=best_feature, threshold=best_threshold, left=left, right=right)

    def _best_split(self, X, y, feat_idxs):
        best_gain = -1.0
        split_idx, split_threshold = None, None

        for feat_idx in feat_idxs:
            thresholds = np.unique(X[:, feat_idx])
            for threshold in thresholds:
                gain = self._information_gain(y, X[:, feat_idx], threshold)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = threshold

        return split_idx, split_threshold

    def _information_gain(self, y, feature_values, threshold):
        parent_gini = self._gini(y)

        left_idxs = feature_values < threshold
        right_idxs = ~left_idxs

        if np.sum(left_idxs) == 0 or np.sum(right_idxs) == 0:
            return 0.0

        n = len(y)
        n_left = np.sum(left_idxs)
        n_right = np.sum(right_idxs)

        child_gini = (
            n_left / n) * self._gini(y[left_idxs]) + (n_right / n) * self._gini(y[right_idxs])
        return parent_gini - child_gini

    def _gini(self, y):
        proportions = np.bincount(y) / len(y)
        return 1.0 - np.sum(proportions ** 2)

    def _most_common_label(self, y):
        return np.bincount(y).argmax()

    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature] < node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)


class RandomForestClassifierScratch:
    def __init__(self, n_trees=20, max_depth=8, min_samples_split=2, max_features='sqrt'):
        '''
        Parameters:
        n_trees (int): Number of trees in the forest.
        max_depth (int): Maximum depth for each tree.
        min_samples_split (int): Minimum samples needed to split.
        max_features (str or int): Feature subset size per split.
        '''
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []

    def _resolve_n_features(self, total_features):
        if isinstance(self.max_features, int):
            return max(1, min(self.max_features, total_features))
        if self.max_features == 'sqrt':
            return max(1, int(np.sqrt(total_features)))
        if self.max_features == 'log2':
            return max(1, int(np.log2(total_features)))
        return total_features

    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[idxs], y[idxs]

    def fit(self, X, y):
        self.trees = []
        n_features = self._resolve_n_features(X.shape[1])

        for _ in range(self.n_trees):
            tree = _DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                n_features=n_features,
            )
            X_sample, y_sample = self._bootstrap_sample(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        return np.array([np.bincount(sample_preds).argmax() for sample_preds in tree_preds])
