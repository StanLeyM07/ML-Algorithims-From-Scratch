import numpy as np
'''
AdaBoost binary classification implementation from scratch.

Beginner intuition:
- Train weak classifiers (decision stumps) sequentially.
- Increase weight on samples previous stumps got wrong.
- Final prediction is a weighted vote of all stumps.
'''


class _DecisionStump:
    def __init__(self):
        self.polarity = 1
        self.feature_idx = None
        self.threshold = None
        self.alpha = 0.0

    def predict(self, X):
        n_samples = X.shape[0]
        predictions = np.ones(n_samples)
        feature_values = X[:, self.feature_idx]

        if self.polarity == 1:
            predictions[feature_values < self.threshold] = -1
        else:
            predictions[feature_values >= self.threshold] = -1

        return predictions


class AdaBoostClassifierScratch:
    def __init__(self, n_learners=50):
        '''
        Parameters:
        n_learners (int): Number of decision stumps.
        '''
        self.n_learners = n_learners
        self.learners = []

    def fit(self, X, y):
        y_transformed = np.where(y <= 0, -1, 1)
        n_samples, n_features = X.shape

        w = np.full(n_samples, 1.0 / n_samples)
        self.learners = []

        for _ in range(self.n_learners):
            stump = _DecisionStump()
            min_error = float('inf')

            for feature_idx in range(n_features):
                feature_values = X[:, feature_idx]
                thresholds = np.unique(feature_values)

                for threshold in thresholds:
                    polarity = 1
                    predictions = np.ones(n_samples)
                    predictions[feature_values < threshold] = -1

                    error = np.sum(w[y_transformed != predictions])

                    if error > 0.5:
                        error = 1.0 - error
                        polarity = -1

                    if error < min_error:
                        min_error = error
                        stump.polarity = polarity
                        stump.threshold = threshold
                        stump.feature_idx = feature_idx

            eps = 1e-10
            stump.alpha = 0.5 * \
                np.log((1.0 - min_error + eps) / (min_error + eps))

            predictions = stump.predict(X)
            w *= np.exp(-stump.alpha * y_transformed * predictions)
            w /= np.sum(w)

            self.learners.append(stump)

    def predict(self, X):
        learner_preds = np.array(
            [learner.alpha * learner.predict(X) for learner in self.learners])
        y_pred = np.sign(np.sum(learner_preds, axis=0))
        return np.where(y_pred <= 0, 0, 1)
