# Traditional Machine Learning Algorithms From Scratch

## Overview

This repository is focused on implementing traditional machine learning algorithms from scratch, then comparing their behavior and performance against trusted library implementations.

The purpose is educational and practical:

- Understand how each algorithm works internally.
- Build intuition for training behavior, splits, optimization, and prediction logic.
- Validate custom implementations against mature library versions.
- Create a growing reference project that covers almost all major traditional ML algorithms.

## Core Philosophy

This project follows two parallel tracks for each algorithm:

1. From-scratch implementation
	 - Build the core algorithm logic ourselves.
	 - Avoid using high-level ML training APIs for the model internals.

2. Library comparison
	 - Train the equivalent model from a standard library.
	 - Compare outputs, confusion matrices, and metrics.
	 - Study where and why results differ.

The comparison approach helps ensure correctness and gives deeper insight into model quality, limitations, and optimization opportunities.

## Current Project Structure

- Decision_tree.py
	- Custom Decision Tree implementation built from scratch.
- Linear_regression.py
	- Custom Linear Regression implementation built from scratch using gradient descent.
- Logistic_regression.py
	- Custom Logistic Regression implementation built from scratch using sigmoid + gradient descent.
- KNN_classifier.py
	- Custom K-Nearest Neighbors classifier implementation built from scratch.
- Naive_bayes.py
	- Custom Gaussian Naive Bayes classifier implementation built from scratch.
- Perceptron.py
	- Custom Perceptron binary classifier implementation built from scratch.
- SVM_classifier.py
	- Custom Linear SVM binary classifier implementation built from scratch.
- Random_forest.py
	- Custom Random Forest classifier implementation built from scratch.
- Gradient_boosting.py
	- Custom Gradient Boosting regressor implementation built from scratch.
- AdaBoost.py
	- Custom AdaBoost classifier implementation built from scratch.
- KMeans.py
	- Custom K-Means clustering implementation built from scratch.
- PCA.py
	- Custom Principal Component Analysis (PCA) implementation built from scratch.
- main.py
	- Runs clearly separated experiments.
	- Prints dataset name, task type, tested models, and evaluation metrics.
- README.md
	- Project documentation and roadmap.

## Implemented So Far

### Decision Tree Classifier (Custom)

Implemented features:

- Recursive tree construction.
- Best split search over features and thresholds.
- Gini impurity split scoring.
- Leaf prediction using most common class.
- Sample-by-sample prediction by tree traversal.

### Library Baseline

- DecisionTreeClassifier from scikit-learn is used as a reference baseline.

### Linear Regression (Custom)

Implemented features:

- Weight and bias learning with gradient descent.
- Iterative optimization over mean squared error objective.
- Numeric prediction for continuous targets.

### Library Baseline

- LinearRegression from scikit-learn is used as a reference baseline.

### Logistic Regression (Custom)

Implemented features:

- Sigmoid-based probability prediction.
- Gradient descent optimization for binary classification.
- Threshold-based class prediction.

### Library Baseline

- LogisticRegression from scikit-learn is used as a reference baseline.

### K-Nearest Neighbors (KNN) Classifier (Custom)

Implemented features:

- Euclidean distance-based nearest-neighbor search.
- Majority vote among top-k neighbors.
- Full prediction pipeline for binary or multiclass classification.

### Library Baseline

- KNeighborsClassifier from scikit-learn is used as a reference baseline.

### Gaussian Naive Bayes (Custom)

Implemented features:

- Class-wise mean, variance, and prior estimation.
- Gaussian likelihood computation in log-space for numerical stability.
- Maximum posterior class prediction.

### Library Baseline

- GaussianNB from scikit-learn is used as a reference baseline.

### Perceptron (Custom)

Implemented features:

- Binary linear classifier with iterative weight updates.
- Mistake-driven perceptron learning rule.
- Threshold-based class prediction.

### Library Baseline

- Perceptron from scikit-learn is used as a reference baseline.

## Three Problem Types In This Project

### 1) Classification Problems

Goal: Predict discrete class labels (for example 0 or 1, or multiple classes).

Models that work here:

- Decision Tree (custom and sklearn)
- Logistic Regression (custom and sklearn)
- KNN (custom and sklearn)
- Gaussian Naive Bayes (custom and sklearn)
- Perceptron (custom and sklearn)
- Linear SVM (custom and sklearn)
- Random Forest (custom and sklearn)
- AdaBoost (custom and sklearn)

### 2) Regression Problems

Goal: Predict continuous numeric values.

Models that work here:

- Linear Regression (custom and sklearn)
- Gradient Boosting Regressor (custom and sklearn)
- Future additions: Polynomial Regression, regularized linear models

### 3) Unsupervised Learning Problems

Goal: Discover data structure without target labels.

Models that work here:

- K-Means clustering (custom and sklearn)
- PCA dimensionality reduction (custom and sklearn)

## Current Experiments (Exactly What Is Tested)

### Experiment 1: Classification

- Dataset: Breast Cancer dataset from scikit-learn.
- Task: Binary classification.
- Models compared:
	- Custom DecisionTree (from scratch, max_depth=3).
	- sklearn DecisionTreeClassifier (library baseline).
- Metrics shown:
	- Accuracy
	- Confusion Matrix
	- Classification Report (Precision, Recall, F1)
	- Predicted classes

### Experiment 2: Classification

- Dataset: Breast Cancer dataset from scikit-learn.
- Task: Binary classification.
- Models compared:
	- Custom LogisticRegressionModel (from scratch, gradient descent).
	- sklearn LogisticRegression (library baseline).
- Metrics shown:
	- Accuracy
	- Confusion Matrix
	- Classification Report (Precision, Recall, F1)
	- Predicted classes

### Experiment 3: Regression

- Dataset: Diabetes dataset from scikit-learn.
- Task: Regression.
- Models compared:
	- Custom LinearRegressionModel (from scratch, gradient descent).
	- sklearn LinearRegression (library baseline).
- Metrics shown:
	- MSE
	- RMSE
	- MAE
	- R2 Score

### Experiment 4: Classification

- Dataset: Breast Cancer dataset from scikit-learn.
- Task: Binary classification.
- Models compared:
	- Custom KNNClassifier (from scratch, k=5).
	- sklearn KNeighborsClassifier (library baseline, k=5).
- Metrics shown:
	- Accuracy
	- Confusion Matrix
	- Classification Report (Precision, Recall, F1)
	- Predicted classes

### Experiment 5: Classification

- Dataset: Breast Cancer dataset from scikit-learn.
- Task: Binary classification.
- Models compared:
	- Custom GaussianNaiveBayes (from scratch).
	- sklearn GaussianNB (library baseline).
- Metrics shown:
	- Accuracy
	- Confusion Matrix
	- Classification Report (Precision, Recall, F1)
	- Predicted classes

### Experiment 6: Classification

- Dataset: Breast Cancer dataset from scikit-learn.
- Task: Binary classification.
- Models compared:
	- Custom PerceptronModel (from scratch).
	- sklearn Perceptron (library baseline).
- Metrics shown:
	- Accuracy
	- Confusion Matrix
	- Classification Report (Precision, Recall, F1)
	- Predicted classes

### Experiment 7: Classification

- Dataset: Breast Cancer dataset from scikit-learn.
- Task: Binary classification.
- Models compared:
	- Custom LinearSVM (from scratch, hinge loss).
	- sklearn SVC with linear kernel (library baseline).
- Metrics shown:
	- Accuracy
	- Confusion Matrix
	- Classification Report (Precision, Recall, F1)
	- Predicted classes

### Experiment 8: Classification

- Dataset: Breast Cancer dataset from scikit-learn.
- Task: Binary classification.
- Models compared:
	- Custom RandomForestClassifierScratch (from scratch).
	- sklearn RandomForestClassifier (library baseline).
- Metrics shown:
	- Accuracy
	- Confusion Matrix
	- Classification Report (Precision, Recall, F1)
	- Predicted classes

### Experiment 9: Classification

- Dataset: Breast Cancer dataset from scikit-learn.
- Task: Binary classification.
- Models compared:
	- Custom AdaBoostClassifierScratch (from scratch).
	- sklearn AdaBoostClassifier (library baseline).
- Metrics shown:
	- Accuracy
	- Confusion Matrix
	- Classification Report (Precision, Recall, F1)
	- Predicted classes

### Experiment 10: Regression

- Dataset: Diabetes dataset from scikit-learn.
- Task: Regression.
- Models compared:
	- Custom GradientBoostingRegressorScratch (from scratch).
	- sklearn GradientBoostingRegressor (library baseline).
- Metrics shown:
	- MSE
	- RMSE
	- MAE
	- R2 Score

### Experiment 11: Unsupervised Clustering

- Dataset: Iris dataset from scikit-learn.
- Task: Clustering.
- Models compared:
	- Custom KMeansScratch (from scratch).
	- sklearn KMeans (library baseline).
- Metrics shown:
	- Adjusted Rand Index (vs true labels)
	- Silhouette Score

### Experiment 12: Unsupervised Dimensionality Reduction

- Dataset: Breast Cancer dataset from scikit-learn.
- Task: Feature compression to 2 dimensions.
- Models compared:
	- Custom PCAScratch (from scratch).
	- sklearn PCA (library baseline).
- Metrics shown:
	- Explained Variance Ratio
	- Reconstruction MSE

## Dataset and Evaluation Workflow

The project currently supports running comparisons on real datasets (Breast Cancer for classification and PCA, Diabetes for regression, and Iris for clustering), with a standard train/test split pipeline for supervised tasks.

Evaluation includes classification, regression, clustering, and dimensionality-reduction metrics depending on the experiment.

Each experiment also prints model HYPERPARAMETERS in uppercase so it is always clear which configuration is being tested.

This makes it easy to inspect both overall performance and class-level behavior.

## Hyperparameters and Tuning Importance

Hyperparameters are settings chosen before training (for example `MAX_DEPTH`, `LEARNING_RATE`, `N_ITERS`, `K`, and `MAX_ITER`). They are not learned directly from data.

Why they matter:

- They control model complexity and learning behavior.
- They can reduce underfitting or overfitting.
- They often have a direct impact on accuracy, error metrics, and stability.
- Good tuning can make a large performance difference even with the same algorithm and dataset.

In this project, we always show hyperparameters with experiment results so tuning decisions are transparent and reproducible.

## Why This Repository Matters

Most people can call model.fit quickly, but fewer can explain exactly what happens inside the model.

An AI model can reproduce similar code in seconds. The goal of this repository is different: to understand how each model works internally and why it behaves the way it does on real data.

By building algorithms from scratch, this project develops:

- Strong fundamentals in ML math and logic.
- Better debugging skills for model failure modes.
- Better intuition for bias/variance and underfitting/overfitting.
- Confidence in choosing and tuning algorithms for real tasks.

## Roadmap: Traditional ML Algorithms To Implement

The goal is to implement almost all major traditional ML methods from scratch and compare each one against a library equivalent.

Current implementation status:

- Decision Tree (implemented)
- Linear Regression (implemented)
- Logistic Regression (implemented)
- K-Nearest Neighbors (KNN) (implemented)
- Naive Bayes (implemented)
- Perceptron (implemented)
- Support Vector Machine (SVM) (implemented)
- Random Forest (implemented)
- Gradient Boosting (implemented: regressor)
- AdaBoost (implemented)
- K-Means Clustering (implemented)
- Principal Component Analysis (PCA) (implemented)

Planned next algorithms:

- XGBoost-style boosting variants
- Extra Trees
- DBSCAN
- Hierarchical Clustering
- t-SNE or UMAP for nonlinear dimensionality reduction


## Comparison Strategy Per Algorithm

For each algorithm, this project aims to provide:

1. Clear mathematical idea
2. Custom implementation from scratch
3. Training and prediction workflow
4. Library baseline implementation
5. Side-by-side metric comparison
6. Error analysis and notes on differences

## How To Run

1. Install required packages:

```bash
pip install numpy matplotlib scikit-learn
```

2. Run the comparison script:

```bash
python main.py
```

Or run the desktop GUI:

```bash
python gui.py
```

The GUI lets you:

- Select the model to test
- Choose `custom` or `sklearn` implementation
- Run the selected experiment in one click
- View metrics directly in the app instead of terminal output

3. Review output metrics in the terminal:

- Classification metrics for Decision Tree, Logistic Regression, KNN, Naive Bayes, Perceptron, SVM, Random Forest, and AdaBoost experiments
- Regression metrics for Linear Regression and Gradient Boosting experiments
- Clustering metrics for K-Means experiment
- Dimensionality-reduction metrics for PCA experiment

## Future Improvements

- Add reproducible experiment settings across all algorithms.
- Add timing benchmarks for training and prediction.
- Add cross-validation support.
- Add per-algorithm experiment notebooks or reports.
- Add unit tests to verify from-scratch implementations.
- Add visualizations for decision boundaries and error patterns.

## Contribution Direction

Contributions are welcome for:

- New from-scratch algorithm implementations
- Optimization and refactoring of current algorithms
- Better evaluation and visualization utilities
- Documentation improvements

## Final Goal

Build a complete, high-quality traditional machine learning lab where each major algorithm is:

- implemented from scratch,
- compared against a library version,
- evaluated with consistent metrics,
- and explained clearly for learning and practical understanding.
