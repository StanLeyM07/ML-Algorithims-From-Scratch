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

## Two Problem Types In This Project

### 1) Classification Problems

Goal: Predict discrete class labels (for example 0 or 1, or multiple classes).

Models that work here:

- Decision Tree (custom and sklearn)
- Logistic Regression (custom and sklearn)
- KNN (custom and sklearn)
- Gaussian Naive Bayes (custom and sklearn)
- Perceptron (custom and sklearn)
- Future additions: SVM, Random Forest, Gradient Boosting

### 2) Regression Problems

Goal: Predict continuous numeric values.

Models that work here:

- Linear Regression (custom and sklearn)
- Future additions: Polynomial Regression, regularized linear models, tree-based regressors

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

## Dataset and Evaluation Workflow

The project currently supports running comparisons on real datasets (Breast Cancer for classification and Diabetes for regression), with a standard train/test split pipeline.

Evaluation includes classification and regression metrics depending on the experiment.

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

Planned algorithms include:

- Linear Regression (implemented)
- Logistic Regression (implemented)
- K-Nearest Neighbors (KNN) (implemented)
- Naive Bayes (implemented)
- Perceptron (implemented)
- Support Vector Machine (SVM)
- Decision Tree (in progress and improvements ongoing)
- Random Forest
- Gradient Boosting
- AdaBoost
- K-Means Clustering
- Principal Component Analysis (PCA)


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

3. Review output metrics in the terminal:

- Classification metrics for Decision Tree, Logistic Regression, KNN, Naive Bayes, and Perceptron experiments
- Regression metrics for Linear Regression experiment

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
