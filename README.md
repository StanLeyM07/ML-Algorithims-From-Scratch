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
- main.py
	- Dataset loading, train/test split, model training, and evaluation.
	- Compares custom models with library-based versions.
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

## Dataset and Evaluation Workflow

The project currently supports running comparisons on real datasets (for example Breast Cancer and Digits in previous runs), with a standard train/test split pipeline.

Evaluation includes:

- Accuracy
- Confusion Matrix
- Classification Report (Precision, Recall, F1)
- Predicted class coverage

This makes it easy to inspect both overall performance and class-level behavior.

## Why This Repository Matters

Most people can call model.fit quickly, but fewer can explain exactly what happens inside the model.

By building algorithms from scratch, this project develops:

- Strong fundamentals in ML math and logic.
- Better debugging skills for model failure modes.
- Better intuition for bias/variance and underfitting/overfitting.
- Confidence in choosing and tuning algorithms for real tasks.

## Roadmap: Traditional ML Algorithms To Implement

The goal is to implement almost all major traditional ML methods from scratch and compare each one against a library equivalent.

Planned algorithms include:

- Linear Regression
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Naive Bayes
- Support Vector Machine (SVM)
- Decision Tree (in progress and improvements ongoing)
- Random Forest
- Gradient Boosting
- AdaBoost
- K-Means Clustering
- Hierarchical Clustering
- Principal Component Analysis (PCA)
- DBSCAN
- Perceptron

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

- Accuracy scores
- Confusion matrices
- Classification reports

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

Build a complete, high-quality traditional machine learning learning lab where each major algorithm is:

- implemented from scratch,
- compared against a library version,
- evaluated with consistent metrics,
- and explained clearly for learning and practical understanding.
