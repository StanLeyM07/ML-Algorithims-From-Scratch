from Decision_tree import DecisionTree
from KNN_classifier import KNNClassifier
from Linear_regression import LinearRegressionModel
from Logistic_regression import LogisticRegressionModel
from Naive_bayes import GaussianNaiveBayes
from Perceptron import PerceptronModel
import inspect
import numpy as np
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.linear_model import LinearRegression, LogisticRegression, Perceptron as SklearnPerceptron
from sklearn.metrics import (
	accuracy_score,
	classification_report,
	confusion_matrix,
	mean_absolute_error,
	mean_squared_error,
	r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


def get_model_hyperparameters(model):
	if hasattr(model, "get_params"):
		return model.get_params(deep=False)

	params = {}
	signature = inspect.signature(model.__class__.__init__)
	for name in signature.parameters:
		if name == "self":
			continue
		if hasattr(model, name):
			params[name] = getattr(model, name)
	return params


def print_hyperparameters(model):
	params = get_model_hyperparameters(model)
	print("HYPERPARAMETERS:")
	if not params:
		print("- NONE")
		return

	for key in sorted(params.keys()):
		print(f"- {key.upper()}: {params[key]}")


def evaluate_classification_model(model_name, model, X_train, X_test, y_train, y_test):
	print(f"\nModel: {model_name}")
	print_hyperparameters(model)

	model.fit(X_train, y_train)
	y_pred = model.predict(X_test)

	print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
	print("Confusion Matrix:")
	print(confusion_matrix(y_test, y_pred))
	print("Classification Report:")
	print(classification_report(y_test, y_pred, zero_division=0))
	print(f"Predicted classes: {sorted(np.unique(y_pred).tolist())}")


def evaluate_regression_model(model_name, model, X_train, X_test, y_train, y_test):
	print(f"\nModel: {model_name}")
	print_hyperparameters(model)

	model.fit(X_train, y_train)
	y_pred = model.predict(X_test)

	mse = mean_squared_error(y_test, y_pred)
	rmse = np.sqrt(mse)
	mae = mean_absolute_error(y_test, y_pred)
	r2 = r2_score(y_test, y_pred)

	print(f"MSE: {mse:.4f}")
	print(f"RMSE: {rmse:.4f}")
	print(f"MAE: {mae:.4f}")
	print(f"R2 Score: {r2:.4f}")


def run_decision_tree_experiment():
	dataset = load_breast_cancer()
	X = dataset.data
	y = dataset.target

	X_train, X_test, y_train, y_test = train_test_split(
		X,
		y,
		test_size=0.2,
		random_state=42,
		stratify=y,
	)

	print("\n=== Experiment 1: Decision Tree Classification ===")
	print("Dataset: Breast Cancer (Binary Classification)")

	evaluate_classification_model(
		"Custom DecisionTree (from scratch, max_depth=3)",
		DecisionTree(max_depth=3),
		X_train,
		X_test,
		y_train,
		y_test,
	)

	evaluate_classification_model(
		"sklearn DecisionTreeClassifier (library baseline)",
		DecisionTreeClassifier(random_state=42),
		X_train,
		X_test,
		y_train,
		y_test,
	)


def run_logistic_regression_experiment():
	dataset = load_breast_cancer()
	X = dataset.data
	y = dataset.target

	X_train, X_test, y_train, y_test = train_test_split(
		X,
		y,
		test_size=0.2,
		random_state=42,
		stratify=y,
	)

	# Logistic regression benefits strongly from feature scaling.
	scaler = StandardScaler()
	X_train_scaled = scaler.fit_transform(X_train)
	X_test_scaled = scaler.transform(X_test)

	print("\n=== Experiment 2: Logistic Regression Classification ===")
	print("Dataset: Breast Cancer (Binary Classification)")

	evaluate_classification_model(
		"Custom LogisticRegressionModel (from scratch, gradient descent)",
		LogisticRegressionModel(learning_rate=0.05, n_iters=5000),
		X_train_scaled,
		X_test_scaled,
		y_train,
		y_test,
	)

	evaluate_classification_model(
		"sklearn LogisticRegression (library baseline)",
		LogisticRegression(max_iter=5000, random_state=42),
		X_train_scaled,
		X_test_scaled,
		y_train,
		y_test,
	)


def run_linear_regression_experiment():
	dataset = load_diabetes()
	X = dataset.data
	y = dataset.target

	X_train, X_test, y_train, y_test = train_test_split(
		X,
		y,
		test_size=0.2,
		random_state=42,
	)

	print("\n=== Experiment 3: Linear Regression ===")
	print("Dataset: Diabetes (Regression)")

	evaluate_regression_model(
		"Custom LinearRegressionModel (gradient descent)",
		LinearRegressionModel(learning_rate=0.1, n_iters=3000),
		X_train,
		X_test,
		y_train,
		y_test,
	)

	evaluate_regression_model(
		"sklearn LinearRegression (library baseline)",
		LinearRegression(),
		X_train,
		X_test,
		y_train,
		y_test,
	)


def run_knn_experiment():
	dataset = load_breast_cancer()
	X = dataset.data
	y = dataset.target

	X_train, X_test, y_train, y_test = train_test_split(
		X,
		y,
		test_size=0.2,
		random_state=42,
		stratify=y,
	)

	# Distance-based methods require scaling for fair nearest-neighbor distance comparison.
	scaler = StandardScaler()
	X_train_scaled = scaler.fit_transform(X_train)
	X_test_scaled = scaler.transform(X_test)

	print("\n=== Experiment 4: KNN Classification ===")
	print("Dataset: Breast Cancer (Binary Classification)")

	evaluate_classification_model(
		"Custom KNNClassifier (from scratch, k=5)",
		KNNClassifier(k=5),
		X_train_scaled,
		X_test_scaled,
		y_train,
		y_test,
	)

	evaluate_classification_model(
		"sklearn KNeighborsClassifier (library baseline, k=5)",
		KNeighborsClassifier(n_neighbors=5),
		X_train_scaled,
		X_test_scaled,
		y_train,
		y_test,
	)


def run_naive_bayes_experiment():
	dataset = load_breast_cancer()
	X = dataset.data
	y = dataset.target

	X_train, X_test, y_train, y_test = train_test_split(
		X,
		y,
		test_size=0.2,
		random_state=42,
		stratify=y,
	)

	print("\n=== Experiment 5: Naive Bayes Classification ===")
	print("Dataset: Breast Cancer (Binary Classification)")

	evaluate_classification_model(
		"Custom GaussianNaiveBayes (from scratch)",
		GaussianNaiveBayes(),
		X_train,
		X_test,
		y_train,
		y_test,
	)

	evaluate_classification_model(
		"sklearn GaussianNB (library baseline)",
		GaussianNB(),
		X_train,
		X_test,
		y_train,
		y_test,
	)


def run_perceptron_experiment():
	dataset = load_breast_cancer()
	X = dataset.data
	y = dataset.target

	X_train, X_test, y_train, y_test = train_test_split(
		X,
		y,
		test_size=0.2,
		random_state=42,
		stratify=y,
	)

	# Perceptron converges better when features are standardized.
	scaler = StandardScaler()
	X_train_scaled = scaler.fit_transform(X_train)
	X_test_scaled = scaler.transform(X_test)

	print("\n=== Experiment 6: Perceptron Classification ===")
	print("Dataset: Breast Cancer (Binary Classification)")

	evaluate_classification_model(
		"Custom PerceptronModel (from scratch)",
		PerceptronModel(learning_rate=0.01, n_iters=2000),
		X_train_scaled,
		X_test_scaled,
		y_train,
		y_test,
	)

	evaluate_classification_model(
		"sklearn Perceptron (library baseline)",
		SklearnPerceptron(max_iter=2000, random_state=42),
		X_train_scaled,
		X_test_scaled,
		y_train,
		y_test,
	)


def main():
	print("=== Traditional ML: From-Scratch vs Library Comparison ===")
	print("Problem Types Covered: Classification and Regression")
	run_decision_tree_experiment()
	run_logistic_regression_experiment()
	run_linear_regression_experiment()
	run_knn_experiment()
	run_naive_bayes_experiment()
	run_perceptron_experiment()


if __name__ == "__main__":
	main()
