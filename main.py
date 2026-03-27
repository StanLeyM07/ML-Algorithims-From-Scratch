from Decision_tree import DecisionTree
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
	model.fit(X_train, y_train)
	y_pred = model.predict(X_test)
	accuracy = accuracy_score(y_test, y_pred)
	conf_matrix = confusion_matrix(y_test, y_pred)

	print(f"\n{name}")
	print(f"Accuracy: {accuracy:.4f}")
	print("Confusion Matrix:")
	print(conf_matrix)
	print("Classification Report:")
	print(classification_report(y_test, y_pred, zero_division=0))
	print(f"Predicted classes: {sorted(set(y_pred.tolist()))}")


def main(show_plot=False):
	cancer = load_breast_cancer()
	X = cancer.data
	y = cancer.target

	X_train, X_test, y_train, y_test = train_test_split(
		X,
		y,
		test_size=0.2,
		random_state=42,
		stratify=y,
	)

	if show_plot:
		plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=20)
		plt.xlabel(cancer.feature_names[0])
		plt.ylabel(cancer.feature_names[1])
		plt.title("Breast Cancer Dataset (Binary Classification)")
		plt.show()

	custom_tree = DecisionTree(max_depth=3)
	sklearn_tree_full = DecisionTreeClassifier(random_state=42)

	print("=== Model Comparison on Breast Cancer Dataset ===")
	evaluate_model("Custom DecisionTree", custom_tree, X_train, X_test, y_train, y_test)
	evaluate_model("sklearn DecisionTreeClassifier", sklearn_tree_full, X_train, X_test, y_train, y_test)


if __name__ == "__main__":
	main(show_plot=False)
