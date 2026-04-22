import inspect
import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText

import numpy as np
from sklearn.cluster import KMeans as SklearnKMeans
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, Perceptron as SklearnPerceptron
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from AdaBoost import AdaBoostClassifierScratch
from Decision_tree import DecisionTree
from Gradient_boosting import GradientBoostingRegressorScratch
from KMeans import KMeansScratch
from KNN_classifier import KNNClassifier
from Linear_regression import LinearRegressionModel
from Logistic_regression import LogisticRegressionModel
from Naive_bayes import GaussianNaiveBayes
from PCA import PCAScratch
from Perceptron import PerceptronModel
from Random_forest import RandomForestClassifierScratch
from SVM_classifier import LinearSVM


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


def format_hyperparameters(model):
    params = get_model_hyperparameters(model)
    if not params:
        return "- NONE"
    lines = []
    for key in sorted(params.keys()):
        lines.append(f"- {key.upper()}: {params[key]}")
    return "\n".join(lines)


def evaluate_classification(model_name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    lines = []
    lines.append(f"Model: {model_name}")
    lines.append("HYPERPARAMETERS:")
    lines.append(format_hyperparameters(model))
    lines.append(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    lines.append("Confusion Matrix:")
    lines.append(str(confusion_matrix(y_test, y_pred)))
    lines.append("Classification Report:")
    lines.append(classification_report(y_test, y_pred, zero_division=0))
    lines.append(f"Predicted classes: {sorted(np.unique(y_pred).tolist())}")
    return "\n".join(lines)


def evaluate_regression(model_name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    lines = []
    lines.append(f"Model: {model_name}")
    lines.append("HYPERPARAMETERS:")
    lines.append(format_hyperparameters(model))
    lines.append(f"MSE: {mse:.4f}")
    lines.append(f"RMSE: {rmse:.4f}")
    lines.append(f"MAE: {mae:.4f}")
    lines.append(f"R2 Score: {r2:.4f}")
    return "\n".join(lines)


def run_decision_tree(implementation):
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data,
        data.target,
        test_size=0.2,
        random_state=42,
        stratify=data.target,
    )

    if implementation == "custom":
        model = DecisionTree(max_depth=3)
        name = "Custom DecisionTree (from scratch, max_depth=3)"
    else:
        model = DecisionTreeClassifier(random_state=42)
        name = "sklearn DecisionTreeClassifier"

    return (
        "=== Decision Tree Classification ===\n"
        "Dataset: Breast Cancer (Binary Classification)\n\n"
        + evaluate_classification(name, model, X_train,
                                  X_test, y_train, y_test)
    )


def run_logistic_regression(implementation):
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data,
        data.target,
        test_size=0.2,
        random_state=42,
        stratify=data.target,
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    if implementation == "custom":
        model = LogisticRegressionModel(learning_rate=0.05, n_iters=5000)
        name = "Custom LogisticRegressionModel"
    else:
        model = LogisticRegression(max_iter=5000, random_state=42)
        name = "sklearn LogisticRegression"

    return (
        "=== Logistic Regression Classification ===\n"
        "Dataset: Breast Cancer (Binary Classification)\n\n"
        + evaluate_classification(name, model, X_train,
                                  X_test, y_train, y_test)
    )


def run_linear_regression(implementation):
    data = load_diabetes()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data,
        data.target,
        test_size=0.2,
        random_state=42,
    )

    if implementation == "custom":
        model = LinearRegressionModel(learning_rate=0.1, n_iters=3000)
        name = "Custom LinearRegressionModel"
    else:
        model = LinearRegression()
        name = "sklearn LinearRegression"

    return (
        "=== Linear Regression ===\n"
        "Dataset: Diabetes (Regression)\n\n"
        + evaluate_regression(name, model, X_train, X_test, y_train, y_test)
    )


def run_knn(implementation):
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data,
        data.target,
        test_size=0.2,
        random_state=42,
        stratify=data.target,
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    if implementation == "custom":
        model = KNNClassifier(k=5)
        name = "Custom KNNClassifier (k=5)"
    else:
        model = KNeighborsClassifier(n_neighbors=5)
        name = "sklearn KNeighborsClassifier (k=5)"

    return (
        "=== KNN Classification ===\n"
        "Dataset: Breast Cancer (Binary Classification)\n\n"
        + evaluate_classification(name, model, X_train,
                                  X_test, y_train, y_test)
    )


def run_naive_bayes(implementation):
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data,
        data.target,
        test_size=0.2,
        random_state=42,
        stratify=data.target,
    )

    if implementation == "custom":
        model = GaussianNaiveBayes()
        name = "Custom GaussianNaiveBayes"
    else:
        model = GaussianNB()
        name = "sklearn GaussianNB"

    return (
        "=== Naive Bayes Classification ===\n"
        "Dataset: Breast Cancer (Binary Classification)\n\n"
        + evaluate_classification(name, model, X_train,
                                  X_test, y_train, y_test)
    )


def run_perceptron(implementation):
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data,
        data.target,
        test_size=0.2,
        random_state=42,
        stratify=data.target,
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    if implementation == "custom":
        model = PerceptronModel(learning_rate=0.01, n_iters=2000)
        name = "Custom PerceptronModel"
    else:
        model = SklearnPerceptron(max_iter=2000, random_state=42)
        name = "sklearn Perceptron"

    return (
        "=== Perceptron Classification ===\n"
        "Dataset: Breast Cancer (Binary Classification)\n\n"
        + evaluate_classification(name, model, X_train,
                                  X_test, y_train, y_test)
    )


def run_svm(implementation):
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data,
        data.target,
        test_size=0.2,
        random_state=42,
        stratify=data.target,
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    if implementation == "custom":
        model = LinearSVM(learning_rate=0.0005,
                          lambda_param=0.01, n_iters=1500, C=1.0)
        name = "Custom LinearSVM"
    else:
        model = SVC(kernel="linear", random_state=42)
        name = "sklearn SVC (linear kernel)"

    return (
        "=== SVM Classification ===\n"
        "Dataset: Breast Cancer (Binary Classification)\n\n"
        + evaluate_classification(name, model, X_train,
                                  X_test, y_train, y_test)
    )


def run_random_forest(implementation):
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data,
        data.target,
        test_size=0.2,
        random_state=42,
        stratify=data.target,
    )

    if implementation == "custom":
        model = RandomForestClassifierScratch(
            n_trees=25,
            max_depth=8,
            min_samples_split=2,
            max_features="sqrt",
        )
        name = "Custom RandomForestClassifierScratch"
    else:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        name = "sklearn RandomForestClassifier"

    return (
        "=== Random Forest Classification ===\n"
        "Dataset: Breast Cancer (Binary Classification)\n\n"
        + evaluate_classification(name, model, X_train,
                                  X_test, y_train, y_test)
    )


def run_adaboost(implementation):
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data,
        data.target,
        test_size=0.2,
        random_state=42,
        stratify=data.target,
    )

    if implementation == "custom":
        model = AdaBoostClassifierScratch(n_learners=60)
        name = "Custom AdaBoostClassifierScratch"
    else:
        model = AdaBoostClassifier(n_estimators=100, random_state=42)
        name = "sklearn AdaBoostClassifier"

    return (
        "=== AdaBoost Classification ===\n"
        "Dataset: Breast Cancer (Binary Classification)\n\n"
        + evaluate_classification(name, model, X_train,
                                  X_test, y_train, y_test)
    )


def run_gradient_boosting(implementation):
    data = load_diabetes()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data,
        data.target,
        test_size=0.2,
        random_state=42,
    )

    if implementation == "custom":
        model = GradientBoostingRegressorScratch(
            n_estimators=120, learning_rate=0.05, max_depth=2)
        name = "Custom GradientBoostingRegressorScratch"
    else:
        model = GradientBoostingRegressor(random_state=42)
        name = "sklearn GradientBoostingRegressor"

    return (
        "=== Gradient Boosting Regression ===\n"
        "Dataset: Diabetes (Regression)\n\n"
        + evaluate_regression(name, model, X_train, X_test, y_train, y_test)
    )


def run_kmeans(implementation):
    data = load_iris()
    X = StandardScaler().fit_transform(data.data)

    if implementation == "custom":
        model = KMeansScratch(n_clusters=3, max_iters=300,
                              tol=1e-4, random_state=42)
        labels = model.fit_predict(X)
        name = "Custom KMeansScratch"
    else:
        model = SklearnKMeans(n_clusters=3, random_state=42, n_init=10)
        labels = model.fit_predict(X)
        name = "sklearn KMeans"

    lines = []
    lines.append("=== K-Means Clustering ===")
    lines.append("Dataset: Iris (Unsupervised Clustering)")
    lines.append("")
    lines.append(f"Model: {name}")
    lines.append("HYPERPARAMETERS:")
    lines.append(format_hyperparameters(model))
    lines.append(
        f"Adjusted Rand Index vs true labels: {adjusted_rand_score(data.target, labels):.4f}")
    lines.append(f"Silhouette Score: {silhouette_score(X, labels):.4f}")
    return "\n".join(lines)


def run_pca(implementation):
    X = load_breast_cancer().data
    X = StandardScaler().fit_transform(X)

    if implementation == "custom":
        model = PCAScratch(n_components=2)
        reduced = model.fit_transform(X)
        reconstructed = model.inverse_transform(reduced)
        explained_ratio = model.explained_variance_ratio_
        name = "Custom PCAScratch"
    else:
        model = SklearnPCA(n_components=2, random_state=42)
        reduced = model.fit_transform(X)
        reconstructed = model.inverse_transform(reduced)
        explained_ratio = model.explained_variance_ratio_
        name = "sklearn PCA"

    reconstruction_mse = mean_squared_error(X, reconstructed)

    lines = []
    lines.append("=== PCA Dimensionality Reduction ===")
    lines.append("Dataset: Breast Cancer")
    lines.append("")
    lines.append(f"Model: {name}")
    lines.append("HYPERPARAMETERS:")
    lines.append(format_hyperparameters(model))
    lines.append(
        f"Explained variance ratio: {np.round(explained_ratio, 4).tolist()}")
    lines.append(f"Reconstruction MSE: {reconstruction_mse:.4f}")
    return "\n".join(lines)


EXPERIMENTS = {
    "Decision Tree": run_decision_tree,
    "Logistic Regression": run_logistic_regression,
    "Linear Regression": run_linear_regression,
    "KNN": run_knn,
    "Naive Bayes": run_naive_bayes,
    "Perceptron": run_perceptron,
    "SVM": run_svm,
    "Random Forest": run_random_forest,
    "AdaBoost": run_adaboost,
    "Gradient Boosting": run_gradient_boosting,
    "KMeans": run_kmeans,
    "PCA": run_pca,
}


class ExperimentGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ML Algorithms Lab GUI")
        self.geometry("1100x760")
        self.minsize(900, 650)

        self.model_var = tk.StringVar(value="Decision Tree")
        self.impl_var = tk.StringVar(value="custom")

        self._build_header()
        self._build_controls()
        self._build_output()

    def _build_header(self):
        frame = ttk.Frame(self, padding=(12, 10))
        frame.pack(fill=tk.X)

        title = ttk.Label(frame, text="Traditional ML Experiments",
                          font=("Segoe UI", 16, "bold"))
        title.pack(anchor=tk.W)

        subtitle = ttk.Label(
            frame,
            text="Choose a model and implementation (custom or sklearn), then run and inspect results below.",
        )
        subtitle.pack(anchor=tk.W, pady=(4, 0))

    def _build_controls(self):
        controls = ttk.LabelFrame(self, text="Experiment Controls", padding=12)
        controls.pack(fill=tk.X, padx=12, pady=(0, 8))

        ttk.Label(controls, text="Model:").grid(row=0, column=0, sticky=tk.W)
        self.model_combo = ttk.Combobox(
            controls,
            textvariable=self.model_var,
            values=list(EXPERIMENTS.keys()),
            state="readonly",
            width=30,
        )
        self.model_combo.grid(row=0, column=1, sticky=tk.W, padx=(8, 20))

        ttk.Label(controls, text="Implementation:").grid(
            row=0, column=2, sticky=tk.W)
        self.impl_combo = ttk.Combobox(
            controls,
            textvariable=self.impl_var,
            values=["custom", "sklearn"],
            state="readonly",
            width=15,
        )
        self.impl_combo.grid(row=0, column=3, sticky=tk.W, padx=(8, 20))

        run_button = ttk.Button(
            controls, text="Run Selected Experiment", command=self.run_selected)
        run_button.grid(row=0, column=4, sticky=tk.W)

        clear_button = ttk.Button(
            controls, text="Clear Output", command=self.clear_output)
        clear_button.grid(row=0, column=5, sticky=tk.W, padx=(10, 0))

    def _build_output(self):
        output_frame = ttk.LabelFrame(self, text="Results", padding=8)
        output_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 12))

        self.output_text = ScrolledText(
            output_frame, wrap=tk.WORD, font=("Consolas", 10))
        self.output_text.pack(fill=tk.BOTH, expand=True)

        self.output_text.insert(
            tk.END,
            "Ready. Select a model and implementation, then click 'Run Selected Experiment'.\n",
        )

    def clear_output(self):
        self.output_text.delete("1.0", tk.END)

    def run_selected(self):
        model_name = self.model_var.get()
        implementation = self.impl_var.get()

        runner = EXPERIMENTS.get(model_name)
        if runner is None:
            return

        self.output_text.delete("1.0", tk.END)
        self.output_text.insert(
            tk.END,
            f"Running {model_name} with {implementation} implementation...\n\n",
        )
        self.update_idletasks()

        try:
            result = runner(implementation)
            self.output_text.insert(tk.END, result)
        except Exception as exc:
            self.output_text.insert(
                tk.END, f"Error while running experiment: {exc}\n")


if __name__ == "__main__":
    app = ExperimentGUI()
    app.mainloop()
