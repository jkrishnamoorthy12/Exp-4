# Experiment 4: Support Vector Machine (SVM)

## Aim

To implement and analyze the performance of a Support Vector Machine
(SVM) model for classification using Python.

## Overview

A Support Vector Machine is a supervised machine-learning algorithm used
for classification. It finds a decision boundary (hyperplane) that
separates classes while maximizing the margin between the closest data
points, called **support vectors**.

This experiment uses the **Iris dataset** and compares two SVM kernels:

-   **Linear kernel** -- suitable when classes can be separated
    approximately by a straight decision boundary.
-   **RBF (Radial Basis Function) kernel** -- can model non-linear
    decision boundaries.

For visualization, the experiment uses the first two features of the
Iris dataset.

## Algorithm

1.  **Start.**
2.  Load the Iris dataset and select the first two features for
    visualization.
3.  Split the data into training and testing sets.
4.  Standardize the features using `StandardScaler`.
5.  Train an SVM classifier with a linear kernel.
6.  Predict the test labels and calculate the classification accuracy.
7.  Train another SVM classifier with the RBF kernel.
8.  Predict the test labels and calculate the classification accuracy.
9.  Plot the decision boundaries for both models.
10. **Stop.**

## Program
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load Iris dataset and select first two features for visualization
iris = load_iris()
X = iris.data[:, :2]
y = iris.target
feature_names = iris.feature_names[:2]
class_names = iris.target_names

print("Dataset shape:", X.shape)
print("Features:", feature_names)
print("Classes:", class_names)
print("Class counts:", np.bincount(y))

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standardize features (fit only on training data)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# Train and evaluate Linear SVM
svm_linear = SVC(kernel="linear", C=1.0)
svm_linear.fit(X_train_scaled, y_train)
pred_linear = svm_linear.predict(X_test_scaled)
accuracy_linear = accuracy_score(y_test, pred_linear)

print("\n===== Linear SVM =====")
print(f"Accuracy: {accuracy_linear:.4f} ({accuracy_linear * 100:.2f}%)")
print("Confusion Matrix:")
print(confusion_matrix(y_test, pred_linear))
print("Classification Report:")
print(classification_report(y_test, pred_linear, target_names=class_names))

# Train and evaluate RBF SVM
svm_rbf = SVC(kernel="rbf", C=1.0, gamma="scale")
svm_rbf.fit(X_train_scaled, y_train)
pred_rbf = svm_rbf.predict(X_test_scaled)
accuracy_rbf = accuracy_score(y_test, pred_rbf)

print("\n===== RBF SVM =====")
print(f"Accuracy: {accuracy_rbf:.4f} ({accuracy_rbf * 100:.2f}%)")
print("Confusion Matrix:")
print(confusion_matrix(y_test, pred_rbf))
print("Classification Report:")
print(classification_report(y_test, pred_rbf, target_names=class_names))

# Compare the kernels
print("\n===== Accuracy Comparison =====")
display(pd.DataFrame({
    "Kernel": ["Linear", "RBF"],
    "Accuracy (%)": [accuracy_linear * 100, accuracy_rbf * 100]
}).round(2))

# Plot decision boundaries and support vectors
x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300)
)
grid = np.c_[xx.ravel(), yy.ravel()]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
models = [(svm_linear, "Linear Kernel"), (svm_rbf, "RBF Kernel")]

for ax, (model, kernel_name) in zip(axes, models):
    region = model.predict(grid).reshape(xx.shape)
    ax.contourf(xx, yy, region, alpha=0.25, cmap="viridis")
    ax.scatter(
        X_train_scaled[:, 0], X_train_scaled[:, 1],
        c=y_train, cmap="viridis", edgecolors="black", s=35
    )
    ax.scatter(
        model.support_vectors_[:, 0], model.support_vectors_[:, 1],
        s=110, facecolors="none", edgecolors="red",
        linewidths=1.5, label="Support vectors"
    )
    ax.set_title("SVM Decision Boundary - " + kernel_name)
    ax.set_xlabel(feature_names[0] + " (standardized)")
    ax.set_ylabel(feature_names[1] + " (standardized)")
    ax.legend()

plt.tight_layout()
plt.show()


## output
<img width="768" height="607" alt="image" src="https://github.com/user-attachments/assets/dce67dd6-3509-490c-b692-fc2d8a1d77a0" />



## Result
