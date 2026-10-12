[README.md](https://github.com/user-attachments/files/33327170/README.md)
# utelearn

`utelearn` is a custom machine learning library built from scratch for educational purposes at UTE.

## Installation

You can install the package directly from PyPI:

```bash
pip install utelearn
```

## Features

- **KMeans Clustering**: Custom implementation of the KMeans clustering algorithm using NumPy.
- Built-in support for data visualization with Matplotlib.

## Quick Start

Here are quick examples of how to use the machine learning libs from `utelearn`:

## K-Means 
### Kmeans Clustering Example

```python
import numpy as np
import matplotlib.pyplot as plt
from utelearn.kmeans import KMeans

# 1. Tạo dữ liệu mẫu ngẫu nhiên (2 cụm rõ rệt)
np.random.seed(42)
X = np.vstack([
    np.random.randn(50, 2) + np.array([2, 2]),
    np.random.randn(50, 2) + np.array([-2, -2])
])

# 2. Khởi tạo và huấn luyện mô hình K-Means với k=2
kmeans = KMeans(k=2, max_iters=100)
kmeans.fit(X)

# 3. In kết quả trung tâm cụm (centroids)
print("Tọa độ các tâm cụm (Centroids):")
print(kmeans.centroids)

# 4. Dự đoán nhãn cho các điểm dữ liệu (nếu lớp KMeans có hàm predict)
# Hoặc gán nhãn dựa vào khoảng cách gần nhất tới các centroids
labels = np.argmin([np.linalg.norm(X - c, axis=1) for c in kmeans.centroids], axis=0)

# 5. Vẽ biểu đồ trực quan
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o', alpha=0.7, edgecolors='k', label='Dữ liệu mẫu')
plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], c='red', marker='X', s=250, edgecolors='k', label='Tâm cụm (Centroids)')

plt.title("Trực quan hóa thuật toán K-Means - Utelearn", fontsize=14)
plt.xlabel("Trục X", fontsize=12)
plt.ylabel("Trục Y", fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()
```

## K-Nearest Neighbors (KNN)
### 1. KNN Regression & Classification Example

```python
import numpy as np
from utelearn import knn

# Generate synthetic training data
X_train = np.random.rand(100, 1)
y_train = 4 * X_train**2 + 3 * X_train + 2 + np.random.randn(100, 1) * 0.1

# Generate test points
X_test = np.linspace(0, 1, 10).reshape(-1, 1)

# Perform KNN regression with k=5
predictions = knn.knn_regression(X_train, y_train, X_test, k=5)
print("Regression Predictions:", predictions)

# Generate synthetic training and test features/labels
X_train = np.random.rand(100, 2)
y_train = np.random.randint(0, 3, size=100) # 3 classes
X_test = np.random.rand(10, 2)

# Perform KNN classification with k=5
predictions = knn.knn_classify(X_train, y_train, X_test, k=5, n_classes=3)
print("Classification Predictions:", predictions)

```

## K-Nearest Neighbors (KNN)
### 1. KNN Regression & Classification Example using points Lib

```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from utelearn import knn
from utelearn import points as point

# 1. Configuration parameters
n_training_points = 100
n_test_points = 500
n_classes = 3
dimension = 2
k_neighbors = 5

# 2. Generate Training and Test Data using utelearn's points module (Spiral Distribution)
training_data = point.Spiral(n_training_points, n_classes, dimension)
test_data = point.Spiral(n_test_points, n_classes, dimension)

# 3. Perform KNN Classification using utelearn's knn module
print("Running KNN classification on test data using utelearn...")
predicted_labels = knn.knn_classify(
    X_train=training_data.P, 
    y_train=training_data.L, 
    X_test=test_data.P, 
    k=k_neighbors, 
    n_classes=n_classes
)

# 4. Plotting the Results
plt.figure(figsize=(15, 5))

# Subplot 1: Reference Training Data
plt.subplot(1, 3, 1)
plt.scatter(
    training_data.P[:, 0], 
    training_data.P[:, 1], 
    c=training_data.L, 
    s=20, 
    cmap=mcolors.ListedColormap(["red", "purple", "green"])
)
plt.title(f"Reference Data ({n_training_points * n_classes} points)")
plt.grid(True, alpha=0.3)

# Subplot 2: Unclassified Test Data
plt.subplot(1, 3, 2)
plt.scatter(
    test_data.P[:, 0], 
    test_data.P[:, 1], 
    color="blue", 
    s=15
)
plt.title("Unclassified Test Data")
plt.grid(True, alpha=0.3)

# Subplot 3: Classified Test Data via utelearn KNN
plt.subplot(1, 3, 3)
plt.scatter(
    test_data.P[:, 0], 
    test_data.P[:, 1], 
    c=predicted_labels, 
    s=15, 
    cmap=mcolors.ListedColormap(["red", "purple", "green"])
)
plt.title(f"Test Data After KNN (k={k_neighbors})")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## Naives Bayes Classifier (using Gaussian probability density function)
### Classification Example using points Lib

```python
import numpy as np 
import matplotlib.pyplot as plt 
from utelearn import points as point  
from utelearn import naives_bayes as nb

# Configuration parameters
n_generated_points = 1000  
n_classes = 3 
dimension = 2  
N_total = n_generated_points * n_classes  

###############################################################
################### Generate Training Data ####################
###############################################################
Generated_Data = point.Circle(n_generated_points, n_classes, dimension)  
X = np.column_stack((Generated_Data.P[:, 0], Generated_Data.P[:, 1]))
y = Generated_Data.L 

# Train the Naive Bayes classifier using the custom library
means, vars, priors = nb.train_naive_bayes(X, y) 

############################################################## 
################### Generate Test Data ####################### 
############################################################## 
Generated_Data1 = point.Circle(n_generated_points, n_classes, dimension)  
X_test = np.column_stack((Generated_Data1.P[:, 0], Generated_Data1.P[:, 1]))
y_test = Generated_Data1.L 

# Predict on the test data 
y_pred = nb.predict_naive_bayes(X_test, means, vars, priors) 

# Print predictions and actual labels 
print("Predictions:", y_pred) 
print("Actual labels:", y_test) 

# Calculate and print accuracy 
accuracy = np.mean(y_pred == y_test) 
print(f'Accuracy: {accuracy * 100:.2f}%') 

########################################################################## 
####################  Visualization Section  #############################
##########################################################################
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1 
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1 
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1)) 

# Predict labels for all points in the meshgrid 
grid_points = np.c_[xx.ravel(), yy.ravel()] 
Z = nb.predict_naive_bayes(grid_points, means, vars, priors) 
Z = Z.reshape(xx.shape) 

# Plotting side-by-side comparison
plt.figure(figsize=(12, 5))

# Subplot 1: Original Test Data
plt.subplot(1, 2, 1)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='coolwarm', edgecolors='k', marker='o', label='Test Data Points') 
plt.xlabel('Feature X_test[0]') 
plt.ylabel('Feature X_test[1]') 
plt.title(f'Test Data ({n_classes} Classes)') 

# Subplot 2: Decision Boundary
plt.subplot(1, 2, 2)
plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm') 
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='coolwarm', edgecolors='k', marker='o', label='Predictions') 
plt.xlabel('Feature X_test[0]') 
plt.ylabel('Feature X_test[1]') 
plt.title(f'Naive Bayes Decision Boundary ({n_classes} Classes)') 
plt.legend(loc='best') 

plt.tight_layout()
plt.show()
```

## SVM Classifier (using Kernel Trick)
### Classification Example using points Lib

```python
"""
Multi-Class Support Vector Machine (SVM) Evaluation Script.
Demonstrates Non-Linear Classification on a Spiral Dataset using an RBF Kernel 
and the One-vs-Rest (OvR) Decomposition Strategy.
Optimized for academic demonstration with shared Gram matrix computation and Momentum.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from utelearn.points import Spiral
import utelearn.svm as svm_kernel


# =====================================================================
# 1. Configuration & Hyperparameter Setup
# =====================================================================
n_generated_points = 1000  # Number of samples per class (Total: 3,000 points)
n_classes = 3              # Multi-class target count
dimension = 2              # Feature dimension (2D coordinate plane)
C = 10.0                   # Regularization parameter (penalizes margin violations)
GAMMA = 25.0               # RBF kernel width parameter (controls Gaussian influence radius)


# =====================================================================
# 2. Synthetic Data Generation
# =====================================================================
data_obj = Spiral(n_generated_points, n_classes, dimension)
X = data_obj.P
labels = data_obj.L


# =====================================================================
# 3. One-vs-Rest (OvR) Binary Decomposition Training
# =====================================================================
models = []
print("=== STARTING SVM TRAINING (ONE-VS-REST DECOMPOSITION) ===")
start_train_time = time.time()

# Academic Optimization 1: Precompute the shared Gram Matrix once for all OvR classifiers.
# Since X and gamma are identical across all classes, K(X, X) is invariant.
svm_base = svm_kernel.SVM()
K_shared = svm_base.rbf_kernel(X, X, gamma=GAMMA)

for i in range(n_classes):
    # Formulate binary targets: Target class (+1) vs. All remaining classes (-1)
    y_binary = np.where(labels == i, 1, -1)
    
    # Train binary SVM model via Momentum-Accelerated Projected Gradient Ascent (120 epochs)
    svm = svm_kernel.SVM()
    alpha = svm.svm_rbf(X, y_binary, C=C, gamma=GAMMA, max_iter=120, eta=0.005, beta=0.8, K=K_shared)
    b = svm.calculate_bias(X, y_binary, alpha, gamma=GAMMA, C=C, K=K_shared)
    
    # Extract active Support Vectors (samples with non-zero Lagrange multipliers)
    support_vectors = alpha > 1e-4
    n_sv = np.sum(support_vectors)
    sv_ratio = (n_sv / len(X)) * 100
    print(f"Classifier [Class {i} vs. Rest]: Identified {n_sv}/{len(X)} Support Vectors ({sv_ratio:.1f}%)")
    
    # Precompute dual coefficients (alpha_i * y_i) for high-speed inference
    dual_coef = alpha[support_vectors] * y_binary[support_vectors]
    
    models.append({
        'X_sv': X[support_vectors],
        'dual_coef': dual_coef,
        'alpha_sv': alpha[support_vectors],
        'y_sv': y_binary[support_vectors],
        'b': b
    })

training_duration = time.time() - start_train_time
print(f"=== TRAINING COMPLETED IN {training_duration:.2f} SECONDS ===")


# =====================================================================
# 4. Multi-Class Prediction Routine
# =====================================================================
def predict_multiclass(X_test, models):
    """
    Predict multi-class labels using the argmax of binary decision scores.
    """
    scores = []
    svm_inst = svm_kernel.SVM()
    for model in models:
        score = svm_inst.decision_function(
            X_test, model['X_sv'], b=model['b'], gamma=GAMMA, dual_coef=model['dual_coef']
        )
        scores.append(score)
    return np.argmax(scores, axis=0)


# Evaluate overall training accuracy
y_pred = predict_multiclass(X, models)
acc = np.mean(y_pred == labels) * 100
print(f"=== OVERALL TRAINING ACCURACY: {acc:.2f}% ===")


# =====================================================================
# 5. Visualization & Decision Boundary Analysis
# =====================================================================
if __name__ == "__main__":
    from matplotlib.colors import ListedColormap

    # Harmonized Distinct Palettes:
    # Class 0 = Red, Class 1 = Green, Class 2 = Blue
    pt_colors = ['#E63946', '#2A9D8F', '#457B9D']
    bg_colors = ['#FFD6D9', '#D8F3DC', '#D0E1FD']
    cmap_pt = ListedColormap(pt_colors)
    cmap_bg = ListedColormap(bg_colors)

    # Figure 1: Raw Training Distribution
    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap=cmap_pt, s=15, alpha=0.85)
    plt.title("Figure 1: Ground Truth Dataset (3-Arm Spiral, 1000 pts/class)")
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

    # Figure 2: One-vs-Rest Decision Boundaries and Combined Ensemble Result
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    svm_inst = svm_kernel.SVM()

    # Generate 2D evaluation grid
    h = 0.03
    x_min, x_max = X[:, 0].min() - 0.4, X[:, 0].max() + 0.4
    y_min, y_max = X[:, 1].min() - 0.4, X[:, 1].max() + 0.4
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    mesh_points = np.c_[xx.ravel(), yy.ravel()]

    # Cache decision values across the 3 subplots to eliminate duplicate calculations
    mesh_scores = []

    # Subplots 1-3: Individual One-vs-Rest Binary Decision Surfaces
    for i in range(n_classes):
        Z_flat = svm_inst.decision_function(
            mesh_points, models[i]['X_sv'], b=models[i]['b'], 
            gamma=GAMMA, dual_coef=models[i]['dual_coef']
        )
        mesh_scores.append(Z_flat)
        Z = Z_flat.reshape(xx.shape)
        
        # Binary background highlighting the positive class territory
        axes[i].contourf(xx, yy, Z, levels=[-100, 0, 100], colors=['#F8F9FA', bg_colors[i]], alpha=0.6)
        
        # Decision boundary (f(x) = 0) and margins (f(x) = +/- 1)
        axes[i].contour(
            xx, yy, Z, levels=[-1, 0, 1], colors=['#888888', '#111111', '#888888'], 
            linestyles=['--', '-', '--'], linewidths=[1.0, 1.8, 1.0]
        )
        
        # Scatter training observations with vibrant colors
        axes[i].scatter(X[:, 0], X[:, 1], c=labels, cmap=cmap_pt, s=12, alpha=0.7)
        
        # Highlight active Support Vectors with crisp red rings
        axes[i].scatter(
            models[i]['X_sv'][:, 0], models[i]['X_sv'][:, 1], 
            facecolors='none', edgecolors='#D62828', s=35, linewidths=0.9, 
            label='Support Vectors'
        )
        
        axes[i].set_title(f"Subplot {i+1}: Class {i} vs. Rest (Margins & SVs)")
        axes[i].legend(loc='upper right')

    # Subplot 4: Final Combined Ensemble Decision Surface (Synchronized Color Mapping)
    Z_final = np.argmax(mesh_scores, axis=0).reshape(xx.shape)
    axes[3].contourf(xx, yy, Z_final, levels=[-0.5, 0.5, 1.5, 2.5], cmap=cmap_bg, alpha=0.8)
    axes[3].contour(xx, yy, Z_final, levels=[0.5, 1.5], colors=['#222222', '#222222'], linewidths=1.5)
    axes[3].scatter(X[:, 0], X[:, 1], c=labels, cmap=cmap_pt, s=12, alpha=0.85)
    axes[3].set_title(f"Subplot 4: Final Ensembled Boundary (Accuracy: {acc:.2f}%)")

    plt.tight_layout()
    plt.show()
```

## Author

- **Dung Cai** (dung.cai@hcmute.edu.vn)

## Co-Authors
- **Dang Khoa Bui**
- **Minh Tri Huynh**
