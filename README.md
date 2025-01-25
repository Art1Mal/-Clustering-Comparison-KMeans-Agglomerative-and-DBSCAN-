# 🤖🔍 Clustering Comparison: KMeans, Agglomerative, and DBSCAN

**"A Python-based comparison of clustering algorithms on synthetic data."**

This repository contains a Python implementation of three popular clustering algorithms: **KMeans**, **Agglomerative Clustering**, and **DBSCAN**. The project demonstrates how to generate synthetic data, evaluate clustering performance using metrics like Silhouette Score and Distortion, and visualize the results. The goal is to compare the effectiveness of these algorithms on a custom dataset that includes blobs, moons, and circular patterns.

---

## Features

### **Data Generation**
- Synthetic dataset with blobs, moons, and circular patterns.
- Customizable noise levels and sample sizes for each dataset component.

### **Clustering Algorithms**
1. **KMeans**: Partition-based clustering with customizable number of clusters.
2. **Agglomerative Clustering**: Hierarchical clustering with Ward's linkage.
3. **DBSCAN**: Density-based clustering with adjustable `eps` and `min_samples`.

### **Evaluation Metrics**
- **Silhouette Score**: Measures the quality of clustering.
- **Distortion**: Sum of squared distances to cluster centroids (for KMeans and Agglomerative Clustering).

### **Visualization**
- Interactive plots for Silhouette Scores, Distortion, and cluster assignments.
- Side-by-side comparison of clustering results.

---

## Dataset
The synthetic dataset is generated using `make_blobs`, `make_moons`, and a custom circular pattern. The dataset consists of:

- **Blobs**: 1000 samples, 4 centers.
- **Moons**: 50 samples, shifted to avoid overlap.
- **Circular Pattern**: 300 samples with added noise.

---

## Requirements
- Python 3.7+
- NumPy
- scikit-learn
- Matplotlib

---

## Clustering Algorithms

### **KMeans**
- **Number of Clusters**: Optimized using Silhouette Score (best: 8 clusters).
- **Initialization**: k-means++.
- **Max Iterations**: 250.
- **n_init**: 10.

### **Agglomerative Clustering**
- **Number of Clusters**: Optimized using Distortion (best: 6 clusters).
- **Linkage**: Ward's method.
- **Metric**: Euclidean distance.

### **DBSCAN**
- **eps**: 1.5 (maximum distance between two samples for them to be considered as in the same neighborhood).
- **min_samples**: 10 (minimum number of samples in a neighborhood for a point to be considered a core point).

---

### Experiments Overview

#### **KMeans Experiments**

| Experiment | Hyperparameters                                                                                 | Silhouette Score |
|------------|------------------------------------------------------------------------------------------------|------------------|
| #1         | Number of clusters = 2; init = `k-means++`; max_iter = 100; n_init = 5; random_state = 42      | 0.5673           |
| #2         | Number of clusters = 3; init = `random`; max_iter = 200; n_init = 10; random_state = 42        | 0.5858           |
| #3         | Number of clusters = 4; init = `k-means++`; max_iter = 300; n_init = 15; random_state = 42     | 0.6813           |
| #4         | Number of clusters = 5; init = `random`; max_iter = 150; n_init = 20; random_state = 42        | 0.6976           |
| #5         | Number of clusters = 8; init = `k-means++`; max_iter = 250; n_init = 10; random_state = 42     | **0.7160**       |

**Best result:** [Experiment #5] with Silhouette Score = 0.7160

#### **Agglomerative Clustering Experiments**

| Experiment | Hyperparameters                                                   | Distortion      |
|------------|--------------------------------------------------------------------|-----------------|
| #1         | Number of clusters = 6; metric = `euclidean`; linkage = `ward`    | **9854.77**     |
| #2         | Number of clusters = 3; metric = `manhattan`; linkage = `average` | 47923.96        |
| #3         | Number of clusters = 4; metric = `cosine`; linkage = `complete`   | 33394.54        |
| #4         | Number of clusters = 5; metric = `cosine`; linkage = `average`    | 19483.46        |
| #5         | Number of clusters = 8; metric = `cosine`; linkage = `single`     | 47228.04        |
| #6         | Number of clusters = 6; metric = `cosine`; linkage = `single`     | **9854.77**     |

**Best results:** [Experiment #1] and [Experiment #6] with Distortion = 9854.77

#### **DBSCAN Experiments**

| Experiment | Hyperparameters                        | Number of Clusters |
|------------|----------------------------------------|--------------------|
| #1         | eps = 0.5; min_samples = 5            | 23                 |
| #2         | eps = 1.0; min_samples = 5            | **6**              |
| #3         | eps = 0.5; min_samples = 10           | 4                  |
| #4         | eps = 1.5; min_samples = 10           | **6**              |
| #5         | eps = 0.5; min_samples = 15           | 4                  |
| #6         | eps = 0.8; min_samples = 7            | 4                  |

**Best results:** [Experiment #2] and [Experiment #4] with Number of Clusters = 6

---

## Visualizations

### **Synthetic Dataset**
- Visual representation of the generated dataset with blobs, moons, and circular patterns.

### **Silhouette Scores for KMeans**
- Plot showing Silhouette Scores for different numbers of clusters.

### **Distortion for Agglomerative Clustering**
- Plot showing Distortion values for different numbers of clusters.

### **Cluster Assignments**
- Visualizations of cluster assignments for KMeans, Agglomerative Clustering, and DBSCAN.

---

## Future Work
- Add support for real-world datasets (e.g., Iris, Wine).
- Implement additional clustering algorithms (e.g., Spectral Clustering, Gaussian Mixture Models).
- Optimize hyperparameters using grid search or Bayesian optimization.

---

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments
- Thanks to scikit-learn for providing robust implementations of clustering algorithms.
