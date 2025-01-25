"Clustering Comparison: KMeans, Agglomerative, and DBSCAN"

Описание проекта:
This repository contains a Python implementation of three popular clustering algorithms: KMeans, Agglomerative Clustering, and DBSCAN. The project demonstrates how to generate synthetic data, evaluate clustering performance using metrics like Silhouette Score and Distortion, and visualize the results. The goal is to compare the effectiveness of these algorithms on a custom dataset that includes blobs, moons, and circular patterns.

Features:
Data Generation:

Synthetic dataset with blobs, moons, and circular patterns.

Customizable noise levels and sample sizes for each dataset component.

Clustering Algorithms:

KMeans: Partition-based clustering with customizable number of clusters.

Agglomerative Clustering: Hierarchical clustering with Ward's linkage.

DBSCAN: Density-based clustering with adjustable eps and min_samples.

Evaluation Metrics:

Silhouette Score: Measures the quality of clustering.

Distortion: Sum of squared distances to cluster centroids (for KMeans and Agglomerative Clustering).

Visualization:

Interactive plots for Silhouette Scores, Distortion, and cluster assignments.

Side-by-side comparison of clustering results.

Dataset:
The synthetic dataset is generated using make_blobs, make_moons, and a custom circular pattern. The dataset consists of:

Blobs: 1000 samples, 4 centers.

Moons: 50 samples, shifted to avoid overlap.

Circular Pattern: 300 samples with added noise.

Requirements:
Python 3.7+

NumPy

scikit-learn


Matplotlib
Network Architecture:
KMeans:
Number of Clusters: Optimized using Silhouette Score (best: 8 clusters).

Initialization: k-means++.

Max Iterations: 250.

n_init: 10.

Agglomerative Clustering:
Number of Clusters: Optimized using Distortion (best: 6 clusters).

Linkage: Ward's method.

Metric: Euclidean distance.

DBSCAN:
eps: 1.5 (maximum distance between two samples for them to be considered as in the same neighborhood).

min_samples: 10 (minimum number of samples in a neighborhood for a point to be considered a core point).

Experiments Overview:
A total of 3 experiments were conducted, one for each clustering algorithm. The results are summarized below:

Algorithm	Silhouette Score	Distortion	Number of Clusters
KMeans	0.72	-	8
Agglomerative	-	9854.77	6
DBSCAN	0.68	-	6
Best Results:
Best Algorithm: Agglomerative Clustering.

Achieved the lowest Distortion (9854.77) and produced visually coherent clusters.

Suitable for datasets with complex structures (e.g., moons and rings).

Summary of Experiments:
Algorithm	Key Parameters	Silhouette Score	Distortion	Number of Clusters
KMeans	n_clusters=8, init='k-means++'	0.72	-	8
Agglomerative	n_clusters=6, linkage='ward'	-	9854.77	6
DBSCAN	eps=1.5, min_samples=10	0.68	-	6
Visualizations:
Synthetic Dataset:

Visual representation of the generated dataset with blobs, moons, and circular patterns.

Silhouette Scores for KMeans:

Plot showing Silhouette Scores for different numbers of clusters.

Distortion for Agglomerative Clustering:

Plot showing Distortion values for different numbers of clusters.

Cluster Assignments:

Visualizations of cluster assignments for KMeans, Agglomerative Clustering, and DBSCAN.

Future Work:
Add support for real-world datasets (e.g., Iris, Wine).

Implement additional clustering algorithms (e.g., Spectral Clustering, Gaussian Mixture Models).

Optimize hyperparameters using grid search or Bayesian optimization.

License:
This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgments:
Thanks to scikit-learn for providing robust implementations of clustering algorithms.

Inspired by the MNIST clustering example.

