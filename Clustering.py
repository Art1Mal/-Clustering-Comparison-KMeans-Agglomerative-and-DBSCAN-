import numpy as np
from sklearn.datasets import make_blobs, make_moons
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score


def create_data(
    the_noise: float,
    n1_samples: int,
    n2_samples: int,
    n3_samples: int,
    n4_samples: int,
) -> np.ndarray:
    """
    Generates synthetic data for clustering.

    Args:
        the_noise (float): Noise level for the moons dataset.
        n1_samples (int): Number of samples for the blobs dataset.
        n2_samples (int): Number of samples for the moons dataset.
        n3_samples (int): Number of samples for the circular dataset.
        n4_samples (int): Unused parameter (for future extensions).

    Returns:
        np.ndarray: Combined dataset with shape (n_samples, 2).
    """
    # Generate blobs dataset
    X_1, y_1 = make_blobs(n_samples=n1_samples, centers=4, random_state=42)

    # Generate moons dataset and shift it
    X_2, y_2 = make_moons(n_samples=n2_samples, noise=the_noise, random_state=42)
    X_2[:, 0] += 10
    X_2[:, 1] += 10

    # Generate circular dataset with noise
    rng = np.random.RandomState(42)
    angles = rng.rand(n3_samples) * 2 * np.pi
    r = 5.0 + rng.randn(n3_samples) * the_noise
    X_3 = np.column_stack((r * np.cos(angles), r * np.sin(angles)))
    X_3[:, 0] += 20

    # Combine all datasets
    X = np.vstack((X_1, X_2, X_3))
    return X


def plot_clusters(
    X: np.ndarray, labels: np.ndarray, title: str, centers: np.ndarray = None
) -> None:
    """
    Visualizes the clusters.

    Args:
        X (np.ndarray): Input data with shape (n_samples, 2).
        labels (np.ndarray): Cluster labels for each data point.
        title (str): Title of the plot.
        centers (np.ndarray, optional): Cluster centers. Defaults to None.
    """
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", s=10)
    if centers is not None:
        plt.scatter(centers[:, 0], centers[:, 1], c="red", marker="x", s=40)
    plt.title(title)
    plt.show()


def main() -> None:
    """
    Main function to execute clustering algorithms and compare results.
    """
    # Generate synthetic data
    X = create_data(the_noise=1.0, n1_samples=1000, n2_samples=50, n3_samples=300, n4_samples=200)

    # Visualize the synthetic dataset
    plt.scatter(X[:, 0], X[:, 1], s=10)
    plt.title("Synthetic Dataset with Blobs, Moons, and Rings")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.show()

    # KMeans Clustering
    silhouette_kmeans_scores = []
    cluster_range = range(2, 13)

    # Evaluate KMeans for different numbers of clusters
    for n_clusters in cluster_range:
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        cluster_labels = kmeans.fit_predict(X)
        silhouette_avg = silhouette_score(X, cluster_labels)
        silhouette_kmeans_scores.append(silhouette_avg)
        print(f"For n_clusters = {n_clusters}, Silhouette Score = {silhouette_avg}")

    # Plot Silhouette Scores for KMeans and visualize clusters for the best n_clusters
    n_clusters_for_kmeans = 8
    kmeans_cluster = KMeans(
        n_clusters=n_clusters_for_kmeans, init="k-means++", max_iter=250, n_init=10, random_state=42
    )
    kmeans_labels = kmeans_cluster.fit_predict(X)
    kmeans_silhouette = silhouette_score(X, kmeans_labels)

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot Silhouette Scores
    ax1.plot(cluster_range, silhouette_kmeans_scores, marker="o")
    ax1.set_xlabel("Number of clusters")
    ax1.set_ylabel("Silhouette Score")
    ax1.set_title("Silhouette Method for choosing the optimal number of clusters")

    # Visualize KMeans clusters
    ax2.scatter(X[:, 0], X[:, 1], c=kmeans_labels, cmap="viridis", s=10)
    ax2.scatter(kmeans_cluster.cluster_centers_[:, 0], kmeans_cluster.cluster_centers_[:, 1], c="red", marker="x", s=40)
    ax2.set_title(f"KMeans Clustering (n_clusters={n_clusters_for_kmeans})")

    plt.show()
    print(f"KMeans Silhouette Score: {kmeans_silhouette:.2f}")

    # Agglomerative Clustering
    distortion_scores = []
    for n_clusters in cluster_range:
        agglo = AgglomerativeClustering(n_clusters=n_clusters, metric="euclidean", linkage="ward")
        cluster_labels = agglo.fit_predict(X)
        distortion = 0
        for i in range(n_clusters):
            cluster_points = X[cluster_labels == i]
            if len(cluster_points) > 0:
                centroid = np.mean(cluster_points, axis=0)
                distortion += np.sum((cluster_points - centroid) ** 2)
        distortion_scores.append(distortion)
        print(f"For n_clusters = {n_clusters}, Distortion = {distortion}")

    # Fit Agglomerative Clustering with the optimal number of clusters
    number_of_clusters = 6
    agglo = AgglomerativeClustering(n_clusters=number_of_clusters, metric="euclidean", linkage="ward")
    agglo_labels = agglo.fit_predict(X)
    distortion = distortion_scores[number_of_clusters - 2]  # Adjust index for cluster_range

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot Distortion Scores
    ax1.plot(cluster_range, distortion_scores, marker="o")
    ax1.set_xlabel("Number of clusters")
    ax1.set_ylabel("Distortion")
    ax1.set_title("Elbow Method for choosing the optimal number of clusters")

    # Visualize Agglomerative Clustering
    ax2.scatter(X[:, 0], X[:, 1], c=agglo_labels, cmap="viridis", s=10)
    ax2.set_title(f"Agglomerative Clustering (n_clusters={number_of_clusters})")

    plt.show()
    print(f"AgglomerativeClustering Distortion: {distortion:.2f}")

    # DBSCAN Clustering
    eps = 1.5
    min_samples = 10
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan_labels = dbscan.fit_predict(X)

    # Visualize DBSCAN clusters
    plot_clusters(X, dbscan_labels, f"DBSCAN Clustering (eps={eps}, min_samples={min_samples})")

    # Evaluate DBSCAN
    n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    print(f"DBSCAN Number of clusters: {n_clusters_dbscan}")
    if n_clusters_dbscan > 1:
        dbscan_silhouette = silhouette_score(X, dbscan_labels)
        print(f"DBSCAN Silhouette Score: {dbscan_silhouette:.2f}")
    else:
        print("DBSCAN found only one cluster. Silhouette Score cannot be computed.")

    # Compare results
    print("\nComparison of methods:")
    print(f"KMeans Silhouette Score: {kmeans_silhouette:.2f}")
    print(f"AgglomerativeClustering Distortion: {distortion:.2f}")
    print(f"DBSCAN Number of clusters: {n_clusters_dbscan}")
    if n_clusters_dbscan > 1:
        print(f"DBSCAN Silhouette Score: {dbscan_silhouette:.2f}")

    # Final conclusion
    if n_clusters_dbscan > 1:
        if kmeans_silhouette > dbscan_silhouette and kmeans_silhouette > (1 / distortion):
            print("\nBest method: KMeans")
        else:
            print("\nBest method: AgglomerativeClustering")
    else:
        print("\nBest method: AgglomerativeClustering (DBSCAN found only one cluster)")


if __name__ == "__main__":
    main()