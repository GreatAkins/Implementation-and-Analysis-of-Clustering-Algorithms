#StudentID: 201770868

import random
import numpy as np
import matplotlib.pyplot as plt

# Set a fixed seed for reproducibility
random.seed(42)

def generateSyntheticData(num_points, num_features):
    """
    Generate synthetic data points with random values.

    Args:
        num_points (int): Number of data points to generate.
        num_features (int): Number of features for each data point.

    Returns:
        numpy.ndarray: Array containing synthetic data points with random values.
    """
    if num_points <= 0 or num_features <= 0:
        print("Error: Number of points and features must be positive integers.")
        return None
    
    return np.random.rand(num_points, num_features)

def computeDistance(a, b):
    """
    Compute Euclidean distance between two points.

    Args:
        a (array-like): First point.
        b (array-like): Second point.

    Returns:
        float: Euclidean distance between points a and b.
    """
    return np.sqrt(np.sum((np.array(a) - np.array(b)) ** 2))

def initialSelection(data, k):
    """
    Choose initial cluster representatives.

    Args:
        data (numpy.ndarray): Dataset.
        k (int): Number of clusters.

    Returns:
        list: Initial cluster representatives.
    """
    return random.sample(list(data), k)

def assignClusterIds(data, representatives):
    """
    Assign cluster ids to each data point.

    Args:
        data (numpy.ndarray): Dataset.
        representatives (list): Initial cluster representatives.

    Returns:
        dict: Dictionary mapping data point index to assigned cluster id.
    """
    clusters = {}
    for idx, point in enumerate(data):
        distances = [computeDistance(point, rep) for rep in representatives]
        min_distance_index = distances.index(min(distances))
        clusters[idx] = min_distance_index
    return clusters

def computeClusterRepresentatives(data, clusters, k):
    """
    Compute cluster representatives based on assigned cluster IDs.

    Args:
        data (numpy.ndarray): Dataset.
        clusters (dict): Dictionary mapping data point index to assigned cluster id.
        k (int): Number of clusters.

    Returns:
        list: List of cluster representatives.
    """
    # Initialize new representatives and counts
    new_representatives = [np.zeros(len(data[0])) for _ in range(k)]
    counts = [0] * k

    # Aggregate data points for each cluster
    for idx, cluster_id in clusters.items():
        new_representatives[cluster_id] += data[idx]
        counts[cluster_id] += 1

    # Compute mean for each cluster
    for i in range(k):
        if counts[i] > 0:
            new_representatives[i] /= counts[i]

    return new_representatives

def silhouetteCoefficient(data, clusters, k):
    """
    Compute silhouette coefficient for clusters.

    Args:
        data (numpy.ndarray): Dataset.
        clusters (dict): Dictionary mapping data point index to assigned cluster id.
        k (int): Number of clusters.

    Returns:
        float: Silhouette coefficient.
    """
    if k == 1:
        return 0  # Silhouette coefficient is not defined for k = 1

    def intraClusterDistance(point, cluster_id):
        """
        Compute the intra-cluster distance for a point.

        Args:
            point (numpy.ndarray): Data point.
            cluster_id (int): ID of the cluster.

        Returns:
            float: Intra-cluster distance.
        """
        return np.mean([computeDistance(point, data[idx]) for idx in clusters if clusters[idx] == cluster_id])

    def nearestClusterDistance(point, cluster_id):
        """
        Compute the nearest cluster distance for a point.

        Args:
            point (numpy.ndarray): Data point.
            cluster_id (int): ID of the cluster.

        Returns:
            float: Nearest cluster distance.
        """
        distances = [intraClusterDistance(point, cid) for cid in range(k) if cid != cluster_id]
        return min(distances) if distances else 0

    score = 0
    for idx, point in enumerate(data):
        a = intraClusterDistance(point, clusters[idx])
        b = nearestClusterDistance(point, clusters[idx])
        score += (b - a) / max(a, b)
    return score / len(data)

def plot_silhouette(k_values, silhouette_values):
    """
    Plot silhouette coefficients against the number of clusters.

    Args:
        k_values (list): List of integers representing the number of clusters (k).
        silhouette_values (list): List of silhouette coefficients corresponding to each k value.

    Returns:
        None
    """
    if len(k_values) != len(silhouette_values) or len(k_values) == 0:
        print("Error: Invalid input for plotting silhouette.")
        return
    
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, silhouette_values, marker='o')
    plt.title('Silhouette Coefficient by Number of Clusters')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Coefficient')
    plt.xticks(k_values)
    plt.grid(True)
    plt.savefig('KMeansSyntheticPlot.png')
    plt.show()

def Kmeans(data, maxIter=100):
    """
    Perform k-means clustering on the given data.

    Args:
        data (numpy.ndarray): Input data for clustering.
        maxIter (int): Maximum number of iterations for k-means algorithm. Default is 100.

    Returns:
        None
    """
    if data is None or len(data) == 0:
        print("Error: Empty dataset provided.")
        return

    silhouette_values = []
    k_values = range(1, 10)

    for k in k_values:
        representatives = initialSelection(data, k)
        for _ in range(maxIter):
            clusters = assignClusterIds(data, representatives)
            new_representatives = computeClusterRepresentatives(data, clusters, k)
            if np.allclose(new_representatives, representatives):
                break
            representatives = new_representatives
        silhouette = silhouetteCoefficient(data, clusters, k)
        silhouette_values.append(silhouette)

    # Plot silhouette coefficients against k values
    plot_silhouette(k_values, silhouette_values)

# Assuming the original dataset has 327 data points and 300 features
syntheticData = generateSyntheticData(327, 300)

# Perform k-means clustering on the synthetic data
Kmeans(syntheticData)

