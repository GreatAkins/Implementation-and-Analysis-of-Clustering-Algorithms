#StudentID: 201770868

import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def loadDataset():
    """
    Load dataset from file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        numpy.ndarray or None: The loaded dataset as a NumPy array, or None if file not found or empty.
    """
    try:
        dataset = pd.read_csv('dataset', header=None)
        if dataset.empty:
            print("Error: File is empty.")
            return None

        dataset = dataset[0].str.split(' ', expand=True)
        dataset.drop(columns=[0], inplace=True)  # Remove the first column
        # Convert all columns to numeric, replacing non-numeric values with NaN
        dataset = dataset.apply(pd.to_numeric, errors='coerce')
        # Drop rows with NaN values
        dataset.dropna(inplace=True)
        if dataset.empty:
            print("Error: File contains no valid numeric data.")
            return None
        return dataset.to_numpy()
    except FileNotFoundError:
        print("Error: File not found.")
        return None

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

def distanceMatrix(data):
    """
    Compute the distance matrix for a given dataset.

    Args:
        x (numpy.ndarray): Dataset.

    Returns:
        distances (numpy.ndarray): Distance matrix.
    """
    num_points = len(data)
    distances = np.zeros((num_points, num_points))
    for i in range(num_points):
        for j in range(i, num_points):
            distances[i, j] = computeDistance(data[i], data[j])
            distances[j, i] = distances[i, j]
    return distances

def initialSelection(data, k):
    """
    Initialise centroids randomly from the dataset.

    Args:
        x (numpy.ndarray): Dataset from which to draw the initial centroids.
        k (int): Number of centroids to initialise.

    Returns:
        centroids (numpy.ndarray): Initialised centroids.
                                   Returns None if initialSelection fails due to
                                   invalid input.
    """
    if data is None or len(data) == 0 or k <= 0 or k > len(data):
        print("Error: Invalid dataset or k. Ensure dataset is not empty,"
              " k is positive, and k does not exceed the number of"
              " data points.")
        return None
    # Set a fixed seed for reproducibility
    np.random.seed(42)
    centroids_indices = np.random.choice(len(data), k, replace=False)
    centroids = data[centroids_indices]
    return centroids

def assignClusterIds(data, centroids):
    """
    Assign data points to clusters based on centroids.

    Args:
        data (numpy.ndarray): Dataset.
        centroids (numpy.ndarray): Centroids.

    Returns:
        clusters (numpy.ndarray): Assigned clusters.
    """
    clusters = []
    for point in data:
        distances = [computeDistance(point, centroid) for centroid in centroids]
        cluster_id = np.argmin(distances)
        clusters.append(cluster_id)
    return np.array(clusters)

def computeClusterRepresentatives(data, clusters, k):
    """
    Compute the centre of each cluster to update centroids.

    Args:
        x (numpy.ndarray): Dataset.
        clusters (numpy.ndarray): Assigned clusters.
        k (int): Number of clusters.

    Returns:
        new_centroids (numpy.ndarray): Updated centroids.
    """
    new_centroids = []
    for i in range(k):
        cluster_points = data[clusters == i]
        if len(cluster_points) > 0:
            cluster_mean = np.mean(cluster_points, axis=0)
            new_centroids.append(cluster_mean)
    return np.array(new_centroids)

def kMeans(data, k, maxIter=300):
    """
    Implement k-means clustering algorithm.

    Args:
        x (numpy.ndarray): Dataset.
        k (int): Number of clusters.
        maxIter (int): Maximum number of iterations.

    Returns:
        Tuple of numpy.ndarray, numpy.ndarray or
        Tuple of None, None if dataset is insufficient.
    """
    if data is None or len(data) < 2:
        print("Error: Dataset is too small for clustering.")
        return None, None

    try:
        centroids = initialSelection(data, k)
        for _ in range(maxIter):
            clusters = assignClusterIds(data, centroids)
            new_centroids = computeClusterRepresentatives(data, clusters, k)
            if np.allclose(centroids, new_centroids):
                break
            centroids = new_centroids
        return clusters, centroids
    except Exception as e:
        print(f"Error in k-means algorithm: {e}")
        return None

def computeSumOfSquare(cluster_points, centroid):
    """
    Calculate the Sum of Squared Errors (SSE) for a cluster.

    Args:
        cluster_points (numpy.ndarray): Data points in the cluster.
        centroid (numpy.ndarray): Centroid of the cluster.

    Returns:
        sse (float): Sum of Squared Errors for the cluster.
    """
    # Calculate the squared Euclidean distances between each point and the centroid
    squared_distances = np.sum((cluster_points - centroid) ** 2, axis=1)
    
    # Compute the sum of squared distances
    sse = np.sum(squared_distances)
    
    return sse

def selectClusterToBisect(hierarchical_clusters):
    """
    Select the cluster with the highest SSE for bisecting.

    Args:
        hierarchical_clusters (list): List of clusters.

    Returns:
        cluster_to_bisect (dict): Cluster selected for bisecting.
    """
    max_sse = -np.inf
    cluster_to_bisect = None
    for cluster in hierarchical_clusters:
        if cluster['sse'] > max_sse:
            max_sse = cluster['sse']
            cluster_to_bisect = cluster
    return cluster_to_bisect

def bisectCluster(cluster_to_bisect, hierarchical_clusters, centroids, k=2, maxIter=300):
    """
    Bisect a cluster to improve clustering quality.

    Args:
        cluster_to_bisect (dict): Cluster selected for bisecting.
        hierarchical_clusters (list): List of clusters in the hierarchy.
        centroids (list): List of current centroids.
        k (int): Number of clusters after bisecting.
        maxIter (int): Maximum number of iterations for kMeans.

    Raises:
        ValueError: If cluster_to_bisect is not found in hierarchical_clusters,
                    or points_to_bisect is not a 2D array.

    Returns:
        None: Modifies hierarchical_clusters and centroids in place.
    """
    # Find the index of the cluster to bisect
    index_to_bisect = None
    for i, cluster in enumerate(hierarchical_clusters):
        if np.array_equal(cluster['points'], cluster_to_bisect['points']) and \
           np.array_equal(cluster['centroid'], cluster_to_bisect['centroid']):
            index_to_bisect = i
            break

    # Raise an error if the cluster to bisect is not found
    if index_to_bisect is None:
        raise ValueError("Cluster to bisect not found in hierarchical clusters.")

    # Get the points to bisect and check if they form a 2D array
    points_to_bisect = np.array(cluster_to_bisect['points'])
    if points_to_bisect.ndim != 2:
        raise ValueError("Points to bisect must be a 2D array")

    # Perform kMeans clustering on the points to bisect
    subcluster_labels, subcentroids = kMeans(points_to_bisect, k, maxIter)

    # Remove the bisected cluster from the hierarchical clusters
    hierarchical_clusters.pop(index_to_bisect)

    # Update centroids: remove the centroid of the cluster to bisect and add new centroids
    for idx, cent in enumerate(centroids):
        if np.array_equal(cent, cluster_to_bisect['centroid']):
            centroids.pop(idx)
            break

    # Add new subclusters to the hierarchical clusters and centroids
    for idx in range(k):
        subcluster_points = points_to_bisect[subcluster_labels == idx]
        subcentroid = subcentroids[idx]
        new_sse = computeSumOfSquare(subcluster_points, subcentroid)
        hierarchical_clusters.append({
            'points': subcluster_points,
            'centroid': subcentroid,
            'sse': new_sse
        })
        centroids.append(subcentroid)

def extractClustersFromHierarchy(hierarchical_clusters):
    """
    Extract clusters from the hierarchical structure.

    Args:
        hierarchical_clusters (list): List of clusters in the hierarchy.

    Returns:
        list: List of points for each cluster.
    """
    return [cluster['points'] for cluster in hierarchical_clusters]

def silhouetteCoefficient(data, clusters):
    """
    Compute the silhouette coefficient for a given clustering.

    Args:
        data (numpy.ndarray): Dataset.
        clusters (numpy.ndarray): Cluster assignments for each data point.

    Returns:
        float: Silhouette coefficient.
    """
    if len(set(clusters)) == 1:
        return 0  # Silhouette coefficient is not defined for k = 1

    try:
        distances_matrix = distanceMatrix(data)
        silhouette_scores = []
        np.seterr(divide='ignore', invalid='ignore')

        for i in range(len(data)):
            cluster_i = clusters[i]
            cluster_points = data[clusters == cluster_i]
            a_i = np.mean(distances_matrix[i][clusters == cluster_i])
            b_i = np.inf
            for other_cluster in range(len(set(clusters))):
                if other_cluster != cluster_i:
                    mean_distance = np.mean(distances_matrix[i][clusters == other_cluster])
                    if not np.isnan(mean_distance):
                        b_i = min(b_i, mean_distance)
            silhouette_i = (b_i - a_i) / max(a_i, b_i) if len(cluster_points) > 1 else 0
            silhouette_scores.append(silhouette_i)

        np.seterr(divide='warn', invalid='warn')
        return np.mean(silhouette_scores)
    except Exception as e:
        print(f"Error in computing silhouette coefficient: {e}")
        return None

def plotSilhouette(k_values, silhouette_scores):
    """
    Plot silhouette coefficients against the number of clusters.

    Args:
        k_values (list): List of integers representing the number of clusters (k).
        silhouette_values (list): List of silhouette coefficients corresponding to each k value.

    Returns:
        None
    """
    if len(k_values) != len(silhouette_scores) or len(k_values) == 0:
        print("Error: Invalid input for plotting silhouette.")
        return
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, silhouette_scores, marker='o', color='blue')
    plt.title('Silhouette Coefficient vs. Number of Clusters'
              ' (Bisecting K-Means)')
    plt.xlabel('Number of Clusters (s)')
    plt.ylabel('Silhouette Coefficient')
    plt.grid(True)
    plt.savefig('Bisecting K-Means_silhouette_scores.png')
    plt.show()
    

if __name__ == "__main__":
    data = loadDataset()
    desired_clusters = 9
    hierarchical_clusters = [{'points': data, 'centroid': np.mean(data, axis=0), 'sse': np.inf}]
    centroids = [np.mean(data, axis=0)]

    while len(hierarchical_clusters) < desired_clusters:
        cluster_to_bisect = selectClusterToBisect(hierarchical_clusters)
        bisectCluster(cluster_to_bisect, hierarchical_clusters, centroids)

    silhouette_scores = []
    for num_clusters in range(1, desired_clusters + 1):
        clusters = extractClustersFromHierarchy(hierarchical_clusters[:num_clusters])
        flat_clusters = np.concatenate(clusters)
        labels = np.concatenate([[i] * len(cluster) for i, cluster in enumerate(clusters)])
        silhouette_score = silhouetteCoefficient(flat_clusters, labels)
        silhouette_scores.append(silhouette_score)

    plotSilhouette(list(range(1, desired_clusters + 1)), silhouette_scores)