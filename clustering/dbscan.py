import numpy as np


def distance_matrix(x: np.ndarray) -> np.ndarray:
    """
    Compute the pairwise Euclidean distance matrix.

    Parameters
    ----------
    x : np.ndarray
        Input data with shape (n_samples, n_features).

    Returns
    -------
    np.ndarray
        Pairwise distance matrix with shape (n_samples, n_samples).
    """
    gram_matrix = x @ x.T
    squared_norms = np.diag(gram_matrix)
    squared_distances = (
        squared_norms[:, None] + squared_norms[None, :] - 2 * gram_matrix
    )

    # Floating-point errors may create tiny negative values before sqrt.
    squared_distances = np.maximum(squared_distances, 0.0)

    return np.sqrt(squared_distances)


def dbscan(x: np.ndarray, eps: float, min_pts: int) -> np.ndarray:
    """
    Perform DBSCAN clustering.

    Parameters
    ----------
    x : np.ndarray
        Input data with shape (n_samples, n_features).
    eps : float
        Radius of the epsilon neighborhood.
    min_pts : int
        Minimum number of points required to form a core point.

    Returns
    -------
    np.ndarray
        Cluster labels for each sample. Noise points are labeled as -1.
    """
    n_samples, _ = x.shape
    distance_mat = distance_matrix(x)

    neighbor_counts = np.sum(distance_mat <= eps, axis=1)
    core_point_indices = np.where(neighbor_counts >= min_pts)[0]

    labels = np.full(n_samples, -1, dtype=int)
    cluster_id = 0

    for point_id in core_point_indices:
        if labels[point_id] != -1:
            continue

        labels[point_id] = cluster_id
        seeds = set(np.where((distance_mat[:, point_id] <= eps) & (labels == -1))[0])

        while seeds:
            new_point = seeds.pop()
            labels[new_point] = cluster_id

            neighbors = np.where(distance_mat[:, new_point] <= eps)[0]
            if len(neighbors) < min_pts:
                continue

            for neighbor_id in neighbors:
                if labels[neighbor_id] == -1:
                    seeds.add(neighbor_id)

        cluster_id += 1

    return labels
