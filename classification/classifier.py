from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

TASK_CLASSIFICATION = "classification"


def make_dataset(
    n_samples: int = 300,
    n_features: int = 8,
    n_classes: int = 2,
    random_state: int | None = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a synthetic classification dataset.

    :param n_samples: int
            Number of samples.
    :param n_features: int
            Number of features columns.
    :param n_classes: int
            Number of classes columns.
    :param random_state:  int | None
            Random seed used by scikit-learn.
    :return:
            tuple[np.ndarray, np.ndarray]
            Feature matrix and label vector.
    """
    from sklearn.datasets import make_classification

    x, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=max(2, min(n_features, n_classes + 1)),
        n_redundant=0,
        n_classes=n_classes,
        random_state=random_state,
    )

    return x, y


def combine_features_and_labels(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Combine features and labels into one matrix.

    The last column is treated as the label column by the tree-building code.
    """

    return np.concatenate([x, y.reshape(-1, 1)], axis=1)


def bootstrap_samples(
    dataset: np.ndarray,
    n_samples: int,
    rng: np.random.Generator | None = None,
) -> list[np.ndarray]:
    """
    Generate bootstrap samples for bagging.

    :param dataset: np.ndarray
            Dataset whose last column is the label.
    :param n_samples: int
            Number of bootstrap datasets to generate.
    :param rng: np.random.Generator | None
            Random  number generator.
    :return:
            list[np.ndarray]
            Bootstrap datasets sampled with replacement.
    """
    if dataset.size == 0:
        raise ValueError("dataset must not be empty")

    rng = np.random.default_rng() if rng is None else rng
    n_rows = dataset.shape[0]
    samples = []

    for _ in range(n_samples):
        indices = rng.integers(0, n_rows, size=n_rows)
        samples.append(dataset[indices])

    return samples


def split_dataset(
    dataset: np.ndarray,
    feature_idx: int,
    split_value: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split a dataset by one feature and threshold.

    Samples greater than the threshold go to the left branch; all other samples
    go to the right branch.
    """
    left_mask = dataset[:, feature_idx] > split_value
    right_mask = ~left_mask
    return dataset[left_mask], dataset[right_mask]


def majority_class(dataset: np.ndarray) -> int:
    """
    Return the most frequent class label in a dataset.

    :param dataset: np.ndarray
            Dataset whose last column is the label.
    :return:
            int
            Majority class label.
    """
    if dataset.size == 0:
        raise ValueError("dataset must not be empty")

    labels = dataset[:, -1].astype(int)
    return Counter(labels).most_common(1)[0][0]


def gini_index(dataset: np.ndarray) -> float:
    """
    Compute the Gini impurity of a dataset.

    :param dataset: np.ndarray
                Dataset whose last column is the label.
    :return:
            float
            Gini impurity.
    """
    if len(dataset) == 0:
        return 0.0

    labels = dataset[:, -1]
    _, counts = np.unique(labels, return_counts=True)
    probabilities = counts / len(dataset)
    return float(1.0 - np.sum(probabilities**2))


def weighted_gini(left: np.ndarray, right: np.ndarray) -> float:
    """Compute the weighted Gini impurity after a binary split."""
    total = len(left) + len(right)
    if total == 0:
        return 0.0

    return (len(left) / total) * gini_index(left) + (len(right) / total) * gini_index(
        right
    )


def select_best_split(
    dataset: np.ndarray,
    n_features: int,
    min_samples_leaf: int = 1,
    rng: np.random.Generator | None = None,
) -> tuple[int | None, float | int]:
    """
    Select the best split from a random subset of features.

    :param dataset: np.ndarray
                Dataset whose last column is the label.
    :param n_features: int
                Number of random features checked by one tree node.
    :param min_samples_leaf:
                Minimum number of samples allowed in each child node.
    :param rng: np.random.Generator | None
                Random number generator.
    :return:
            tuple[int, float | int]
            Best feature index and split value. If no valid split exists, the first
            value is None and the second value is the leaf prediction.
    """
    if dataset.size == 0:
        raise ValueError("dataset must not be empty")

    labels = dataset[:, -1]
    if len(np.unique(labels)) == 1:
        return None, int(labels[0])

    rng = np.random.default_rng() if rng is None else rng
    total_features = dataset.shape[1] - 1
    n_features = max(1, min(n_features, total_features))
    feature_indices = rng.choice(total_features, size=n_features, replace=False)

    best_score = np.inf
    best_feature = None
    best_value = None

    for feature_idx in feature_indices:
        for split_value in np.unique(dataset[:, feature_idx]):
            left, right = split_dataset(dataset, feature_idx, split_value)
            if len(left) < min_samples_leaf or len(right) < min_samples_leaf:
                continue

            score = weighted_gini(left, right)
            if score < best_score:
                best_score = score
                best_feature = int(feature_idx)
                best_value = split_value

    if best_value is None:
        return None, majority_class(dataset)

    return best_feature, best_value


def create_tree(
    dataset: np.ndarray,
    n_features: int,
    max_depth: int = 10,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    rng: np.random.Generator | None = None,
) -> dict[str, Any] | int:
    """
    Build one CART classification tree.

    :param dataset:np.ndarray
                Dataset whose last column is the label.
    :param n_features:int
                Number of random features checked by one tree node.
    :param max_depth:int
                Maximum tree depth.
    :param min_samples_split:int
                Minimum number of samples required to split a node.
    :param min_samples_leaf:int
                Minimum number of samples allowed in each child node.
    :param rng:np.random.Generator | None
                Random number generator.
    :return:
            dict[str, Any] | int
            Tree dictionary or leaf class label.
    """
    if dataset.size == 0:
        raise ValueError("dataset must not be empty")

    labels = dataset[:, -1]
    if (
        max_depth <= 0
        or len(dataset) < min_samples_split
        or len(np.unique(labels)) == 1
    ):
        return majority_class(dataset)

    feature_idx, split_value = select_best_split(
        dataset,
        n_features=n_features,
        min_samples_leaf=min_samples_leaf,
        rng=rng,
    )
    if feature_idx is None:
        return int(split_value)

    left, right = split_dataset(dataset, feature_idx, float(split_value))
    return {
        "feature_idx": feature_idx,
        "split_value": float(split_value),
        "left": create_tree(
            left,
            n_features=n_features,
            max_depth=max_depth - 1,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            rng=rng,
        ),
        "right": create_tree(
            right,
            n_features=n_features,
            max_depth=max_depth - 1,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            rng=rng,
        ),
    }


def random_forest(
    dataset: np.ndarray,
    n_trees: int = 20,
    n_features: int | None = None,
    max_depth: int = 10,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    random_state: int | None = 42,
) -> list[dict[str, Any] | int]:
    """
    Train a random forest classifier.

    :param dataset: np.ndarray
                Dataset whose last column is the label.
    :param n_trees: int
                Number of trees in the forest.
    :param n_features: int | None
                Number of random features checked at each split.
                If None, sqrt(d) is used.
    :param max_depth: int
                Maximum depth of each tree.
    :param min_samples_split: int
                Minimum number of samples required to split a node.
    :param min_samples_leaf: int
                Minimum number of samples allowed in each child node.
    :param random_state: int | None
                Random seed.
    :return:
            list[dict[str, Any] | int]
            Trained decision trees.
    """
    if dataset.size == 0:
        raise ValueError("dataset must not be empty")
    if n_trees <= 0:
        raise ValueError("n_trees must be positive")

    total_features = dataset.shape[1] - 1
    if n_features is None:
        n_features = max(1, int(np.sqrt(total_features)))

    rng = np.random.default_rng(random_state)
    forest = []

    for sample in bootstrap_samples(dataset, n_trees, rng=rng):
        tree = create_tree(
            sample,
            n_features=n_features,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            rng=rng,
        )
        forest.append(tree)

    return forest


def predict_tree(tree: dict[str, Any] | int, sample: np.ndarray) -> int:
    """
    Predict one sample with one decision tree.

    :param tree: dict[str, Any] | int
            Trained decision tree or leaf label.
    :param sample: np.ndarray
            One feature vector.
    :return:
            int
            Predicted class label.
    """
    if not isinstance(tree, dict):
        return int(tree)

    if sample[tree["feature_idx"]] > tree["split_value"]:
        return predict_tree(tree["left"], sample)
    return predict_tree(tree["right"], sample)


def predict_tree_batch(tree: dict[str, Any] | int, x: np.ndarray) -> np.ndarray:
    """Predict a batch of samples with one decision tree."""
    return np.array([predict_tree(tree, sample) for sample in x])


def predict_forest(forest: list[dict[str, Any] | int], x: np.ndarray) -> np.ndarray:
    """
    Predict class labels with a random forest.

    :param forest: list[dict[str, Any] | int]
            Trained random forest.
    :param x: np.ndarray
            Feature matrix.
    :return:
            np.ndarray
            Predicted class label.
    """
    if not forest:
        raise ValueError("forest must not be empty")

    tree_predictions = np.vstack([predict_tree_batch(tree, x) for tree in forest])
    predictions = []

    for sample_votes in tree_predictions.T:
        predictions.append(Counter(sample_votes).most_common(1)[0][0])

    return np.array(predictions, dtype=int)
