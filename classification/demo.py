from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from classifier import (
    combine_features_and_labels,
    make_dataset,
    predict_forest,
    random_forest,
)
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split


def plot_dataset(x: np.ndarray, y: np.ndarray, output_path: Path) -> None:
    """
    Plot and save the generated classification dataset.

    :param x: np.ndarray
            Feature matrix with two columns.
    :param y: np.ndarray
            Class labels.
    :param output_path: Path
            Path used to save the figure.
    """
    plt.figure(figsize=(6, 5))
    plt.scatter(x[:, 0], x[:, 1], c=y, cmap="viridis", s=25, edgecolors="k")
    plt.xlabel("feature1")
    plt.ylabel("feature2")
    plt.title("Random forest classification dataset")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)


def plot_decision_boundary(
    forest: list,
    x: np.ndarray,
    y: np.ndarray,
    output_path: Path,
    grid_step: float = 0.03,
) -> None:
    """
    Plot and save the decision boundary of the random forest.

    :param forest: list
            Trained random forest.
    :param x: np.ndarray
            Feature matrix with two columns.
    :param y:np.ndarray
            Class labels.
    :param output_path:Path
            Path used to save the figure.
    :param grid_step:float
            Mesh grid step size.
    """
    x_min, x_max = x[:, 0].min() - 0.8, x[:, 0].max() + 0.8
    y_min, y_max = x[:, 1].min() - 0.8, x[:, 1].max() + 0.8
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, grid_step),
        np.arange(y_min, y_max, grid_step),
    )
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_labels = predict_forest(forest, grid_points).reshape(xx.shape)

    plt.figure(figsize=(6, 5))
    plt.contourf(xx, yy, grid_labels, alpha=0.25, cmap="viridis")
    plt.scatter(x[:, 0], x[:, 1], c=y, cmap="viridis", s=25, edgecolors="k")
    plt.xlabel("feature1")
    plt.ylabel("feature2")
    plt.title("Random forest decision boundary")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
) -> None:
    """
    Plot and save the confusion matrix.

    :param y_true: np.ndarray
                Ground-truth labels.
    :param y_pred:np.ndarray
                Predicted labels.
    :param output_path:Path
                Path used to save the figure.
    """
    matrix = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(5, 4))
    plt.imshow(matrix, cmap="Blues")
    plt.title("Confusion matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.colorbar()

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            plt.text(col, row, matrix[row, col], ha="center", va="center")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)


def count_feature_usage(
    tree: dict | int,
    feature_counts: np.ndarray,
) -> None:
    """Count how often each feature is used for tree splitting."""
    if not isinstance(tree, dict):
        return

    feature_counts[tree["feature_idx"]] += 1
    count_feature_usage(tree["left"], feature_counts)
    count_feature_usage(tree["right"], feature_counts)


def plot_feature_usage(forest: list, n_features: int, output_path: Path) -> None:
    """
    Plot and save split-feature usage counts.

    This is a simple interpretability view for the hand-written random forest.
    """
    feature_counts = np.zeros(n_features, dtype=int)
    for tree in forest:
        count_feature_usage(tree, feature_counts)

    plt.figure(figsize=(6, 4))
    plt.bar(np.arange(n_features), feature_counts, color="#4C78A8")
    plt.xlabel("Feature index")
    plt.ylabel("Split count")
    plt.title("Feature usage in random forest")
    plt.xticks(np.arange(n_features))
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)


def main() -> None:
    """Run the random forest classification demo."""
    assets_dir = Path(__file__).resolve().parent / "assets"
    assets_dir.mkdir(exist_ok=True)

    x, y = make_dataset(
        n_samples=300,
        n_features=2,
        n_classes=2,
        random_state=7,
    )
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=42, stratify=y
    )
    train_dataset = combine_features_and_labels(x_train, y_train)

    forest = random_forest(
        train_dataset,
        n_trees=15,
        max_depth=8,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
    )
    y_pred = predict_forest(forest, x_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Random forest accuracy: {accuracy:.3f}")

    plot_dataset(x, y, assets_dir / "random_forest_dataset.png")
    plot_decision_boundary(
        forest,
        x,
        y,
        assets_dir / "random_forest_decision_boundary.png",
    )
    plot_confusion_matrix(y_test, y_pred, assets_dir / "random_forest_confusion.png")
    plot_feature_usage(
        forest, x.shape[1], assets_dir / "random_forest_feature_usage.png"
    )

    if matplotlib.get_backend().lower() != "agg":
        plt.show()


if __name__ == "__main__":
    main()
