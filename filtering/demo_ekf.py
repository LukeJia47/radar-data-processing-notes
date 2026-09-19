from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from ekf import IMMParameters, ModelState, ctrv_kf, cv_kf, imm_filter


def generate_truth(dt: float, steps: int) -> np.ndarray:
    """
    Generate the target trajectory used by the EKF demo.

    Parameters
    ----------
    dt : float
        Sampling interval.
    steps : int
        Number of simulation steps.

    Returns
    -------
    np.ndarray
        Ground-truth states with shape (6, steps).
    """
    truth = np.zeros((6, steps))
    position = np.array([0.0, 0.0])

    for i in range(steps):
        step_id = i + 1
        if step_id < 50:
            velocity = np.array([5.0, 0.0])
        elif step_id < 150:
            radius = 5.0
            speed = 10.0
            yaw_rate = speed / radius
            angle = (step_id - 50) / 10.0 * yaw_rate
            velocity = np.array([speed * np.cos(angle), speed * np.sin(angle)])
        else:
            velocity = np.array([3.0, 5.0])

        position = position + velocity * dt
        speed_norm = np.linalg.norm(velocity)
        if speed_norm > 0.0:
            heading = velocity / speed_norm
        else:
            heading = np.array([1.0, 0.0])

        truth[:, i] = np.array(
            [position[0], position[1], speed_norm, heading[0], heading[1], 0.0]
        )
    return truth


def predict_cv_trajectory(
    parameter: dict[str, np.ndarray],
    x: np.ndarray,
    P: np.ndarray,
    z: np.ndarray,
    dt: float,
    horizon: int,
) -> np.ndarray:
    """Predict a short CV trajectory for visualization."""
    prediction = np.zeros((6, horizon + 1))
    prediction[:, 0] = x

    for i in range(horizon):
        A, _, _, _ = cv_kf(parameter, z, prediction[:, i], P, dt)
        prediction[:, i + 1] = A @ prediction[:, i]

    return prediction


def predict_ctrv_trajectory(
    parameter: dict[str, np.ndarray],
    x: np.ndarray,
    P: np.ndarray,
    z: np.ndarray,
    dt: float,
    horizon: int,
) -> np.ndarray:
    """Predict a short CTRV trajectory for visualization."""
    prediction = np.zeros((6, horizon + 1))
    prediction[:, 0] = x

    for i in range(horizon):
        A, _, _, _ = ctrv_kf(parameter, z, prediction[:, i], P, dt)
        prediction[:, i + 1] = A @ prediction[:, i]

    return prediction


def predict_imm_trajectory(
    cv_parameter: dict[str, np.ndarray],
    ctrv_parameter: dict[str, np.ndarray],
    mu_weight: np.ndarray,
    x: np.ndarray,
    cv_P: np.ndarray,
    ctrv_P: np.ndarray,
    z: np.ndarray,
    dt: float,
    horizon: int,
) -> np.ndarray:
    """Predict a short IMM trajectory for visualization."""
    prediction = np.zeros((6, horizon + 1))
    prediction[:, 0] = x

    for i in range(horizon):
        cv_A, _, _, _ = cv_kf(cv_parameter, z, prediction[:, i], cv_P, dt)
        ctrv_A, _, _, _ = ctrv_kf(ctrv_parameter, z, prediction[:, i], ctrv_P, dt)
        prediction[:, i + 1] = (
            mu_weight[0] * cv_A @ prediction[:, i]
            + mu_weight[1] * ctrv_A @ prediction[:, i]
        )

    return prediction


def plot_tracking_result(
    truth: np.ndarray,
    measurements: np.ndarray,
    cv_estimation: np.ndarray,
    ctrv_estimation: np.ndarray,
    imm_estimation: np.ndarray,
    cv_prediction: np.ndarray,
    ctrv_prediction: np.ndarray,
    imm_prediction: np.ndarray,
    output_path: Path,
) -> None:
    """Plot and save the final tracking result."""
    plt.figure(figsize=(10, 7))
    plt.plot(truth[0], truth[1], color="black", linewidth=2, label="Truth trajectory")
    plt.plot(
        measurements[0],
        measurements[1],
        ".",
        markersize=6,
        linewidth=2,
        label="Measurements",
    )

    plt.plot(cv_estimation[0, -1], cv_estimation[1, -1], "bo", linewidth=2)
    plt.plot(ctrv_estimation[0, -1], ctrv_estimation[1, -1], "go", linewidth=2)
    plt.plot(imm_estimation[0, -1], imm_estimation[1, -1], "ro", linewidth=2)

    plt.plot(cv_estimation[0], cv_estimation[1], "b", linewidth=2, label="CV-KF")
    plt.plot(ctrv_estimation[0], ctrv_estimation[1], "g", linewidth=2, label="CTRV-EKF")
    plt.plot(imm_estimation[0], imm_estimation[1], "r", linewidth=2, label="IMM")

    plt.plot(
        cv_prediction[0],
        cv_prediction[1],
        "b--",
        linewidth=2,
        label="CV-KF prediction",
    )
    plt.plot(
        ctrv_prediction[0],
        ctrv_prediction[1],
        "g--",
        linewidth=2,
        label="CTRV-EKF prediction",
    )
    plt.plot(
        imm_prediction[0],
        imm_prediction[1],
        "r--",
        linewidth=2,
        label="IMM prediction",
    )

    plt.axis([-5, 65, -14, 30])
    plt.axis("equal")
    plt.xlabel("X / m")
    plt.ylabel("Y / m")
    plt.grid(True)
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)


def plot_position_error(
    time: np.ndarray,
    truth: np.ndarray,
    cv_estimation: np.ndarray,
    ctrv_estimation: np.ndarray,
    imm_estimation: np.ndarray,
    output_path: Path,
) -> None:
    """Plot and save the position estimation error."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    labels = ["px error (m)", "py error (m)"]

    for i, ax in enumerate(axes):
        ax.plot(time, cv_estimation[i] - truth[i], "b", linewidth=2, label="CV-KF")
        ax.plot(time, ctrv_estimation[i] - truth[i], "g", linewidth=2, label="CTRV-EKF")
        ax.plot(time, imm_estimation[i] - truth[i], "r", linewidth=2, label="IMM")
        ax.set_ylabel(labels[i])
        ax.grid(True)

    axes[-1].set_xlabel("t (s)")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)


def plot_imm_weights(
    time: np.ndarray, mu_history: np.ndarray, output_path: Path
) -> None:
    """Plot and save the IMM model weights."""
    plt.figure(figsize=(10, 5))
    plt.plot(time, mu_history[0], linewidth=2, label="IMM-CV")
    plt.plot(time, mu_history[1], linewidth=2, label="IMM-CTRV")
    plt.xlabel("t (s)")
    plt.ylabel("weight value")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)


def main() -> None:
    """Run the EKF and IMM tracking demo."""
    dt = 0.1
    steps = 200
    time = np.arange(1, steps + 1) * dt
    assets_dir = Path(__file__).resolve().parent / "assets"
    assets_dir.mkdir(exist_ok=True)

    truth = generate_truth(dt, steps)
    measurements = truth[:2]

    cv_estimation = np.zeros((6, steps + 1))
    cv_estimation[3, 0] = 1.0
    cv_P = np.eye(6)
    cv_parameter = {
        "cv_H": np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ),
        "cv_Q": np.diag([0.1, 0.1, 1.0, 0.01, 0.01, 0.0]),
        "cv_R": np.eye(2) * 1.0,
    }

    ctrv_estimation = np.zeros((6, steps + 1))
    ctrv_estimation[3, 0] = 1.0
    ctrv_estimation[5, 0] = 0.1
    ctrv_P = np.eye(6)
    ctrv_parameter = {
        "ctrv_H": cv_parameter["cv_H"].copy(),
        "ctrv_Q": np.diag([0.1, 0.1, 1.0, 0.01, 0.01, 1.0]),
        "ctrv_R": np.eye(2) * 1.2,
    }

    imm_estimation = np.zeros((6, steps + 1))
    imm_estimation[3, 0] = 1.0
    imm_estimation[5, 0] = 0.1
    imm_parameter = IMMParameters(
        cv_H=cv_parameter["cv_H"],
        cv_Q=cv_parameter["cv_Q"],
        cv_R=cv_parameter["cv_R"],
        ctrv_H=ctrv_parameter["ctrv_H"],
        ctrv_Q=ctrv_parameter["ctrv_Q"],
        ctrv_R=ctrv_parameter["ctrv_R"],
        P_model=np.array([[0.96, 0.04], [0.04, 0.96]]),
        mu_weight=np.array([0.7, 0.3]),
        models=[
            ModelState(x=np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]), P=np.eye(6)),
            ModelState(x=np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.1]), P=np.eye(6)),
        ],
    )

    mu_history = np.zeros((2, steps))
    prediction_horizon = 10
    cv_prediction = np.zeros((6, prediction_horizon + 1))
    ctrv_prediction = np.zeros((6, prediction_horizon + 1))
    imm_prediction = np.zeros((6, prediction_horizon + 1))

    for i in range(steps):
        z = measurements[:, i]

        if i == 169:
            ctrv_estimation[5, i] = 0.1

        cv_prediction = predict_cv_trajectory(
            cv_parameter,
            cv_estimation[:, i],
            cv_P,
            z,
            dt,
            prediction_horizon,
        )
        ctrv_prediction = predict_ctrv_trajectory(
            ctrv_parameter,
            ctrv_estimation[:, i],
            ctrv_P,
            z,
            dt,
            prediction_horizon,
        )
        imm_prediction = predict_imm_trajectory(
            cv_parameter,
            ctrv_parameter,
            imm_parameter.mu_weight,
            imm_estimation[:, i],
            cv_P,
            ctrv_P,
            z,
            dt,
            prediction_horizon,
        )

        _, cv_estimation[:, i + 1], cv_P, _ = cv_kf(
            cv_parameter, z, cv_estimation[:, i], cv_P, dt
        )
        _, ctrv_estimation[:, i + 1], ctrv_P, _ = ctrv_kf(
            ctrv_parameter, z, ctrv_estimation[:, i], ctrv_P, dt
        )
        imm_parameter, imm_estimation[:, i + 1], _ = imm_filter(imm_parameter, z, dt)
        mu_history[:, i] = imm_parameter.mu_weight

    plot_tracking_result(
        truth,
        measurements,
        cv_estimation[:, 1:],
        ctrv_estimation[:, 1:],
        imm_estimation[:, 1:],
        cv_prediction,
        ctrv_prediction,
        imm_prediction,
        assets_dir / "ekf_tracking_result.png",
    )
    plot_position_error(
        time,
        truth,
        cv_estimation[:, 1:],
        ctrv_estimation[:, 1:],
        imm_estimation[:, 1:],
        assets_dir / "ekf_position_error.png",
    )
    plot_imm_weights(time, mu_history, assets_dir / "ekf_imm_weights.png")

    if matplotlib.get_backend().lower() != "agg":
        plt.show()


if __name__ == "__main__":
    main()
