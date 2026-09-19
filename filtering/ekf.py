from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EPS = 1e-8


@dataclass
class ModelState:
    """State and covariance for one IMM sub-model."""

    x: np.ndarray
    P: np.ndarray


@dataclass
class IMMParameters:
    """Parameters and internal states for a two-model IMM filter."""

    cv_H: np.ndarray
    cv_Q: np.ndarray
    cv_R: np.ndarray
    ctrv_H: np.ndarray
    ctrv_Q: np.ndarray
    ctrv_R: np.ndarray
    P_model: np.ndarray
    mu_weight: np.ndarray
    models: list[ModelState]


def normalize_heading(x: np.ndarray) -> np.ndarray:
    """
    Normalize the heading vector stored in state elements [cos_theta, sin_theta]

    Parameters
    -----------
    x : np.ndarray
        State vector [px, py, v, cos_theta, sin_theta, yaw_rate].

    Returns
    ----------
    np.ndarray
        State vector with a normalized heading vector.
    """
    norm_heading = np.linalg.norm(x[3:5])
    if norm_heading > EPS:
        x[3:5] = x[3:5] / norm_heading
    else:
        x[3:5] = np.array([1.0, 0.0])
    return x


def cv_kf(
    cv_parameter: dict[str, np.ndarray],
    z: np.ndarray,
    x: np.ndarray,
    P: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run one constant-velocity Kalman filter step.

    Parameters
    ----------
    cv_parameter : dict[str, np.ndarray]
        Dictionary containing ``cv_H``, ``cv_Q``, and ``cv_R``.
    z : np.ndarray
        Position measurement [px, py].
    x : np.ndarray
        Previous state [px, py, v, cos_theta, sin_theta, yaw_rate].
    P : np.ndarray
        Previous covariance matrix.
    dt : float
        Sampling interval.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        State transition matrix, updated state, updated covariance, and predicted state.
    """
    H = cv_parameter["cv_H"]
    Q = cv_parameter["cv_Q"]
    R = cv_parameter["cv_R"]

    v = x[2]
    cos_i = x[3]
    sin_i = x[4]

    A_delta = np.array(
        [
            [0.0, 0.0, cos_i, v, 0.0, 0.0],
            [0.0, 0.0, sin_i, 0.0, v, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    A = np.eye(6) + A_delta * dt

    x_pred = A @ x
    P_pred = A @ P @ A.T + Q

    innovation = z - H @ x_pred
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)

    x_update = x_pred + K @ innovation
    P_update = (np.eye(6) - K @ H) @ P_pred

    x_update = normalize_heading(x_update)
    return A, x_update, P_update, x_pred


def ctrv_kf(
    ctrv_parameter: dict[str, np.ndarray],
    z: np.ndarray,
    x: np.ndarray,
    P: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run one constant-turn-rate-and-velocity EKF step.

    Parameters
    ----------
    ctrv_parameter : dict[str, np.ndarray]
        Dictionary containing ``ctrv_H``, ``ctrv_Q``, and ``ctrv_R``.
    z : np.ndarray
        Position measurement [px, py].
    x : np.ndarray
        Previous state [px, py, v, cos_theta, sin_theta, yaw_rate].
    P : np.ndarray
        Previous covariance matrix.
    dt : float
        Sampling interval.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Jacobian transition matrix, updated state, updated covariance,
        and predicted state.
    """
    H = ctrv_parameter["ctrv_H"]
    Q = ctrv_parameter["ctrv_Q"]
    R = ctrv_parameter["ctrv_R"]

    v_i = x[2]
    cos_i = x[3]
    sin_i = x[4]
    w_i = x[5]
    if abs(w_i) < EPS:
        w_i = EPS if w_i >= 0.0 else -EPS

    wt = w_i * dt
    cos_wt = np.cos(wt)
    sin_wt = np.sin(wt)

    A_13 = (sin_i * (cos_wt - 1.0) + cos_i * sin_wt) / w_i
    A_14 = v_i * sin_wt / w_i
    A_15 = v_i * (cos_wt - 1.0) / w_i
    A_16 = (
        -v_i * (sin_i * (cos_wt - 1.0) + cos_i * sin_wt) / (w_i**2)
        + v_i * dt * (-sin_i * sin_wt + cos_i * cos_wt) / w_i
    )

    A_23 = (cos_i * (1.0 - cos_wt) + sin_i * sin_wt) / w_i
    A_24 = -v_i * (cos_wt - 1.0) / w_i
    A_25 = v_i * sin_wt / w_i
    A_26 = (
        -v_i * (cos_i * (1.0 - cos_wt) + sin_i * sin_wt) / (w_i**2)
        + v_i * dt * (sin_i * cos_wt + cos_i * sin_wt) / w_i
    )

    A = np.array(
        [
            [1.0, 0.0, A_13, A_14, A_15, A_16],
            [0.0, 1.0, A_23, A_24, A_25, A_26],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [
                0.0,
                0.0,
                0.0,
                cos_wt,
                -sin_wt,
                -cos_i * dt * sin_wt - sin_i * dt * cos_wt,
            ],
            [
                0.0,
                0.0,
                0.0,
                sin_wt,
                cos_wt,
                -sin_i * dt * sin_wt + cos_i * dt * cos_wt,
            ],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ]
    )

    x_pred = A @ x
    P_pred = A @ P @ A.T + Q

    innovation = z - H @ x_pred
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)

    x_update = x_pred + K @ innovation
    P_update = (np.eye(6) - K @ H) @ P_pred

    x_update[2] = abs(np.linalg.norm(x_update[3:5]) * x_update[2])
    x_update = normalize_heading(x_update)
    return A, x_update, P_update, x_pred


def gaussian_likelihood(v: np.ndarray, S: np.ndarray) -> float:
    """
    Compute the Gaussian likelihood for a measurement residual.

    Parameters
    ----------
    v : np.ndarray
        Measurement residual.
    S : np.ndarray
        Innovation covariance matrix.

    Returns
    -------
    float
        Likelihood value.
    """
    dimension = v.shape[0]
    det_s = max(np.linalg.det(S), EPS)
    exponent = -0.5 * float(v.T @ np.linalg.inv(S) @ v)
    normalizer = np.sqrt(((2.0 * np.pi) ** dimension) * det_s)
    return float(np.exp(exponent) / normalizer)


def imm_filter(
    parameter: IMMParameters,
    z: np.ndarray,
    dt: float,
) -> tuple[IMMParameters, np.ndarray, np.ndarray]:
    """
    Run one two-model IMM filter step using CV-KF and CTRV-EKF models.

    Parameters
    ----------
    parameter : IMMParameters
        IMM parameters and internal model states.
    z : np.ndarray
        Position measurement [px, py].
    dt : float
        Sampling interval.

    Returns
    -------
    tuple[IMMParameters, np.ndarray, np.ndarray]
        Updated IMM parameters, fused state, and fused covariance.
    """
    P_model = parameter.P_model
    mu_weight = parameter.mu_weight
    models = parameter.models

    c_bar = P_model @ mu_weight
    c_bar = np.maximum(c_bar, EPS)

    mix_prob = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            mix_prob[i, j] = P_model[i, j] * mu_weight[j] / c_bar[i]

    previous_models = [ModelState(model.x.copy(), model.P.copy()) for model in models]

    for i in range(2):
        mixed_x = np.zeros(6)
        for j in range(2):
            mixed_x += mix_prob[i, j] * previous_models[j].x
        models[i].x = normalize_heading(mixed_x)

    for i in range(2):
        mixed_P = np.zeros((6, 6))
        for j in range(2):
            delta_x = models[i].x - previous_models[j].x
            mixed_P += mix_prob[i, j] * (
                previous_models[j].P + np.outer(delta_x, delta_x)
            )
        models[i].P = mixed_P

    cv_params = {
        "cv_H": parameter.cv_H,
        "cv_Q": parameter.cv_Q,
        "cv_R": parameter.cv_R,
    }
    ctrv_params = {
        "ctrv_H": parameter.ctrv_H,
        "ctrv_Q": parameter.ctrv_Q,
        "ctrv_R": parameter.ctrv_R,
    }

    _, models[0].x, models[0].P, cv_x_pred = cv_kf(
        cv_params, z, models[0].x, models[0].P, dt
    )

    _, models[1].x, models[1].P, ctrv_x_pred = ctrv_kf(
        ctrv_params, z, models[1].x, models[1].P, dt
    )

    cv_residual = z - parameter.cv_H @ cv_x_pred
    ctrv_residual = z - parameter.ctrv_H @ ctrv_x_pred

    cv_S = parameter.cv_H @ models[0].P @ parameter.cv_H.T + parameter.cv_R
    ctrv_S = parameter.ctrv_H @ models[1].P @ parameter.ctrv_H.T + parameter.ctrv_R

    likelihood = np.array(
        [
            gaussian_likelihood(cv_residual, cv_S),
            gaussian_likelihood(ctrv_residual, ctrv_S),
        ]
    )

    c = float(np.sum(likelihood * c_bar))
    if c <= EPS:
        mu_weight = np.array([0.5, 0.5])
    else:
        mu_weight = likelihood * c_bar / c

    fused_x = models[0].x * mu_weight[0] + models[1].x * mu_weight[1]
    fused_x = normalize_heading(fused_x)

    fused_P = np.zeros((6, 6))
    for i in range(2):
        delta_x = fused_x - models[i].x
        fused_P += mu_weight[i] * (models[i].P + np.outer(delta_x, delta_x))

    parameter.mu_weight = mu_weight
    parameter.models = models
    return parameter, fused_x, fused_P
