"""This file contains the posterior distributions for the different parameters in this example."""

import numpy as np
from .config import data, sigma_0, tau_0, mu_0, alpha_0, beta_0, T, K


Sy = np.sum(data, axis=1)
sigma_0_inv_2 = sigma_0 ** (-2)
tau_0_inv_2 = tau_0 ** (-2)


def mu_k_dist(k: int, mu: np.array, theta: np.array, sigma_inv_2: np.array) -> float:
    """Provides the posterior distribution for \mu_k"""
    x = 1 / (T * sigma_inv_2[k] + sigma_0_inv_2)
    return np.random.normal((sigma_inv_2[k] * Sy[k] + theta[0] * sigma_0_inv_2) * x, x)


def theta_dist(k: int, mu: np.array, theta: np.array, sigma_inv_2: np.array) -> float:
    """Provides the posterior distribution for \theta"""
    x = 1 / (K * sigma_0_inv_2 + tau_0_inv_2)
    return np.random.normal((sigma_0_inv_2 * np.sum(mu) + mu_0 * tau_0_inv_2) * x, x)


def sigma_inv_2_dist(
    k: int, mu: np.array, theta: np.array, sigma_inv_2: np.array
) -> float:
    """Provides the posterior distribution for \sigma_k^{-2}"""
    return np.random.gamma(
        alpha_0 + T / 2, beta_0 + 0.5 * np.sum(np.square(data[k] - mu[k]))
    )
