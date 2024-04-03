import numpy as np

from gibbs import gibbs_sampler

K = 14  # There are 14 football teams
T = 5  # 5 years of data

sigma_0 = 10
alpha_0 = 10 ** (-5)
beta_0 = 10 ** (-3)
mu_0 = 60
tau_0 = 20

data = np.genfromtxt("src\II-10-9-2023football.csv", delimiter=",", skip_header=1, usecols=(1, 2, 3, 4, 5))

# We will run a Gibbs sampler with m = 2K + 1. This is because, for each iteration, we have K mu_k, 1 theta, and 1 sigma_k.
# For initial values, we shall use:
# mu_k = mu_0
# theta = mu_0
# sigma_k = alpha_0 / beta_0 (mean of Gamma distribution)

x_0 = [np.array([mu_0] * K), np.array([mu_0]), np.array([alpha_0 / beta_0] * K)]

# For the conditional distributions:
# mu_k:
Sy = np.sum(data, axis=1)
sigma_0_inv_2 = sigma_0 ** (-2)
tau_0_inv_2 = tau_0 ** (-2)


def mu_k_dist(k: int, mu: np.array, theta: np.array, sigma_inv_2: np.array) -> float:
    x = 1 / (T * sigma_inv_2[k] + sigma_0_inv_2)
    return np.random.normal((sigma_inv_2[k] * Sy[k] + theta[0] * sigma_0_inv_2) * x, x)


def theta_dist(k: int, mu: np.array, theta: np.array, sigma_inv_2: np.array) -> float:
    x = 1 / (K * sigma_0_inv_2 + tau_0_inv_2)
    return np.random.normal((sigma_0_inv_2 * np.sum(mu) + mu_0 * tau_0_inv_2) * x, x)


def sigma_inv_2_dist(k: int, mu: np.array, theta: np.array, sigma_inv_2: np.array) -> float:
    return np.random.gamma(
        alpha_0 + T / 2, beta_0 + 0.5 * np.sum(np.square(data[k] - mu[k]))
    )


g = gibbs_sampler(
    x_0=x_0, n=10, conditional=[mu_k_dist, theta_dist, sigma_inv_2_dist]
)

print(g)