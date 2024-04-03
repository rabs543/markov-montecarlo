import numpy as np
import matplotlib.pyplot as plt

from src.gibbs.gibbs import GibbsData, gibbs_sampler

from .config import mu_0, K, alpha_0, beta_0
from .distributions import mu_k_dist, theta_dist, sigma_inv_2_dist

# We will run a Gibbs sampler with m = 2K + 1. This is because, for each iteration, we have K mu_k, 1 theta, and 1 sigma_k.
# For initial values, we shall use:
# mu_k = mu_0
# theta = mu_0
# sigma_k = alpha_0 / beta_0 (mean of Gamma distribution)

x_0 = [np.array([mu_0] * K), np.array([mu_0]), np.array([alpha_0 / beta_0] * K)]


g = gibbs_sampler(
    n=1000,
    parameters={
        "mu": GibbsData(x_0=np.array([mu_0] * K), conditional=mu_k_dist),
        "theta": GibbsData(x_0=np.array([mu_0]), conditional=theta_dist),
        "sigma_inv_2": GibbsData(
            x_0=np.array([alpha_0 / beta_0] * K), conditional=sigma_inv_2_dist
        ),
    },
)

# Plot a histogram
counts, bins = np.histogram(g["theta"].flatten())
plt.stairs(counts, bins)
plt.show()