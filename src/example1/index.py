import numpy as np
import matplotlib as plt

from gibbs import gibbs_sampler

from config import mu_0, K, alpha_0, beta_0
from distributions import mu_k_dist, theta_dist, sigma_inv_2_dist

# We will run a Gibbs sampler with m = 2K + 1. This is because, for each iteration, we have K mu_k, 1 theta, and 1 sigma_k.
# For initial values, we shall use:
# mu_k = mu_0
# theta = mu_0
# sigma_k = alpha_0 / beta_0 (mean of Gamma distribution)

x_0 = [np.array([mu_0] * K), np.array([mu_0]), np.array([alpha_0 / beta_0] * K)]

g = gibbs_sampler(
    x_0=x_0, n=10, conditional=[mu_k_dist, theta_dist, sigma_inv_2_dist]
)



print(g)