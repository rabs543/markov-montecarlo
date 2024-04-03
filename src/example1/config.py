"""This file contains the initial guesses and data for this example."""

import numpy as np

K = 14  # There are 14 football teams
T = 5  # 5 years of data

sigma_0 = 10
alpha_0 = 10 ** (-5)
beta_0 = 10 ** (-3)
mu_0 = 60
tau_0 = 20

data = np.genfromtxt("src\II-10-9-2023football.csv", delimiter=",", skip_header=1, usecols=(1, 2, 3, 4, 5))