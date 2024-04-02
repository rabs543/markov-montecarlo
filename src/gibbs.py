from typing import Callable, List, get_args

import numpy as np


def gibbs_sampler(
    m: int,  # The dimension of the sampler
    x_0: np.array,  # The initial value
    T: int,  # The time to calculate to
    conditional: Callable[..., float],  # The conditional distributions p(x_i | x_{-i})
):
    if len(x_0) != m:
        raise TypeError("x_0 should have length m")

    if len(get_args(conditional)) != m - 1:
        raise TypeError("conditional should take m-1 arguments")

    result = np.zeros((T + 1) * m)
    result[:m] = x_0
    for i in range(T * m):
        result[m + i] = conditional(*result[i : m + i])

    result.reshape(m, T)

    return result
