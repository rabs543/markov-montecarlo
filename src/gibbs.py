from typing import Callable, List, Union, get_args

import numpy as np


def gibbs_sampler(
    x_0: List[np.array],  # The initial value
    n: int,  # The length of the Markov chain
    conditional: List[
        Callable[..., float]
    ],  # The conditional distributions p(x_i | x_{-i})
):
    if len(x_0) != len(conditional):
        raise TypeError("There should be a conditional distribution for each x_0")

    parameter_dimensions = [len(x) for x in x_0]

    result: List[np.array] = []
    for i, x in enumerate(x_0):
        m = parameter_dimensions[i]
        y = np.zeros(m * n)
        y[:m] = x
        result.append(y)

    for i in range(0, n - 1):
        for j in range(len(x_0)):
            m = parameter_dimensions[j]
            for k in range(m):
                result[j][(i + 1) * m + k] = conditional[j](
                    # We always want to take the most recent len(x_0) values
                    k,
                    *(
                        x[
                            parameter_dimensions[h]
                            * i : parameter_dimensions[h]
                            * (i + 1)
                        ]
                        for h, x in enumerate(result)
                    )
                )

    reshaped_results = []
    for i, parameter in enumerate(result):
        reshaped_results.append(parameter.reshape(-1, parameter_dimensions[i]))

    return reshaped_results
