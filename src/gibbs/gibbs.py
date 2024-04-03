from typing import Callable, get_args

import numpy as np


class GibbsData:
    def __init__(self, x_0: np.array, conditional) -> None:
        self.x_0 = x_0
        self.dim = len(x_0)
        self.conditional = conditional
        print("Creating Gibbs data with length ", self.dim)


def gibbs_sampler(
    n: int, parameters: dict[str, GibbsData]  # The length of the Markov chain
):
    result: dict[str, np.array] = {}
    for label, parameter in parameters.items():
        x = np.zeros(parameter.dim * n)
        x[: parameter.dim] = parameter.x_0
        result[label] = x

    for i in range(0, n - 1):
        for label, parameter in parameters.items():
            for k in range(parameter.dim):
                result[label][(i + 1) * parameter.dim + k] = parameter.conditional(
                    # We always want to take the most recent parameter.dim values
                    k,
                    **{
                        label: values[
                            parameters[label].dim * i : parameters[label].dim * (i + 1)
                        ]
                        for label, values in result.items()
                    },
                )

    reshaped_results = {}
    for label, values in result.items():
        reshaped_results[label] = values.reshape(-1, parameters[label].dim)

    return reshaped_results
