import numpy as np


# All functions are defined over unit-cubes
def one_dim_f(x, noise=False):
    if np.ndim(x) == 1:
        x = x[:, np.newaxis]
    if x.shape[1] != 1:
        raise Exception("Input must be 1-dim")

    y = 2 * x * np.sin(x ** 3 * 8 * np.pi)
    if noise:
        return y.ravel() + np.random.normal(0, 1e-2, len(x))
    else:
        return y.ravel()


def two_dim_f(x, noise=False):
    if np.ndim(x) == 1:
        x = x[np.newaxis, :]
    if x.shape[1] != 2:
        raise Exception("Input must be 2-dim")

    x = 6 * (x - 0.5) + 1
    y = x[:, 0] * np.exp(-np.square(x[:, 0])-np.square(x[:, 1]))

    if noise:
        return y + np.random.normal(0, 1e-3, len(x))
    else:
        return y
