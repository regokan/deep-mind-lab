"""Helper functions for policies."""

import numpy as np


def discretize(sample, grid):
    """Discretize a sample as per given grid.

    Parameters
    ----------
    sample : array_like
        A single sample from the (original) continuous space.
    grid : list of array_like
        A list of arrays containing split points for each dimension.

    Returns
    -------
    discretized_sample : array_like
        A sequence of integers with the same number of dimensions as sample.
    """
    discretized_sample = [np.digitize(sample[i], grid[i]) for i in range(len(sample))]
    return np.array(discretized_sample)
