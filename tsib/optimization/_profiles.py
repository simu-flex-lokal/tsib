# -*- coding: utf-8 -*-
"""
Helper to accept scalar or time series parameters interchangeably.
"""

import numpy as np
import pandas as pd


def profile_values(value, n_steps, name="parameter"):
    """
    Normalizes float | list | numpy.ndarray | pandas.Series to a numpy
    array of length n_steps (scalars are broadcast).
    """
    if value is None:
        return None
    if np.isscalar(value):
        return np.full(n_steps, float(value))
    if isinstance(value, pd.Series):
        values = value.values
    else:
        values = np.asarray(value)
    if len(values) != n_steps:
        raise ValueError(
            "{} has {} entries but the time index has {} steps".format(
                name, len(values), n_steps
            )
        )
    return values.astype(float)
