# -*- coding: utf-8 -*-
"""
Effective comfort band of a thermal zone.

The nominal band from the building configuration is only the starting
point: which bound actually holds at a given hour depends on the installed
comfort controls and on where the occupants are.
"""

import numpy as np


def comfort_bounds(cfg, n_steps=None):
    """
    Effective comfort band per time step [degC].

    `comfortT_lb` / `comfortT_ub` are only the nominal band. What the zone is
    actually held to depends on which comfort controls are installed and on
    where the occupants are: `capControl` releases the upper bound,
    `nightReduction` lets the floor drop while they sleep, and `occControl`
    widens both while nobody is home.

    LIMITATION (deliberate, Kotzur 2018 - eq. 3.2): the upper bound is hard,
    so gains above the band force cooling even in a building without any
    cooling device. See docs/model-deviations.md item 4.

    Parameters
    ----------
    cfg: dict or ThermalZoneConfig, required
        Resolved building configuration.
    n_steps: int, optional
        Length of the returned arrays. Defaults to the weather index.

    Returns
    -------
    (lower, upper) - two float arrays of length n_steps.
    """
    cfg = getattr(cfg, "cfg", cfg)
    if n_steps is None:
        n_steps = len(cfg["weather"].index)

    def profile(key):
        values = cfg.get(key)
        if values is None:
            return np.zeros(n_steps)
        values = values.values if hasattr(values, "values") else np.asarray(values)
        return np.asarray(values[:n_steps], dtype=float)

    def control(flag):
        return 1.0 if cfg[flag] else 0.0

    T_lb = cfg["comfortT_lb"]
    T_ub = cfg["comfortT_ub"]
    nothome = profile("occ_nothome")
    sleeping = profile("occ_sleeping")

    upper = (
        T_lb
        + (T_ub - T_lb) * control("capControl")
        - (T_ub - 30.0) * nothome * control("occControl")
    )
    lower = (
        T_lb
        - (T_lb - 18.0) * sleeping * control("nightReduction")
        - (T_lb - 14.0) * nothome * control("occControl")
    )
    return lower, upper
