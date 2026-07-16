# -*- coding: utf-8 -*-
"""
Shared fixtures/helpers for the tsib test suite.
"""

import os

import numpy as np
import pandas as pd
import pytest

import tsib
import tsib.data


BUILDING_SET = pd.read_csv(
    os.path.join(tsib.data.PATH, "episcope", "tabula_DE_wPersons.csv"),
    header=0,
    index_col=0,
)


def example_building(ix=24, **kwargs):
    """The reference building used across the heat load tests: a
    TABULA archetype in TRY zone 4 with default control settings."""
    ID = BUILDING_SET.index[ix]
    try_data, loc = tsib.readTRY(try_num=4)
    params = {
        "ID": ID,
        "weatherData": try_data,
        "weatherID": "TRY_4",
        "refurbishment": False,
        "nightReduction": False,
        "occControl": False,
        "capControl": True,
        "n_persons": 2,
        "comfortT_lb": 20.0,
        "comfortT_ub": 26.0,
        "roofOrientation": 0.0,
        "n_apartments": 1,
        "longitude": loc["longitude"],
        "latitude": loc["latitude"],
    }
    params.update(kwargs)
    return tsib.Building(configurator=tsib.BuildingConfiguration(params)), ID


def zone_cfg_with_occupancy(ix=24, n_steps=None, **kwargs):
    """Resolved building cfg including simulated occupancy profiles,
    optionally truncated to the first n_steps hours for fast tests."""
    bdg, ID = example_building(ix=ix, **kwargs)
    bdg._get_occupancy_profile(bdg.cfg)
    cfg = dict(bdg.cfg)
    if n_steps is not None:
        cfg["weather"] = cfg["weather"].iloc[:n_steps]
        for key in ["Q_ig", "occ_nothome", "occ_sleeping", "elecLoad"]:
            value = cfg[key]
            if isinstance(value, np.ndarray):
                cfg[key] = value[:n_steps]
            else:
                cfg[key] = value.iloc[:n_steps]
    return cfg, ID
