# -*- coding: utf-8 -*-
"""
Phase 0 equivalence tests: the solve-free calculations extracted from
Building5R1C into ThermalZoneConfig produce identical results.
"""

import os

import numpy as np
import pandas as pd

import tsib
import tsib.data
from tsib.energysystem import ThermalZoneConfig


def get_example_cfg():
    buildingSet = pd.read_csv(
        os.path.join(tsib.data.PATH, "episcope", "tabula_DE_wPersons.csv"),
        header=0,
        index_col=0,
    )
    ID = buildingSet.index[24]
    try_data, loc = tsib.readTRY(try_num=4)
    bdgcfg = tsib.BuildingConfiguration(
        {
            "ID": ID,
            "weatherData": try_data,
            "weatherID": "TRY_4",
            "longitude": loc["longitude"],
            "latitude": loc["latitude"],
        }
    )
    return bdgcfg.getBdgCfg(includeSupply=True)


def test_design_heat_load():
    cfg = get_example_cfg()
    zone = ThermalZoneConfig(cfg)

    design_load = zone.calcDesignHeatLoad()
    assert design_load > 0

    # manual reference of the DIN design load formula
    b = cfg
    reference = (
        (
            b["A_Roof_1"] * b["U_Roof_1"] * b["b_Transmission_Roof_1"]
            + b["A_Roof_2"] * b["U_Roof_2"] * b["b_Transmission_Roof_2"]
            + b["A_Wall_1"] * b["U_Wall_1"] * b["b_Transmission_Wall_1"]
            + b["A_Wall_2"] * b["U_Wall_2"] * b["b_Transmission_Wall_2"]
            + b["A_Wall_3"] * b["U_Wall_3"] * b["b_Transmission_Wall_3"]
            + b["A_Window"] * b["U_Window"]
            + b["A_Door_1"] * b["U_Door_1"]
            + b["A_ref"]
            * b["h_room"]
            * 1.2
            * 1006
            * (b["n_air_infiltration"] + b["n_air_use"])
            / 3600
        )
        * (22.917 - b["design_T_min"])
        + (
            b["A_Floor_1"] * b["U_Floor_1"] * b["b_Transmission_Floor_1"]
            + b["A_Floor_2"] * b["U_Floor_2"] * b["b_Transmission_Floor_2"]
        )
        * (22.917 - b["design_T_min"])
        * 1.45
    ) / 1000
    np.testing.assert_almost_equal(design_load, reference, decimal=10)


def test_scale_heat_load():
    cfg = get_example_cfg()
    zone = ThermalZoneConfig(cfg)

    orig = zone.calcDesignHeatLoad()
    zone.scaleHeatLoad(scale=0.5)
    reduced = zone.calcDesignHeatLoad()
    np.testing.assert_almost_equal(reduced / 0.5, orig, decimal=2)

    # scaling back restores the original values
    zone.scaleHeatLoad(scale=1.0)
    np.testing.assert_almost_equal(zone.calcDesignHeatLoad(), orig, decimal=10)
