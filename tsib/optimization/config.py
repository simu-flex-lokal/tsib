# -*- coding: utf-8 -*-
"""
Parameterization of a 5R1C thermal building zone (DIN EN ISO 13790,
formulation following Schuetz et al. 2017).

This module is pure data preparation and carries no optimization
framework: it turns a resolved tsib building configuration into the
derived 5R1C parameters, the solve-free design heat load, and the
plane-of-array irradiance per surface direction. The MILP formulation
that consumes it lives in `zone5r1c.py`.
"""

import numpy as np
import pandas as pd
import pvlib


class ThermalZoneConfig(object):
    """
    Parameterization of a 5R1C thermal zone from a resolved tsib
    building configuration dict. Provides the derived DIN EN ISO 13790
    parameters plus the solve-free design heat load calculations.

    Parameters
    ----------
    cfg: dict, required
        Resolved building configuration, e.g. from
        BuildingConfiguration.getBdgCfg().
    """

    CONST = {
        # specific heat transfer coefficient between internal air and surface [kW/m^2/K]
        # (DIN EN ISO 13790, section 7.2.2.2, page 35)
        "h_is": 3.45 / 1000,
        # non-dimensional relation between the area of all indoor surfaces
        # and the effective floor area A["f"]
        # (DIN EN ISO 13790, section 7.2.2.2, page 36)
        "lambda_at": 4.5,
        # specific heat transfer coefficient thermal capacity [kW/m^2/K]
        # (DIN EN ISO 13790, section 12.2.2, page 79)
        "h_ms": 9.1 / 1000,
        # ISO 6946 Table 1, Heat transfer resistances for opaque components
        "R_se": 0.04 * 1000,  # external heat transfer coefficient m²K/W
        # external specific radiative heat transfer [kW/m^2/K] (ISO 13790, Schuetz et al. 2017, 2.3.4)
        "h_r": 0.9 * 5.0 / 1000.0,
        # ASHRAE 140 : 2011, Table 5.3, page 18 (absorption opaque comps)
        "alpha": 0.6,
        # average difference external air temperature and sky temperature
        "delta_T_sky": 11.0,  # K
        # density air
        "rho_air": 1.2,  # kg/m^3
        # heat capacity air
        "C_air": 1.006,  # kJ/kg/K
    }

    # thermal capacity class definition of the building
    # (DIN EN ISO 13790, section 12.3.1.2, page 81, table 12)
    B_CLASS_F_LB = {
        "very light": 0.0,
        "light": 95.0,
        "medium": 137.5,
        "heavy": 212.5,
        "very heavy": 313.5,
    }
    B_CLASS_F_UB = {
        "very light": 95.0,
        "light": 137.5,
        "medium": 212.5,
        "heavy": 313.5,
        "very heavy": 313.5 * 2,
    }
    B_CLASS_F_A = {
        "very light": 2.5,
        "light": 2.5,
        "medium": 2.5,
        "heavy": 3.0,
        "very heavy": 3.5,
    }

    def __init__(self, cfg):
        self.cfg = cfg

        # specific heat [kJ/m^2/K] (Note: The standard value is overwritten!)
        self.cfg["c_m"] = (
            self.B_CLASS_F_LB[cfg["thermalClass"]]
            + self.B_CLASS_F_UB[cfg["thermalClass"]]
        ) / 2.0

        # storage for orignal U-values for scaling of the heat demand
        self._orig_U_Values = {}

    def __getitem__(self, key):
        return self.cfg[key]

    def __contains__(self, key):
        return key in self.cfg

    @property
    def times(self):
        return self.cfg["weather"].index

    @property
    def weather(self):
        return self.cfg["weather"]

    # --- derived 5R1C parameters ------------------------------------

    @property
    def A_f(self):
        """Heated floor area [m^2]."""
        return self.cfg["A_ref"]

    @property
    def A_m(self):
        """Effective heat transfer area of the thermal mass [m^2]."""
        return self.A_f * self.B_CLASS_F_A[self.cfg["thermalClass"]]

    @property
    def H_ms(self):
        """Effective transfer coefficient mass <-> surface [kW/K]."""
        return self.A_m * self.CONST["h_ms"]

    @property
    def C_m(self):
        """Internal heat capacity [kWh/K]."""
        return self.cfg["A_ref"] * self.cfg["c_m"] / 3600.0

    @property
    def H_door(self):
        """Heat transfer through the door [kW/K]."""
        return (self.cfg["A_Door_1"] * self.cfg["U_Door_1"]) / 1000

    @property
    def A_tot(self):
        """Internal surface area [m^2]."""
        return self.cfg["A_ref"] * self.CONST["lambda_at"]

    @property
    def H_is(self):
        """Heat transfer between surface and air node [kW/K]
        (Schuetz et al. 2017 - eq. 11)."""
        return self.A_tot * self.CONST["h_is"]

    # --- solve-free calculations (previous Building5R1C methods) ----

    def scaleHeatLoad(self, scale=1):
        """
        Scales the original heat demand of the model to a relative value
        by scaling all U-values and the infiltration rate.
        """
        # check if original values have already been saved
        if not bool(self._orig_U_Values):
            for key in self.cfg:
                if str(key).startswith("U_"):
                    self._orig_U_Values[key] = self.cfg[key]
            self._orig_U_Values["n_air_infiltration"] = self.cfg["n_air_infiltration"]
            self._orig_U_Values["n_air_use"] = self.cfg["n_air_use"]

        # replace the values in the model
        for key in self._orig_U_Values:
            self.cfg[key] = self._orig_U_Values[key] * scale

        return

    def calcDesignHeatLoad(self):
        """
        Calculates the design heat load which is needed to satisfy the
        nominal outdoor temperature.

        Returns
        -------
        designHeatLoad [kW]
        """
        b = self.cfg
        designHeatLoad = (
            b["A_Roof_1"] * b["U_Roof_1"] * b["b_Transmission_Roof_1"]
            + b["A_Roof_2"] * b["U_Roof_2"] * b["b_Transmission_Roof_2"]
            + b["A_Wall_1"] * b["U_Wall_1"] * b["b_Transmission_Wall_1"]
            + b["A_Wall_2"] * b["U_Wall_2"] * b["b_Transmission_Wall_2"]
            + b["A_Wall_3"] * b["U_Wall_3"] * b["b_Transmission_Wall_3"]
            + b["A_Window"] * b["U_Window"]
            + b["A_Door_1"] * b["U_Door_1"]
            + (
                b["A_ref"]
                * b["h_room"]
                * 1.2
                * 1006
                * (b["n_air_infiltration"] + b["n_air_use"])
                / 3600
            )
        ) * (22.917 - self.cfg["design_T_min"]) + (
            (
                b["A_Floor_1"] * b["U_Floor_1"] * b["b_Transmission_Floor_1"]
                + b["A_Floor_2"] * b["U_Floor_2"] * b["b_Transmission_Floor_2"]
            )
            * (22.917 - self.cfg["design_T_min"])
            * 1.45
        )
        return designHeatLoad / 1000


def calc_surface_irradiance(cfg):
    """
    Calculates the plane-of-array irradiance [kW/m^2] on all considered
    walls and roofs.

    Returns
    -------
    pandas.DataFrame with one column per surface direction.
    """
    # shorten code
    rt = cfg["roofTilt"]
    ro = cfg["roofOrientation"]

    # set relevant geometrical data
    surf_az = {
        "North": 0.0 + ro,
        "East": 90.0 + ro,
        "South": 180.0 + ro,
        "West": 270.0 + ro,
        "Horizontal": 180.0,
        "Roof 1": 90.0 + ro,
        "Roof 2": 270.0 + ro,
    }
    surf_tilt = {
        "North": 90.0,
        "East": 90.0,
        "South": 90.0,
        "West": 90.0,
        "Horizontal": 0.0,
        "Roof 1": rt,
        "Roof 2": rt,
    }

    SOL_POS = pvlib.solarposition.get_solarposition(
        cfg["weather"].index, cfg["latitude"], cfg["longitude"]
    )
    AM = pvlib.atmosphere.get_relative_airmass(SOL_POS["apparent_zenith"])
    DNI_ET = pvlib.irradiance.get_extra_radiation(cfg["weather"].index.dayofyear)

    irrad_surf = pd.DataFrame(index=cfg["weather"].index)
    for key in surf_az:
        # calculate total irradiance depending on surface tilt and azimuth
        total = pvlib.irradiance.get_total_irradiance(
            surf_tilt[key],
            surf_az[key],
            SOL_POS["apparent_zenith"],
            SOL_POS["azimuth"],
            dni=cfg["weather"]["DNI"],
            ghi=cfg["weather"]["GHI"],
            dhi=cfg["weather"]["DHI"],
            dni_extra=DNI_ET,
            airmass=AM,
            model="perez",
            surface_type="urban",
        )
        # get plane of array (POA) irradiance and replace nan
        irrad_surf[key] = total["poa_global"].fillna(0)

    # W to kW
    return irrad_surf / 1000
