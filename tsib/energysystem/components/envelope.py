# -*- coding: utf-8 -*-
"""
Building envelope refurbishment catalog: loads the Excel cost data
(sheets Walls/Roof/Floor/Windows/Ventilation) and derives per option
the resulting heat transfer coefficients and investment cost. Pure data
preparation - the constraints are built by ThermalZone5R1C via
DiscreteOptionInvestment.switched_flow().
"""

import numpy as np
import pandas as pd

ENVELOPE_ELEMENTS = ["Walls", "Roof", "Floor", "Windows", "Ventilation"]

# specific heat capacity and density of air for the ventilation heat flow
RHO_AIR = 1.2  # kg/m^3
C_AIR = 1.006  # kJ/kg/K


def _combined_u_value(u_values):
    """Serial combination of layer U-values; a 0.0 layer short-circuits
    to a perfectly insulated component."""
    if 0.0 in u_values:
        return 0.0
    return 1.0 / sum(1.0 / u for u in u_values)


def _layer_u_values(raw, option, existing_u):
    """U-values of the insulation layers of `option` plus the existing
    construction layer."""
    u_insul = raw.loc[option, "U_Value"]
    if isinstance(u_insul, pd.Series):
        return np.append(u_insul.values, existing_u)
    return [u_insul, existing_u]


def load_envelope_options(cfg, lifetime_default=40.0):
    """
    Reads the refurbishment catalog for all envelope elements from
    cfg["costdatapath"] and derives the option parameters.

    Parameters
    ----------
    cfg: dict, required
        Resolved building configuration (BuildingConfiguration.getBdgCfg).
    lifetime_default: float, optional (default: 40.)
        Economic lifetime of envelope measures without an own lifetime
        column.

    Returns
    -------
    dict element -> OrderedDict option -> dict with keys
        "H" [kW/K], "capex" [EUR], "opex_fix_share", "lifetime" [a]
        and for windows additionally "U" [kW/m^2/K] and "g_gl".
    """
    raw = {}
    for element in ENVELOPE_ELEMENTS:
        raw[element] = pd.read_excel(
            cfg["costdatapath"], sheet_name=element, skiprows=[1], index_col=0
        ).dropna(how="all")
        # derive u values of each insulation layer
        if element in ["Walls", "Roof", "Floor"]:
            raw[element]["U_Value"] = (
                raw[element]["Lambda"] / raw[element]["Thickness"]
            )

    options = {element: {} for element in ENVELOPE_ELEMENTS}

    # opaque elements: serial layer combination with the existing
    # construction, summed over all part surfaces
    opaque_surfaces = {
        "Walls": ["Wall_1", "Wall_2", "Wall_3"],
        "Roof": ["Roof_1", "Roof_2"],
        "Floor": ["Floor_1", "Floor_2"],
    }
    for element, surfaces in opaque_surfaces.items():
        # "Investment only energy" separates the energy-related share
        # from general renovation cost (not available for floors)
        invest_col = "Investment"
        if element != "Floor" and cfg["onlyEnergyInvest"]:
            invest_col = "Investment only energy"
        for option in raw[element].index.unique():
            H = 0.0
            capex = 0.0
            for surf in surfaces:
                u_layers = _layer_u_values(raw[element], option, cfg["U_" + surf])
                H += (
                    _combined_u_value(u_layers)
                    * cfg["A_" + surf]
                    * cfg["b_Transmission_" + surf]
                ) / 1000
                capex += cfg["A_" + surf] * float(
                    np.asarray(raw[element].loc[option, invest_col]).sum()
                )
            options[element][option] = {
                "H": H,
                "capex": capex,
                "opex_fix_share": 0.0,
                "lifetime": lifetime_default,
            }

    # windows: the "Nothing" option is the existing window
    win = raw["Windows"]
    win.loc["Nothing", "g_gl"] = cfg["g_gl_n_Window"]
    win.loc["Nothing", "U_Value"] = cfg["U_Window"]
    win.loc["Nothing", "Investment"] = 0
    for option in win.index.unique():
        u_value = float(win.loc[option, "U_Value"])
        options["Windows"][option] = {
            "H": cfg["A_Window"] * u_value / 1000,
            "U": u_value / 1000,
            "g_gl": float(win.loc[option, "g_gl"]),
            "capex": cfg["A_Window"] * float(win.loc[option, "Investment"]),
            "opex_fix_share": 0.0,
            "lifetime": lifetime_default,
        }

    # ventilation: heat flow corrected by the recovery rate
    C_air_room = cfg["A_ref"] * cfg["h_room"] * RHO_AIR * C_AIR
    vent = raw["Ventilation"]
    for option in vent.index.unique():
        options["Ventilation"][option] = {
            "H": (
                C_air_room
                * (
                    cfg["n_air_use"]
                    * (1 - float(vent.loc[option, "Recovery rate"]))
                    + cfg["n_air_infiltration"]
                )
                / 3600
            ),  # [kW/K]
            "capex": cfg["A_ref"] * float(vent.loc[option, "Investment"]),
            "opex_fix_share": float(vent.loc[option, "OPEX-Fix"]),
            "lifetime": float(vent.loc[option, "Lifetime"]),
        }

    return options


def load_control_options(cfg):
    """
    Reads the comfort controller catalog (sheet "Control") from
    cfg["costdatapath"].

    Returns
    -------
    dict feature -> dict with keys "capex", "opex_fix_share", "lifetime"
    """
    raw = pd.read_excel(
        cfg["costdatapath"], sheet_name="Control", skiprows=[1], index_col=0
    ).dropna(how="all")

    options = {}
    for feature in raw.index:
        options[feature] = {
            "capex": cfg["A_ref"] * float(raw.loc[feature, "Investment spec"])
            + float(raw.loc[feature, "Investment fix"]),
            "opex_fix_share": float(raw.loc[feature, "OPEX-Fix"]),
            "lifetime": float(raw.loc[feature, "Lifetime"]),
        }
    return options
