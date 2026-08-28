# -*- coding: utf-8 -*-
"""
Heat transfer coefficients of the building envelope as it is built,
derived from the resolved building configuration.

The values feed the 5R1C node balances (H per element) and the window
solar gains (U and g_gl of the glazing).
"""

ENVELOPE_ELEMENTS = ["Walls", "Roof", "Floor", "Windows", "Ventilation"]

# specific heat capacity and density of air for the ventilation heat flow
RHO_AIR = 1.2  # kg/m^3
C_AIR = 1.006  # kJ/kg/K

#: part surfaces summed up per opaque envelope element
OPAQUE_SURFACES = {
    "Walls": ["Wall_1", "Wall_2", "Wall_3"],
    "Roof": ["Roof_1", "Roof_2"],
    "Floor": ["Floor_1", "Floor_2"],
}


def existing_envelope(cfg):
    """
    Heat transfer coefficients of the existing construction.

    Parameters
    ----------
    cfg: dict, required
        Resolved building configuration (BuildingConfiguration.getBdgCfg).

    Returns
    -------
    dict element -> dict with key "H" [kW/K], for windows additionally
        "U" [kW/m^2/K] and "g_gl".
    """
    envelope = {}

    # opaque elements: transmission over all part surfaces
    for element, surfaces in OPAQUE_SURFACES.items():
        envelope[element] = {
            "H": sum(
                (
                    cfg["U_" + surf]
                    * cfg["A_" + surf]
                    * cfg["b_Transmission_" + surf]
                )
                / 1000
                for surf in surfaces
            )
        }

    envelope["Windows"] = {
        "H": cfg["A_Window"] * cfg["U_Window"] / 1000,
        "U": cfg["U_Window"] / 1000,
        "g_gl": cfg["g_gl_n_Window"],
    }

    # ventilation: air exchange of the room volume, no heat recovery
    C_air_room = cfg["A_ref"] * cfg["h_room"] * RHO_AIR * C_AIR
    envelope["Ventilation"] = {
        "H": C_air_room * (cfg["n_air_use"] + cfg["n_air_infiltration"]) / 3600
    }

    return envelope
