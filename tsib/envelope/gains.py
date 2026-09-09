# -*- coding: utf-8 -*-
"""
Solar and internal gains on the nodes of a 5R1C thermal zone, and the
complete parameter set such a zone needs.

All of this is exogenous: the gains enter the node balances as a right hand
side and depend on no state and on no decision, so they can be computed for
a building before any thermal model exists. `zone_parameters` collects them
together with the derived heat transfer coefficients into the ten scalars
and five time series which are the whole contract of the zone - the same
contract an external energy system model is handed.

Physics in the formulation of Schuetz et al. 2017. The deviations the solved
zone still carries are documented in esmkit, which owns that register:
https://github.com/simu-flex-lokal/esmkit/blob/master/docs/model-deviations.md
"""

import numpy as np
import pandas as pd

from .comfort import comfort_bounds
from .config import ThermalZoneConfig, calc_surface_irradiance
from .elements import ENVELOPE_ELEMENTS, existing_envelope


#: envelope elements the solar gains are distributed over
SOLAR_ELEMENTS = ["Windows", "Walls", "Roof"]

#: design heat load adjustment factor per envelope element
DESIGN_ADJUST = {
    "Floor": 1.45,
    "Walls": 1.0,
    "Roof": 1.0,
    "Windows": 1.0,
    "Ventilation": 1.0,
}

#: nominal design indoor temperature [degC]
DESIGN_T_INDOOR = 22.917

#: the five time series of the zone contract
ZONE_SERIES = ["T_e", "gain_mass", "gain_surface", "comfort_lb", "comfort_ub"]


class ZoneGains(object):
    """
    Exogenous profiles and node gains of one building's thermal zone.

    Parameters
    ----------
    config: ThermalZoneConfig or dict, required
        Resolved building configuration.
    """

    def __init__(self, config):
        if not isinstance(config, ThermalZoneConfig):
            config = ThermalZoneConfig(config)
        self.config = config
        self.cfg = config.cfg

        if self.cfg.get("ventControl", False):
            raise NotImplementedError(
                "The unvalidated ventilation control path was not ported"
            )

        self.envelope = existing_envelope(self.cfg)

        self.n_steps = len(self.config.times)
        self._profiles = {}
        self._sol_profiles = {}
        self._prepare()

    # --- exogenous profiles ---------------------------------------------

    def _optional_profile(self, key):
        n = self.n_steps
        if key in self.cfg and self.cfg[key] is not None:
            values = self.cfg[key]
            return values.values if hasattr(values, "values") else np.asarray(values)
        return np.zeros(n)

    def _prepare(self):
        """Precomputes the exogenous profiles and the option-dependent solar
        gains. Ported unchanged from the pre-migration build_parameters()."""
        cfg = self.cfg
        const = self.config.CONST

        self._profiles["T_e"] = cfg["weather"]["T"].values
        self._profiles["Q_ig"] = self._optional_profile("Q_ig")
        self._profiles["occ_nothome"] = self._optional_profile("occ_nothome")
        self._profiles["occ_sleeping"] = self._optional_profile("occ_sleeping")

        irrad_surf = calc_surface_irradiance(cfg)
        F_r = {"North": 0.5, "East": 0.5, "South": 0.5, "West": 0.5, "Horizontal": 1.0}

        window_area_s = {
            di: cfg["A_Window_" + di] * cfg["F_sh_vert"]
            for di in ["North", "East", "South", "West"]
        }
        window_area_s["Horizontal"] = cfg["A_Window_Horizontal"] * cfg["F_sh_hor"]
        window_area_t = sum(cfg["A_Window_" + key] * F_r[key] for key in F_r)
        irrad_on_windows = irrad_surf.mul(pd.Series(window_area_s)).sum(axis=1).values

        # solar gains per element, incl. the thermal radiation correction
        # (Schuetz et al. 2017 - eq. 13)
        win = self.envelope["Windows"]
        thermal_rad = (
            window_area_t
            * const["h_r"]
            * win["U"]
            * const["R_se"]
            * const["delta_T_sky"]
        )
        self._sol_profiles["Windows"] = (
            irrad_on_windows * (1.0 - cfg["F_f"]) * cfg["F_w"] * win["g_gl"]
            - thermal_rad
        )

        mean_ver_irr = (
            irrad_surf.loc[:, ["North", "East", "South", "West"]].mean(axis=1).values
        )
        H = self.envelope["Walls"]["H"]
        thermal_rad = H * const["h_r"] * const["R_se"] * const["delta_T_sky"]
        self._sol_profiles["Walls"] = (
            mean_ver_irr * H * cfg["F_sh_vert"] * const["R_se"] * const["alpha"]
            - thermal_rad
        )

        mean_roof_irr = irrad_surf.loc[:, ["Roof 1", "Roof 2"]].mean(axis=1).values
        H = self.envelope["Roof"]["H"]
        thermal_rad = H * const["h_r"] * const["R_se"] * const["delta_T_sky"]
        self._sol_profiles["Roof"] = (
            mean_roof_irr * H * cfg["F_sh_hor"] * const["R_se"] * const["alpha"]
            - thermal_rad
        )

    # --- gain expressions (all scalars) ---------------------------------

    def H_element(self, element):
        """Heat transfer coefficient [kW/K] of the installed element."""
        return self.envelope[element]["H"]

    def solar_sum(self, t):
        """Total solar gains through the installed envelope [kW]."""
        return sum(
            self._sol_profiles[element][t] for element in SOLAR_ELEMENTS
        )

    def gain_mass_node(self, t):
        """Heat gain of the thermal mass node (Schuetz et al. 2017 - eq. 15)."""
        cfg = self.config
        return (
            0.5 * cfg.A_m / cfg.A_tot * self._profiles["Q_ig"][t]
            + cfg.A_f / cfg.A_tot * self.solar_sum(t)
        )

    def gain_surface_node(self, t):
        """Heat gain of the thermal mass surface node
        (Schuetz et al. 2017 - eq. 16)."""
        cfg = self.config
        const = cfg.CONST
        win_data = self.envelope["Windows"]

        u_win = win_data["U"]
        cross = sum(
            win_data["H"] * self._sol_profiles[element][t]
            for element in SOLAR_ELEMENTS
        )
        return (
            (1 - u_win / const["h_ms"] / cfg.A_tot) * (0.5 * self._profiles["Q_ig"][t])
            - cross / const["h_ms"] / cfg.A_tot
            + self.solar_sum(t)
            - self.gain_mass_node(t)
        )

    def design_heat_load_value(self):
        """Design heat load [kW] of the installed envelope."""
        H_total = self.config.H_door
        for element in ENVELOPE_ELEMENTS:
            H_total += self.envelope[element]["H"] * DESIGN_ADJUST[element]
        return H_total * (DESIGN_T_INDOOR - self.cfg["design_T_min"])


def zone_parameters(cfg, max_load=None):
    """
    The complete parameterization of the building's 5R1C thermal zone.

    Everything building-shaped happens here and nowhere downstream: the
    envelope heat transfer coefficients, the irradiance on tilted surfaces,
    the distribution of internal and solar gains over the thermal nodes, and
    the comfort band implied by the installed control equipment. What comes
    out is ten numbers and five arrays - serializable, framework free, and
    free of any notion of a building.

    Parameters
    ----------
    cfg: dict or ThermalZoneConfig, required
        A resolved building configuration, i.e. `Building.cfg`.
    max_load: float, optional (default: the design heat load)
        Maximal load of the heating/cooling system [kW].

    Returns
    -------
    dict with the ten scalars and the five arrays of the zone contract.
    """
    gains = ZoneGains(cfg)
    n = gains.n_steps
    lower, upper = comfort_bounds(gains.cfg, n)

    if max_load is None:
        max_load = gains.config.calcDesignHeatLoad()

    return {
        "H_ms": float(gains.config.H_ms),
        "H_is": float(gains.config.H_is),
        "H_door": float(gains.config.H_door),
        "C_m": float(gains.config.C_m),
        "H": {element: float(gains.H_element(element))
              for element in ENVELOPE_ELEMENTS},
        "max_load": float(max_load),
        "design_capacity": float(gains.design_heat_load_value()),
        "T_e": np.asarray(gains._profiles["T_e"], dtype=float),
        "gain_mass": np.array([gains.gain_mass_node(t) for t in range(n)]),
        "gain_surface": np.array([gains.gain_surface_node(t) for t in range(n)]),
        "comfort_lb": np.asarray(lower, dtype=float),
        "comfort_ub": np.asarray(upper, dtype=float),
    }
