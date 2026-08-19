# -*- coding: utf-8 -*-
"""
5R1C thermal building zone as an oemof-solph component.

This is the one piece of tsib's building physics that has no equivalent in
solph's component library, and therefore the only custom component of the
whole framework: every other tsib component (grid, PV, storage, heat pump,
demand, metering) maps onto a stock solph object.

Why it must be a component and not a precomputed load profile: the zone's
temperature states (T_air, T_s, T_m) are free decision variables inside a
comfort band, so the building's thermal mass is a dispatchable flexibility
resource in the same optimization as storage, PV and prices. Reducing the
zone to a fixed heat demand time series and optimizing dispatch afterwards
would throw that flexibility away.

The heat (and optionally cooling) supply is not a variable of this
component: it is `m.flow[heat_bus, zone, t]`, the edge solph creates for
the zone's input flow. That edge is the entire coupling mechanism.

Four deviations from the source papers are carried over from the
pre-migration model on purpose - they are load-bearing for the validated
~197 kWh/m2/a result. See `docs/model-deviations.md` before "correcting"
any node balance or the comfort bound.
"""

import warnings

import numpy as np
import pandas as pd
import pyomo.environ as po
from oemof.solph import Flow

from .base import TsibBlock, TsibComponent
from .config import ThermalZoneConfig, calc_surface_irradiance
from .envelope import ENVELOPE_ELEMENTS, existing_envelope


class ThermalZone5R1C(TsibComponent):
    """
    5R1C thermal zone (DIN EN ISO 13790 / Schuetz et al. 2017).

    Parameters
    ----------
    label: str, required
        Node label, unique within the energy system.
    config: ThermalZoneConfig or dict, required
        Resolved building configuration, e.g. BuildingConfiguration.getBdgCfg().
    heat_bus: solph.Bus, required
        Bus the zone draws its heating from. Unlike the pre-migration model
        the zone never prices its own supply: put a Source with
        `variable_costs` on the bus instead.
    cool_bus: solph.Bus, optional (default: None)
        Bus the zone draws cooling from. If None, cooling is an internal
        free variable (still needed because the comfort ceiling is a hard
        bound - see deviation 4).
    max_load: float, optional (default: design heat load)
        Maximal load of the heating/cooling system [kW], enforced as a soft
        constraint so that an undersized system warns instead of turning
        the model infeasible.
    max_load_violation_penalty: float, optional (default: 100.)
        Penalty of the soft constraint [EUR/kW].
    initial_T_m: float, optional (default: None)
        Initial thermal mass temperature [degC]. Replaces the periodic wrap
        of the mass node when given.
    """

    def __init__(
        self,
        label,
        config,
        heat_bus,
        cool_bus=None,
        max_load=None,
        max_load_violation_penalty=100.0,
        initial_T_m=None,
    ):
        inputs = {heat_bus: Flow()}
        if cool_bus is not None:
            inputs[cool_bus] = Flow()
        super().__init__(label=label, inputs=inputs)

        if not isinstance(config, ThermalZoneConfig):
            config = ThermalZoneConfig(config)
        self.config = config
        self.cfg = config.cfg
        self.heat_bus = heat_bus
        self.cool_bus = cool_bus
        self.max_load = max_load
        self.max_load_violation_penalty = max_load_violation_penalty
        self.initial_T_m = initial_T_m

        if self.cfg.get("ventControl", False):
            raise NotImplementedError(
                "The unvalidated ventilation control path was not ported"
            )

        self.envelope = existing_envelope(self.cfg)

        # filled in by the block on first build
        self._profiles = {}
        self._sol_profiles = {}
        self._prepared = False

    # --- parameter preparation (framework independent) ------------------

    def _optional_profile(self, key, n):
        if key in self.cfg and self.cfg[key] is not None:
            values = self.cfg[key]
            return values.values if hasattr(values, "values") else np.asarray(values)
        return np.zeros(n)

    def prepare(self, n):
        """Precomputes the exogenous profiles and the option-dependent solar
        gains. Ported unchanged from the pre-migration build_parameters()."""
        if self._prepared:
            return
        if n != len(self.config.times):
            raise ValueError(
                "The model time index ({} steps) does not match the zone "
                "weather data ({} steps)".format(n, len(self.config.times))
            )
        cfg = self.cfg
        const = self.config.CONST

        self._profiles["T_e"] = cfg["weather"]["T"].values
        self._profiles["Q_ig"] = self._optional_profile("Q_ig", n)
        self._profiles["occ_nothome"] = self._optional_profile("occ_nothome", n)
        self._profiles["occ_sleeping"] = self._optional_profile("occ_sleeping", n)

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

        if self.max_load is None:
            self.max_load = self.config.calcDesignHeatLoad()

        self._prepared = True

    # --- gain expressions (all scalars) ---------------------------------

    def control(self, flag):
        """Installed comfort control feature as 1./0. coefficient."""
        return 1.0 if self.cfg[flag] else 0.0

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

    def constraint_group(self):
        return ThermalZone5R1CBlock


#: envelope elements the solar gains are distributed over
SOLAR_ELEMENTS = ["Windows", "Walls", "Roof"]

# design heat load adjustment factor per envelope element
DESIGN_ADJUST = {
    "Floor": 1.45,
    "Walls": 1.0,
    "Roof": 1.0,
    "Windows": 1.0,
    "Ventilation": 1.0,
}
# nominal design indoor temperature [degC]
DESIGN_T_INDOOR = 22.917


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


class ThermalZone5R1CBlock(TsibBlock):
    """
    Constraints of all ThermalZone5R1C nodes in an energy system.
    """

    def _create(self, group=None):
        if group is None:
            return
        m = self.parent_block()
        n_steps = len(m.TIMESTEPS)
        for zone in group:
            zone.prepare(n_steps)

        self.ZONES = po.Set(initialize=list(group), ordered=True)

        # --- variables --------------------------------------------------
        # node temperatures [degC]; free reals, the comfort band bounds them
        self.T_air = po.Var(self.ZONES, m.TIMESTEPS, within=po.Reals)
        self.T_s = po.Var(self.ZONES, m.TIMESTEPS, within=po.Reals)
        self.T_m = po.Var(self.ZONES, m.TIMESTEPS, within=po.Reals)
        # cooling when no cool bus is attached (the comfort ceiling is hard)
        self.Q_cool_internal = po.Var(
            self.ZONES, m.TIMESTEPS, within=po.NonNegativeReals
        )
        # soft violation of the maximal system load [kW]
        self.max_load_violation = po.Var(self.ZONES, within=po.NonNegativeReals)

        def Q_heat(zone, t):
            return m.flow[zone.heat_bus, zone, t]

        def Q_cool(zone, t):
            if zone.cool_bus is None:
                return self.Q_cool_internal[zone, t]
            return m.flow[zone.cool_bus, zone, t]

        self._Q_heat = Q_heat
        self._Q_cool = Q_cool

        # --- envelope heat flows ----------------------------------------
        # POTENTIAL BUG (kept for parity with the pre-migration model):
        # only the opaque elements may use (T_m - T_e) per Schuetz et al.
        # 2017 - eq. 20/9. Windows should be driven by (T_s - T_e) (eq. 21)
        # and Ventilation by (T_air - T_e) (eq. 22).
        # See docs/model-deviations.md, items 1 and 2.
        def envelope_flow(zone, element, t):
            T_e = zone._profiles["T_e"][t]
            return zone.H_element(element) * (self.T_m[zone, t] - T_e)

        # --- 1) thermal mass node, the difference equation --------------
        def mass_node_balance(zone, t, t_next):
            cfg = zone.config
            T_e = zone._profiles["T_e"][t]
            return (
                cfg.H_ms * (self.T_m[zone, t] - self.T_s[zone, t])
                + sum(
                    envelope_flow(zone, e, t) for e in ["Walls", "Roof", "Floor"]
                )
                + cfg.H_door * (self.T_m[zone, t] - T_e)
                == zone.gain_mass_node(t)
                - cfg.C_m
                * (self.T_m[zone, t_next] - self.T_m[zone, t])
                / m.timeincrement[t]
            )

        last = n_steps - 1

        def mass_rule(b, zone, t):
            if t < last:
                return mass_node_balance(zone, t, t + 1)
            if zone.initial_T_m is None:
                # periodic wrap: the last step feeds the first one
                return mass_node_balance(zone, last, 0)
            return po.Constraint.Skip

        self.mass_node_balance = po.Constraint(self.ZONES, m.TIMESTEPS, rule=mass_rule)

        self.mass_node_initial = po.Constraint(
            self.ZONES,
            rule=lambda b, zone: (
                po.Constraint.Skip
                if zone.initial_T_m is None
                else self.T_m[zone, 0] == zone.initial_T_m
            ),
        )

        # --- 2) surface node --------------------------------------------
        self.surface_node_balance = po.Constraint(
            self.ZONES,
            m.TIMESTEPS,
            rule=lambda b, zone, t: (
                zone.config.H_ms * (self.T_s[zone, t] - self.T_m[zone, t])
                + zone.config.H_is * (self.T_s[zone, t] - self.T_air[zone, t])
                + envelope_flow(zone, "Windows", t)
                == zone.gain_surface_node(t)
            ),
        )

        # --- 3) air node, supplied by heating and cooling ---------------
        # POTENTIAL BUG (kept for parity): per Schuetz et al. 2017 - eq. 22
        # the air node receives Q_ia = 0.5 * Q_ig (eq. 14), not the surface
        # gain Q_st (eq. 19). See docs/model-deviations.md, item 3.
        self.air_node_balance = po.Constraint(
            self.ZONES,
            m.TIMESTEPS,
            rule=lambda b, zone, t: (
                envelope_flow(zone, "Ventilation", t)
                + zone.config.H_is * (self.T_air[zone, t] - self.T_s[zone, t])
                == zone.gain_surface_node(t) - Q_cool(zone, t) + Q_heat(zone, t)
            ),
        )

        # --- comfort band ------------------------------------------------
        # Schuetz et al. 2017 has no upper bound at all (eq. 26) and
        # free-floats instead; see comfort_bounds() and
        # backlog/open-problem-summer-overheating.md.
        band = {zone: comfort_bounds(zone.cfg, n_steps) for zone in group}

        def comfort_ub(b, zone, t):
            return self.T_air[zone, t] <= band[zone][1][t]

        def comfort_lb(b, zone, t):
            return self.T_air[zone, t] >= band[zone][0][t]

        self.comfort_ub = po.Constraint(self.ZONES, m.TIMESTEPS, rule=comfort_ub)
        self.comfort_lb = po.Constraint(self.ZONES, m.TIMESTEPS, rule=comfort_lb)

        # --- maximal system load as a soft constraint --------------------
        self.max_heating_load = po.Constraint(
            self.ZONES,
            m.TIMESTEPS,
            rule=lambda b, zone, t: Q_heat(zone, t) - self.max_load_violation[zone]
            <= zone.max_load,
        )
        self.max_cooling_load = po.Constraint(
            self.ZONES,
            m.TIMESTEPS,
            rule=lambda b, zone, t: Q_cool(zone, t) - self.max_load_violation[zone]
            <= zone.max_load,
        )

    def _objective_expression(self):
        """Only the penalty of the soft max-load constraint. Energy cost
        belongs to whatever supplies the heat bus - a component pricing
        energy it did not import would double-count."""
        return sum(
            self.max_load_violation[zone] * zone.max_load_violation_penalty
            for zone in self.ZONES
        )


def zone_results(model, zone):
    """
    Extracts the solved zone results in the shape `Building` and the TinyDB
    result cache expect: a "timeseries" frame of loads and temperatures plus
    a "static" dict of capacities and costs.

    Parameters
    ----------
    model: solph.Model, required
        The solved model.
    zone: ThermalZone5R1C, required

    Returns
    -------
    dict with "timeseries", "static", "max_load_violation"
    """
    block = model.ThermalZone5R1CBlock
    steps = list(model.TIMESTEPS)

    violation = po.value(block.max_load_violation[zone])
    if violation is not None and violation > 1e-9:
        warnings.warn(
            "Maximal heat load exceeded by " + str(round(violation, 3)) + " kW",
            UserWarning,
        )

    heat = np.array([po.value(model.flow[zone.heat_bus, zone, t]) for t in steps])
    if zone.cool_bus is None:
        cool = np.array([po.value(block.Q_cool_internal[zone, t]) for t in steps])
    else:
        cool = np.array([po.value(model.flow[zone.cool_bus, zone, t]) for t in steps])

    timeseries = pd.DataFrame(index=zone.config.times)
    timeseries["Heating Load"] = heat
    timeseries["Cooling Load"] = cool
    timeseries["T_air"] = [po.value(block.T_air[zone, t]) for t in steps]
    timeseries["T_s"] = [po.value(block.T_s[zone, t]) for t in steps]
    timeseries["T_m"] = [po.value(block.T_m[zone, t]) for t in steps]
    timeseries["T_e"] = zone._profiles["T_e"]

    static = {
        "Capacity": zone.design_heat_load_value(),
        "FixCost": 0,
        "CAPEX": 0,
        "OPEX fix": 0.0,
        "VarCost": 0.0,
        "OPEX var": 0.0,
        "OPEX": 0.0,
    }

    return {
        "timeseries": timeseries,
        "static": static,
        "max_load_violation": violation,
    }
