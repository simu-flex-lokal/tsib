# -*- coding: utf-8 -*-
"""
5R1C thermal building zone (DIN EN ISO 13790, formulation following
Schuetz et al. 2017) as an energy system component.

The zone's temperature states (T_air, T_s, T_m) and comfort-band
constraints are free decision variables coupled via a heat port to
whatever supplies the heat bus (grid price, storage, heat pump, ...).
This keeps the building's thermal mass available as a flexibility
resource for operational dispatch - pre-heating within the comfort band
in response to prices or PV surplus - which would be lost if the zone
were reduced to a fixed heat load time series.

Envelope refurbishment is an optional, composed investment decision
(DiscreteOptionInvestment per envelope element): without it, no binary
variables or Big-M constraints are created at all.
"""

import warnings

import numpy as np
import pandas as pd
import pvlib
import pyomo.environ as pyomo

from ..core.component import Component, Port
from ..core.investment import DiscreteOptionInvestment
from .envelope import ENVELOPE_ELEMENTS, load_envelope_options
from .control import ComfortControl


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


class ThermalZone5R1C(Component):
    """
    5R1C thermal zone component.

    Parameters
    ----------
    name: str, required
    config: ThermalZoneConfig or dict, required
        Resolved building configuration.
    heat_bus: Bus, optional (default: None)
        Heat bus the zone draws its heating from. If None, the zone
        owns its heat supply and prices it with `heat_cost`.
    cool_bus: Bus, optional (default: None)
        Analogous for the cooling demand; if None, priced with
        `cool_cost`.
    refurbishment: bool, optional (default: cfg["refurbishment"])
        Attach the envelope/control investment decisions to the zone.
    force_refurbishment: bool, optional (default: cfg["force_refurbishment"])
        Exclude the "Nothing" options, so a measure has to be chosen.
    max_load: float, optional (default: design heat load)
        Maximal load of the heating/cooling system [kW], enforced as a
        soft constraint.
    heat_cost, cool_cost: float, optional (default: 0.08, 0.02)
        Energy prices [EUR/kWh] used when no heat/cool bus is attached.
    initial_T_m: float, optional (default: None)
        Initial thermal mass temperature [degC]. Only applied by
        strategies without a periodic state wrap (myopic/rolling-
        horizon window chaining).
    """

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

    def __init__(
        self,
        name,
        config,
        heat_bus=None,
        cool_bus=None,
        refurbishment=None,
        force_refurbishment=None,
        max_load=None,
        heat_cost=0.08,
        cool_cost=0.02,
        lifetime_default=40.0,
        max_load_violation_penalty=100.0,
        initial_T_m=None,
    ):
        super().__init__(name)
        if not isinstance(config, ThermalZoneConfig):
            config = ThermalZoneConfig(config)
        self.config = config
        self.cfg = config.cfg
        self.heat_bus = heat_bus
        self.cool_bus = cool_bus
        self.heat_cost = heat_cost
        self.cool_cost = cool_cost
        self.max_load = max_load
        self.max_load_violation_penalty = max_load_violation_penalty
        self.initial_T_m = initial_T_m

        if refurbishment is None:
            refurbishment = bool(self.cfg.get("refurbishment", False))
        if force_refurbishment is None:
            force_refurbishment = bool(self.cfg.get("force_refurbishment", False))
        self.refurbishment = refurbishment

        if self.cfg.get("ventControl", False):
            raise NotImplementedError(
                "The unvalidated ventilation control path was not ported to "
                "the energy system framework"
            )

        # one discrete investment decision per envelope element; without
        # refurbishment the existing state ("Nothing") is fixed and no
        # decision variables will be created
        self.envelope_options = load_envelope_options(
            self.cfg, lifetime_default=lifetime_default
        )
        self.envelope_investments = {}
        for element in ENVELOPE_ELEMENTS:
            options = self.envelope_options[element]
            existing = next(iter(options))
            if refurbishment:
                if force_refurbishment:
                    options = {o: options[o] for o in options if o != existing}
                self.envelope_investments[element] = DiscreteOptionInvestment(
                    element, options
                )
            else:
                self.envelope_investments[element] = DiscreteOptionInvestment(
                    element, options, fixed_option=existing
                )

        # comfort control features (smart thermostat, occupancy control,
        # night reduction)
        self.control = ComfortControl(self.cfg, refurbishment)

        # filled in build_parameters
        self._sol_profiles = {}
        self._solar_index = []
        self._profiles = {}
        self._big_m = {}
        self._small_m = {}
        self._element_flows = {}

    def has_free_investment(self):
        return (
            any(not inv.is_fixed for inv in self.envelope_investments.values())
            or self.control.has_free_investment()
        )

    # --- model building ----------------------------------------------

    def _optional_profile(self, key, n):
        if key in self.cfg and self.cfg[key] is not None:
            values = self.cfg[key]
            return values.values if hasattr(values, "values") else np.asarray(values)
        return np.zeros(n)

    def build_parameters(self, model, block):
        n = len(model.timeindex)
        if n != len(self.config.times):
            raise ValueError(
                "The model time index ({} steps) does not match the zone "
                "weather data ({} steps)".format(n, len(self.config.times))
            )
        cfg = self.cfg
        const = self.config.CONST

        # exogenous profiles
        self._profiles["T_e"] = cfg["weather"]["T"].values
        self._profiles["Q_ig"] = self._optional_profile("Q_ig", n)
        self._profiles["occ_nothome"] = self._optional_profile("occ_nothome", n)
        self._profiles["occ_sleeping"] = self._optional_profile("occ_sleeping", n)

        # irradiance on all surface directions
        irrad_surf = calc_surface_irradiance(cfg)

        # reduction factors of the sky view per direction
        F_r = {"North": 0.5, "East": 0.5, "South": 0.5, "West": 0.5, "Horizontal": 1.0}

        # WINDOWS
        # get relevant window area for solar irradiance
        window_area_s = {
            di: cfg["A_Window_" + di] * cfg["F_sh_vert"]
            for di in ["North", "East", "South", "West"]
        }
        window_area_s["Horizontal"] = cfg["A_Window_Horizontal"] * cfg["F_sh_hor"]

        # get relevant window area for thermal irradiance
        window_area_t = sum(cfg["A_Window_" + key] * F_r[key] for key in F_r)

        # get solar radiation on window area
        irrad_on_windows = irrad_surf.mul(pd.Series(window_area_s)).sum(axis=1).values

        # solar gain time series per element and refurbishment option
        # (thermal radiation correction: Schuetz et al. 2017 - eq. 13)
        self._sol_profiles = {element: {} for element in ["Windows", "Walls", "Roof"]}
        win_inv = self.envelope_investments["Windows"]
        for option in win_inv.option_names:
            data = self.envelope_options["Windows"][option]
            thermal_rad = (
                window_area_t
                * const["h_r"]
                * data["U"]
                * const["R_se"]
                * const["delta_T_sky"]
            )
            self._sol_profiles["Windows"][option] = (
                irrad_on_windows * (1.0 - cfg["F_f"]) * cfg["F_w"] * data["g_gl"]
                - thermal_rad
            )

        mean_ver_irr = (
            irrad_surf.loc[:, ["North", "East", "South", "West"]].mean(axis=1).values
        )
        for option in self.envelope_investments["Walls"].option_names:
            H = self.envelope_options["Walls"][option]["H"]
            thermal_rad = H * const["h_r"] * const["R_se"] * const["delta_T_sky"]
            self._sol_profiles["Walls"][option] = (
                mean_ver_irr * H * cfg["F_sh_vert"] * const["R_se"] * const["alpha"]
                - thermal_rad
            )

        mean_roof_irr = irrad_surf.loc[:, ["Roof 1", "Roof 2"]].mean(axis=1).values
        for option in self.envelope_investments["Roof"].option_names:
            H = self.envelope_options["Roof"][option]["H"]
            thermal_rad = H * const["h_r"] * const["R_se"] * const["delta_T_sky"]
            self._sol_profiles["Roof"][option] = (
                mean_roof_irr * H * cfg["F_sh_hor"] * const["R_se"] * const["alpha"]
                - thermal_rad
            )

        # index over all solar-relevant (element, option) combinations
        self._solar_index = [
            (element, option)
            for element in ["Windows", "Walls", "Roof"]
            for option in self.envelope_investments[element].option_names
        ]

        # Big-M/small-m bounds of the switched envelope heat flows
        T_min = cfg["weather"]["T"].min()
        T_max = cfg["weather"]["T"].max()
        for element, inv in self.envelope_investments.items():
            self._big_m[element] = {
                o: self.envelope_options[element][o]["H"]
                * (cfg["comfortT_ub"] - (T_min - 10))
                for o in inv.option_names
            }
            self._small_m[element] = {
                o: self.envelope_options[element][o]["H"]
                * (cfg["comfortT_lb"] - (T_max + 10))
                for o in inv.option_names
            }

        # limit maximal heat or cooling load
        if self.max_load is None:
            self.max_load = self.config.calcDesignHeatLoad()

    def build_variables(self, model, block):
        steps = model.timeindex.steps

        # node temperatures [degC]
        block.T_air = pyomo.Var(steps)
        block.T_s = pyomo.Var(steps)
        block.T_m = pyomo.Var(steps)

        # heating and cooling supply into the air node [kW]
        block.Q_heat = pyomo.Var(steps, within=pyomo.NonNegativeReals)
        block.Q_cool = pyomo.Var(steps, within=pyomo.NonNegativeReals)

        # soft violation of the maximal system load [kW]
        block.max_load_violation = pyomo.Var(within=pyomo.NonNegativeReals)

        # investment decisions
        for inv in self.envelope_investments.values():
            inv.build(model, block)
        self.control.build(model, block)

        # auxiliary variables linearizing the window x solar-gain binary
        # product (Schuetz et al. 2017 - eq. 16); only needed where both
        # decisions are free
        win_inv = self.envelope_investments["Windows"]
        self._px_index = []
        if not win_inv.is_fixed:
            for w in win_inv.option_names:
                for element, option in self._solar_index:
                    if not self.envelope_investments[element].is_fixed:
                        self._px_index.append((w, element, option))
        if self._px_index:
            block.P_X = pyomo.Var(self._px_index, within=pyomo.Binary)

    def _pair_selection(self, block, w, element, option):
        """Product of the window selection `w` and the solar element
        selection (element, option), linearized where both are free."""
        win_inv = self.envelope_investments["Windows"]
        other_inv = self.envelope_investments[element]
        sel_w = win_inv.selection(block, w)
        sel_x = other_inv.selection(block, option)
        if isinstance(sel_w, float) and isinstance(sel_x, float):
            return sel_w * sel_x
        if isinstance(sel_w, float):
            return sel_w * sel_x if sel_w != 0.0 else 0.0
        if isinstance(sel_x, float):
            return sel_x * sel_w if sel_x != 0.0 else 0.0
        return block.P_X[w, element, option]

    # --- 5R1C gain expressions ----------------------------------------

    def _solar_sum(self, block, t):
        """Total solar gains through the chosen envelope options [kW]."""
        return sum(
            self._sol_profiles[element][option][t]
            * self.envelope_investments[element].selection(block, option)
            for element, option in self._solar_index
        )

    def _gain_mass_node(self, block, t):
        """Heat gain of the thermal mass node
        (Schuetz et al. 2017 - eq. 15)."""
        cfg = self.config
        return (
            0.5 * cfg.A_m / cfg.A_tot * self._profiles["Q_ig"][t]
            + cfg.A_f / cfg.A_tot * self._solar_sum(block, t)
        )

    def _gain_surface_node(self, block, t):
        """Heat gain of the thermal mass surface node
        (Schuetz et al. 2017 - eq. 16)."""
        cfg = self.config
        const = cfg.CONST
        win_inv = self.envelope_investments["Windows"]
        win_options = self.envelope_options["Windows"]

        u_win = sum(
            win_options[w]["U"] * win_inv.selection(block, w)
            for w in win_inv.option_names
        )
        cross = sum(
            win_options[w]["H"]
            * self._sol_profiles[element][option][t]
            * self._pair_selection(block, w, element, option)
            for w in win_inv.option_names
            for element, option in self._solar_index
        )
        return (
            (1 - u_win / const["h_ms"] / cfg.A_tot)
            * (0.5 * self._profiles["Q_ig"][t])
            - cross / const["h_ms"] / cfg.A_tot
            + self._solar_sum(block, t)
            - self._gain_mass_node(block, t)
        )

    # --- constraints ---------------------------------------------------

    def build_constraints(self, model, block):
        steps = model.timeindex.steps
        dt = model.timeindex.step_size_h
        cfg = self.config
        T_e = self._profiles["T_e"]

        # linearization of the window x solar binary products
        # (Schuetz et al. 2017 - eq. 16-18)
        if self._px_index:
            win_inv = self.envelope_investments["Windows"]

            def px_rule_1(b, w, element, option):
                return b.P_X[w, element, option] <= self.envelope_investments[
                    element
                ].selection(b, option)

            def px_rule_2(b, w, element, option):
                return b.P_X[w, element, option] <= win_inv.selection(b, w)

            def px_rule_3(b, w, element, option):
                return (
                    b.P_X[w, element, option]
                    >= win_inv.selection(b, w)
                    + self.envelope_investments[element].selection(b, option)
                    - 1
                )

            block.px_con_1 = pyomo.Constraint(self._px_index, rule=px_rule_1)
            block.px_con_2 = pyomo.Constraint(self._px_index, rule=px_rule_2)
            block.px_con_3 = pyomo.Constraint(self._px_index, rule=px_rule_3)

        # switched heat flows of the envelope elements, all driven by
        # the mass node temperature difference to the environment; fixed
        # elements collapse to a plain linear expression without any
        # additional variables
        #
        # POTENTIAL BUG (kept for parity with the pre-refactor model):
        # only the opaque elements may use (T_m - T_e) per Schuetz et al.
        # 2017 - eq. 20/9. Windows should be driven by (T_s - T_e) (eq. 21)
        # and Ventilation by (T_air - T_e) (eq. 22).
        # See docs/model-deviations.md, items 1 and 2.
        driver = lambda b, t: b.T_m[t] - T_e[t]
        for element, inv in self.envelope_investments.items():
            self._element_flows[element] = inv.switched_flow(
                model,
                block,
                "Q_" + element,
                {o: self.envelope_options[element][o]["H"] for o in inv.option_names},
                driver,
                self._big_m[element],
                self._small_m[element],
            )

        flows = self._element_flows

        # 1) energy balance of the thermal mass node including the
        # storage behavior of the building mass
        def mass_node_balance(b, t, t_next):
            return (
                cfg.H_ms * (b.T_m[t] - b.T_s[t])
                + sum(flows[e](b, t) for e in ["Walls", "Roof", "Floor"])
                + cfg.H_door * (b.T_m[t] - T_e[t])
                == self._gain_mass_node(b, t)
                - cfg.C_m * (b.T_m[t_next] - b.T_m[t]) / dt
            )

        block.mass_node_balance = pyomo.Constraint(
            model.timeindex.pairs(), rule=mass_node_balance
        )

        last = len(model.timeindex) - 1
        initial = self.initial_state()
        if model.strategy.wraps_state:
            # periodic wrap: the last step feeds the first one
            block.mass_node_wrap = pyomo.Constraint(
                rule=lambda b: mass_node_balance(b, last, 0)
            )
        elif initial is not None:
            block.mass_node_initial = pyomo.Constraint(
                expr=block.T_m[0] == initial["T_m"]
            )

        # 2) energy balance of the surface node
        def surface_node_balance(b, t):
            return (
                cfg.H_ms * (b.T_s[t] - b.T_m[t])
                + cfg.H_is * (b.T_s[t] - b.T_air[t])
                + flows["Windows"](b, t)
                == self._gain_surface_node(b, t)
            )

        block.surface_node_balance = pyomo.Constraint(
            steps, rule=surface_node_balance
        )

        # 3) energy balance of the air node, supplied by heating and
        # cooling
        #
        # POTENTIAL BUG (kept for parity with the pre-refactor model):
        # per Schuetz et al. 2017 - eq. 22 the air node receives the
        # internal gain Q_ia = 0.5 * Q_ig (eq. 14), not the surface gain
        # Q_st (eq. 19). The pre-refactor model computed a Q_ia term and
        # never used it. See docs/model-deviations.md, item 3.
        def air_node_balance(b, t):
            return (
                flows["Ventilation"](b, t)
                + cfg.H_is * (b.T_air[t] - b.T_s[t])
                == self._gain_surface_node(b, t) - b.Q_cool[t] + b.Q_heat[t]
            )

        block.air_node_balance = pyomo.Constraint(steps, rule=air_node_balance)

        # comfort band, controllable by the installed/invested features:
        # without a smart thermostat the zone is kept at the lower
        # comfort temperature, occupancy control and night reduction
        # relax the band while absent/asleep
        #
        # LIMITATION (deliberate, Kotzur 2018 - eq. 3.2): comfort_ub is a
        # hard bound, so gains above the band force the unbounded Q_cool
        # even for a building without any cooling device. Schuetz et al.
        # 2017 has no upper bound at all (eq. 26) and free-floats instead.
        # See docs/model-deviations.md item 4, written up in full in
        # docs/open-problem-summer-overheating.md.
        sel_smart = lambda b: self.control.selection(b, "SmartThermostat")
        sel_occ = lambda b: self.control.selection(b, "Occupancy")
        sel_night = lambda b: self.control.selection(b, "NightReduction")
        T_lb = cfg["comfortT_lb"]
        T_ub = cfg["comfortT_ub"]
        not_home = self._profiles["occ_nothome"]
        sleeping = self._profiles["occ_sleeping"]

        def comfort_ub(b, t):
            return b.T_air[t] <= (
                T_lb
                + (T_ub - T_lb) * sel_smart(b)
                - (T_ub - 30.0) * not_home[t] * sel_occ(b)
            )

        block.comfort_ub = pyomo.Constraint(steps, rule=comfort_ub)

        def comfort_lb(b, t):
            return b.T_air[t] >= (
                T_lb
                - (T_lb - 18.0) * sleeping[t] * sel_night(b)
                - (T_lb - 14.0) * not_home[t] * sel_occ(b)
            )

        block.comfort_lb = pyomo.Constraint(steps, rule=comfort_lb)

        # occupancy control requires the other controllers to be
        # installed (only relevant while the decisions are free)
        if self.control.has_free_investment():
            block.control_order_1 = pyomo.Constraint(
                expr=sel_occ(block) <= sel_smart(block)
            )
            block.control_order_2 = pyomo.Constraint(
                expr=sel_occ(block) <= sel_night(block)
            )

        # maximal heating/cooling system load as soft constraint in
        # order to avoid infeasibility
        block.max_heating_load = pyomo.Constraint(
            steps,
            rule=lambda b, t: b.Q_heat[t] - b.max_load_violation <= self.max_load,
        )
        block.max_cooling_load = pyomo.Constraint(
            steps,
            rule=lambda b, t: b.Q_cool[t] - b.max_load_violation <= self.max_load,
        )

    # --- coupling, objective, results -----------------------------------

    def ports(self):
        ports = {}
        if self.heat_bus is not None:
            ports["heat"] = Port(
                self.heat_bus, lambda block, t: block.Q_heat[t], sign=-1
            )
        if self.cool_bus is not None:
            ports["cool"] = Port(
                self.cool_bus, lambda block, t: block.Q_cool[t], sign=-1
            )
        return ports

    def objective_terms(self, model, block):
        dt = model.timeindex.step_size_h
        steps = model.timeindex.steps
        terms = []
        # energy cost of unconnected heating/cooling supply
        if self.heat_bus is None:
            terms.append(sum(block.Q_heat[t] * dt for t in steps) * self.heat_cost)
        if self.cool_bus is None:
            terms.append(sum(block.Q_cool[t] * dt for t in steps) * self.cool_cost)
        # penalty of exceeding the maximal system load
        terms.append(block.max_load_violation * self.max_load_violation_penalty)
        # investment cost
        for inv in self.envelope_investments.values():
            terms.extend(inv.annual_cost_terms(model, block))
        terms.extend(self.control.annual_cost_terms(model, block))
        return terms

    def state_variables(self, block):
        return {"T_m": block.T_m}

    def initial_state(self):
        if self.initial_T_m is None:
            return None
        return {"T_m": self.initial_T_m}

    def design_heat_load_value(self, block):
        """Design heat load [kW] including the chosen refurbishment
        (previously the bQ_des variable)."""
        H_total = self.config.H_door
        for element, inv in self.envelope_investments.items():
            for option in inv.option_names:
                H_total += (
                    self.envelope_options[element][option]["H"]
                    * self.DESIGN_ADJUST[element]
                    * inv.selection_value(block, option)
                )
        return H_total * (self.DESIGN_T_INDOOR - self.cfg["design_T_min"])

    def extract_results(self, block):
        if block.max_load_violation.value is not None:
            if block.max_load_violation.value > 1e-9:
                warnings.warn(
                    "Maximal heat load exceeded by "
                    + str(round(block.max_load_violation.value, 3))
                    + " kW",
                    UserWarning,
                )

        timeseries = pd.DataFrame(index=self.config.times)
        timeseries["Heating Load"] = np.array(
            [block.Q_heat[t].value for t in block.Q_heat]
        )
        timeseries["Cooling Load"] = np.array(
            [block.Q_cool[t].value for t in block.Q_cool]
        )
        timeseries["T_air"] = np.array([block.T_air[t].value for t in block.T_air])
        timeseries["T_s"] = np.array([block.T_s[t].value for t in block.T_s])
        timeseries["T_m"] = np.array([block.T_m[t].value for t in block.T_m])
        timeseries["T_e"] = self._profiles["T_e"]

        # investment decision table (previously detailedRefurbish)
        decisions = pd.DataFrame()
        all_investments = dict(self.envelope_investments)
        for feature, inv in self.control.investments.items():
            all_investments["Control_" + feature] = inv
        for label, inv in all_investments.items():
            for option in inv.option_names:
                selected = inv.selection_value(block, option)
                data = inv.options[option]
                decisions.loc["Capacity", (label, option)] = selected
                decisions.loc["CAPEX", (label, option)] = data["capex"] * selected

        static = {
            "Capacity": self.design_heat_load_value(block),
            "FixCost": 0,
            "CAPEX": 0,
            "OPEX fix": 0.0,
            "VarCost": 0.0,
            "OPEX var": 0.0,
            "OPEX": 0.0,
        }

        return {
            "timeseries": timeseries,
            "refurbishment": decisions,
            "static": static,
            "max_load_violation": block.max_load_violation.value,
        }
