# -*- coding: utf-8 -*-
"""
Generic energy storage with electrical and thermal specializations.

The base class is deliberately shaped so that a future EVBattery can be
attached purely via inheritance/constructor parameters:
- charge/discharge power limits accept time series (0 while driving,
  charger power while plugged in),
- _external_energy_delta() is a protected hook in the SOC balance
  (used by ThermalStorage for standby losses, and by a future EVBattery
  for forced driving-energy withdrawal).
"""

import numpy as np
import pandas as pd
import pyomo.environ as pyomo

from ..core.component import Component, Port
from ..core.investment import ContinuousInvestment
from ._profiles import profile_values


class StorageComponent(Component):
    """
    Energy storage on a single bus.

    Parameters
    ----------
    name: str, required
    bus: Bus, required
    capacity: float or ContinuousInvestment, required
        Storage capacity [kWh], fixed or as investment decision.
    charge_efficiency: float, optional (default: 0.95)
    discharge_efficiency: float, optional (default: 0.95)
    self_discharge_per_h: float, optional (default: 0.)
        Relative SOC loss per hour.
    charge_power_limit: float or pandas.Series, optional
        Maximal charging power [kW]; time series allowed (e.g. future
        EV availability).
    discharge_power_limit: float or pandas.Series, optional
    soc_min, soc_max: float, optional (default: 0., 1.)
        Usable SOC window as share of the capacity.
    initial_soc: float, optional (default: None)
        Initial state of charge [kWh]. Only applied by strategies
        without a periodic state wrap (myopic/rolling-horizon window
        chaining); under perfect foresight the SOC is closed
        periodically instead.
    """

    carrier = None

    def __init__(
        self,
        name,
        bus,
        capacity,
        charge_efficiency=0.95,
        discharge_efficiency=0.95,
        self_discharge_per_h=0.0,
        charge_power_limit=None,
        discharge_power_limit=None,
        soc_min=0.0,
        soc_max=1.0,
        initial_soc=None,
    ):
        super().__init__(name)
        self.bus = bus
        if isinstance(capacity, ContinuousInvestment):
            self.investment = capacity
        else:
            self.investment = ContinuousInvestment(
                capex_per_unit=0.0, lifetime=20.0, capacity=float(capacity)
            )
        self.charge_efficiency = charge_efficiency
        self.discharge_efficiency = discharge_efficiency
        self.self_discharge_per_h = self_discharge_per_h
        self.charge_power_limit = charge_power_limit
        self.discharge_power_limit = discharge_power_limit
        self.soc_min = soc_min
        self.soc_max = soc_max
        self.initial_soc = initial_soc
        self._charge_lim = None
        self._discharge_lim = None

    def build_parameters(self, model, block):
        n = len(model.timeindex)
        self._charge_lim = profile_values(
            self.charge_power_limit, n, self.name + " charge_power_limit"
        )
        self._discharge_lim = profile_values(
            self.discharge_power_limit, n, self.name + " discharge_power_limit"
        )

    def build_variables(self, model, block):
        steps = model.timeindex.steps
        self.investment.build(model, block)
        block.soc = pyomo.Var(steps, within=pyomo.NonNegativeReals)
        block.charge = pyomo.Var(steps, within=pyomo.NonNegativeReals)
        block.discharge = pyomo.Var(steps, within=pyomo.NonNegativeReals)

    def _external_energy_delta(self, block, t):
        """Additional exogenous energy change [kW] in the SOC balance
        (default 0). Protected hook: ThermalStorage uses it for standby
        heat losses, a future EVBattery would use it for the forced
        driving-energy withdrawal."""
        return 0.0

    def _soc_change(self, block, t, dt):
        """Energy change of the SOC within step t [kWh]."""
        return (
            block.charge[t] * self.charge_efficiency
            - block.discharge[t] / self.discharge_efficiency
            + self._external_energy_delta(block, t)
        ) * dt - block.soc[t] * self.self_discharge_per_h * dt

    def build_constraints(self, model, block):
        steps = model.timeindex.steps
        dt = model.timeindex.step_size_h
        cap = self.investment.capacity_expr

        block.soc_upper = pyomo.Constraint(
            steps, rule=lambda b, t: b.soc[t] <= self.soc_max * cap(b)
        )
        block.soc_lower = pyomo.Constraint(
            steps, rule=lambda b, t: b.soc[t] >= self.soc_min * cap(b)
        )

        if self._charge_lim is not None:
            block.charge_limit = pyomo.Constraint(
                steps, rule=lambda b, t: b.charge[t] <= self._charge_lim[t]
            )
        if self._discharge_lim is not None:
            block.discharge_limit = pyomo.Constraint(
                steps, rule=lambda b, t: b.discharge[t] <= self._discharge_lim[t]
            )

        # SOC difference equation
        block.soc_balance = pyomo.Constraint(
            model.timeindex.pairs(),
            rule=lambda b, t, t_next: b.soc[t_next]
            == b.soc[t] + self._soc_change(b, t, dt),
        )

        # state coupling of the horizon boundary
        last = len(model.timeindex) - 1
        initial = self.initial_state()
        if model.strategy.wraps_state:
            # periodic wrap: the last step feeds the first one
            block.soc_wrap = pyomo.Constraint(
                expr=block.soc[0] == block.soc[last] + self._soc_change(block, last, dt)
            )
        elif initial is not None:
            block.soc_initial = pyomo.Constraint(
                expr=block.soc[0] == initial["soc"]
            )

    def ports(self):
        return {
            "storage": Port(
                self.bus, lambda block, t: block.discharge[t] - block.charge[t], sign=1
            )
        }

    def objective_terms(self, model, block):
        return self.investment.annual_cost_terms(model, block)

    def state_variables(self, block):
        return {"soc": block.soc}

    def initial_state(self):
        if self.initial_soc is None:
            return None
        return {"soc": self.initial_soc}

    def extract_results(self, block):
        return {
            "capacity": self.investment.capacity_value(block),
            "soc": pd.Series(np.array([block.soc[t].value for t in block.soc])),
            "charge": pd.Series(
                np.array([block.charge[t].value for t in block.charge])
            ),
            "discharge": pd.Series(
                np.array([block.discharge[t].value for t in block.discharge])
            ),
        }


class ElectricalStorage(StorageComponent):
    """Battery storage on an electricity bus."""

    carrier = "electricity"


class ThermalStorage(StorageComponent):
    """
    Hot water/thermal storage on a heat bus with a constant standby
    heat loss, implemented via the _external_energy_delta hook.

    Parameters
    ----------
    standby_loss_kW: float or pandas.Series, optional (default: 0.)
        Constant heat loss to the environment [kW].
    """

    carrier = "heat"

    def __init__(self, name, bus, capacity, standby_loss_kW=0.0, **kwargs):
        super().__init__(name, bus, capacity, **kwargs)
        self.standby_loss_kW = standby_loss_kW
        self._standby_loss = None

    def build_parameters(self, model, block):
        super().build_parameters(model, block)
        self._standby_loss = profile_values(
            self.standby_loss_kW, len(model.timeindex), self.name + " standby_loss_kW"
        )

    def _external_energy_delta(self, block, t):
        return -self._standby_loss[t]
