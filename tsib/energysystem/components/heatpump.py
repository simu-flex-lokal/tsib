# -*- coding: utf-8 -*-
"""
Tutorial component: an air source heat pump coupling an electricity bus
to a heat bus with a temperature dependent COP.
"""

import numpy as np
import pandas as pd
import pyomo.environ as pyomo

from ..core.component import Component, Port
from ..core.investment import ContinuousInvestment
from ._profiles import profile_values


class HeatPump(Component):
    """
    Heat pump which converts electricity into heat with a time varying
    coefficient of performance.

    Parameters
    ----------
    name: str, required
    elec_bus: Bus, required
        Electricity bus the compressor draws its power from.
    heat_bus: Bus, required
        Heat bus the delivered heat is fed into.
    cop: float or pandas.Series, required
        Coefficient of performance [-], e.g. from tsib.simHeatpump.
        Steps with cop <= 0 shut the heat pump off.
    capacity: float or ContinuousInvestment, required
        Thermal capacity [kW_th], fixed or as investment decision.
    """

    def __init__(self, name, elec_bus, heat_bus, cop, capacity):
        super().__init__(name)
        self.elec_bus = elec_bus
        self.heat_bus = heat_bus
        self.cop = cop
        if isinstance(capacity, ContinuousInvestment):
            self.investment = capacity
        else:
            self.investment = ContinuousInvestment(
                capex_per_unit=0.0, lifetime=18.0, capacity=float(capacity)
            )
        self._cop = None

    def build_parameters(self, model, block):
        self._cop = profile_values(self.cop, len(model.timeindex), self.name + " cop")

    def build_variables(self, model, block):
        self.investment.build(model, block)
        block.heat_out = pyomo.Var(model.timeindex.steps, within=pyomo.NonNegativeReals)

    def build_constraints(self, model, block):
        cap = self.investment.capacity_expr

        block.capacity_limit = pyomo.Constraint(
            model.timeindex.steps,
            rule=lambda b, t: b.heat_out[t] <= cap(b),
        )
        # cut-off steps (cop == 0) cannot deliver any heat
        off = [t for t in model.timeindex.steps if self._cop[t] <= 0.0]
        if off:
            block.cutoff = pyomo.Constraint(off, rule=lambda b, t: b.heat_out[t] == 0.0)

    def _power_in(self, block, t):
        if self._cop[t] <= 0.0:
            return 0.0
        return block.heat_out[t] / self._cop[t]

    def ports(self):
        return {
            "heat": Port(self.heat_bus, lambda b, t: b.heat_out[t], sign=1),
            "elec": Port(self.elec_bus, self._power_in, sign=-1),
        }

    def objective_terms(self, model, block):
        return self.investment.annual_cost_terms(model, block)

    def extract_results(self, block):
        heat = np.array([block.heat_out[t].value for t in block.heat_out])
        power = np.where(
            self._cop > 0.0, heat / np.where(self._cop > 0, self._cop, 1), 0.0
        )
        return {
            "capacity": self.investment.capacity_value(block),
            "heat": pd.Series(heat),
            "power": pd.Series(power),
        }
