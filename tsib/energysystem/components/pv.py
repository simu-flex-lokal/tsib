# -*- coding: utf-8 -*-
"""
Photovoltaic generator: feeds a specific yield profile (e.g. the output
of tsib.simPhotovoltaic [kW/kWp]) scaled by a fixed or optimized
capacity into an electricity bus.
"""

import numpy as np
import pandas as pd
import pyomo.environ as pyomo

from ..core.component import Component, Port
from ..core.investment import ContinuousInvestment
from ._profiles import profile_values


class PVGenerator(Component):
    """
    PV plant with fixed or continuously optimized capacity.

    Parameters
    ----------
    name: str, required
    bus: Bus, required
        Electricity bus the generator feeds into.
    specific_yield: pandas.Series, required
        Generation per installed capacity [kW/kWp], e.g. from
        tsib.simPhotovoltaic.
    capacity: float or ContinuousInvestment, required
        Installed capacity [kWp], either fixed or as investment
        decision.
    curtailable: bool, optional (default: True)
        Whether generation may be curtailed below the potential.
    """

    def __init__(self, name, bus, specific_yield, capacity, curtailable=True):
        super().__init__(name)
        self.bus = bus
        self.specific_yield = specific_yield
        if isinstance(capacity, ContinuousInvestment):
            self.investment = capacity
        else:
            self.investment = ContinuousInvestment(
                capex_per_unit=0.0, lifetime=25.0, capacity=float(capacity)
            )
        self.curtailable = curtailable
        self._yield = None

    def build_parameters(self, model, block):
        self._yield = profile_values(
            self.specific_yield, len(model.timeindex), self.name + " specific_yield"
        )

    def build_variables(self, model, block):
        self.investment.build(model, block)
        if self.curtailable:
            block.generation = pyomo.Var(
                model.timeindex.steps, within=pyomo.NonNegativeReals
            )

    def build_constraints(self, model, block):
        if self.curtailable:
            block.generation_limit = pyomo.Constraint(
                model.timeindex.steps,
                rule=lambda b, t: b.generation[t]
                <= self._yield[t] * self.investment.capacity_expr(b),
            )

    def ports(self):
        if self.curtailable:
            flow = lambda block, t: block.generation[t]
        else:
            flow = lambda block, t: self._yield[t] * self.investment.capacity_expr(
                block
            )
        return {"out": Port(self.bus, flow, sign=1)}

    def objective_terms(self, model, block):
        return self.investment.annual_cost_terms(model, block)

    def extract_results(self, block):
        capacity = self.investment.capacity_value(block)
        if self.curtailable:
            generation = np.array(
                [block.generation[t].value for t in block.generation]
            )
        else:
            generation = self._yield * capacity
        return {"capacity": capacity, "generation": pd.Series(generation)}
