# -*- coding: utf-8 -*-
"""
Grid connection: import (and optionally export) of an energy carrier at
a time-varying price. Also used as generic heat/cool supply with an
energy price.
"""

import numpy as np
import pandas as pd
import pyomo.environ as pyomo

from ..core.component import Component, Port
from ._profiles import profile_values


class GridConnection(Component):
    """
    Import/export connection of a bus to an external network.

    Parameters
    ----------
    name: str, required
    bus: Bus, required
    import_price: float or pandas.Series, required
        Price of imported energy [EUR/kWh].
    export_price: float or pandas.Series, optional (default: None)
        Remuneration of exported energy [EUR/kWh]; if None, no export
        is possible.
    max_import: float or pandas.Series, optional (default: unbounded)
    max_export: float or pandas.Series, optional (default: unbounded)
    """

    def __init__(
        self,
        name,
        bus,
        import_price,
        export_price=None,
        max_import=None,
        max_export=None,
    ):
        super().__init__(name)
        self.bus = bus
        self.import_price = import_price
        self.export_price = export_price
        self.max_import = max_import
        self.max_export = max_export
        self._p_imp = None
        self._p_exp = None
        self._imp_lim = None
        self._exp_lim = None

    @property
    def has_export(self):
        return self.export_price is not None

    def build_parameters(self, model, block):
        n = len(model.timeindex)
        self._p_imp = profile_values(self.import_price, n, self.name + " import_price")
        self._p_exp = profile_values(self.export_price, n, self.name + " export_price")
        self._imp_lim = profile_values(self.max_import, n, self.name + " max_import")
        self._exp_lim = profile_values(self.max_export, n, self.name + " max_export")

    def build_variables(self, model, block):
        steps = model.timeindex.steps
        block.import_ = pyomo.Var(steps, within=pyomo.NonNegativeReals)
        if self.has_export:
            block.export_ = pyomo.Var(steps, within=pyomo.NonNegativeReals)

    def build_constraints(self, model, block):
        steps = model.timeindex.steps
        if self._imp_lim is not None:
            block.import_limit = pyomo.Constraint(
                steps, rule=lambda b, t: b.import_[t] <= self._imp_lim[t]
            )
        if self.has_export and self._exp_lim is not None:
            block.export_limit = pyomo.Constraint(
                steps, rule=lambda b, t: b.export_[t] <= self._exp_lim[t]
            )

    def ports(self):
        if self.has_export:
            flow = lambda block, t: block.import_[t] - block.export_[t]
        else:
            flow = lambda block, t: block.import_[t]
        return {"grid": Port(self.bus, flow, sign=1)}

    def objective_terms(self, model, block):
        dt = model.timeindex.step_size_h
        terms = [
            sum(
                block.import_[t] * self._p_imp[t] * dt for t in model.timeindex.steps
            )
        ]
        if self.has_export:
            terms.append(
                -sum(
                    block.export_[t] * self._p_exp[t] * dt
                    for t in model.timeindex.steps
                )
            )
        return terms

    def extract_results(self, block):
        results = {
            "import": pd.Series(
                np.array([block.import_[t].value for t in block.import_])
            )
        }
        if self.has_export:
            results["export"] = pd.Series(
                np.array([block.export_[t].value for t in block.export_])
            )
        return results
