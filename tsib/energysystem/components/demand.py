# -*- coding: utf-8 -*-
"""
Fixed demand time series on a bus.
"""

import pandas as pd

from ..core.component import Component, Port
from ._profiles import profile_values


class FixedDemand(Component):
    """
    Exogenous, inflexible demand profile which withdraws energy from a
    bus.

    Parameters
    ----------
    name: str, required
    bus: Bus, required
        The bus the demand is connected to.
    profile: float or pandas.Series, required
        Demand per time step [kW].
    """

    def __init__(self, name, bus, profile):
        super().__init__(name)
        self.bus = bus
        self.profile = profile
        self._values = None

    def build_parameters(self, model, block):
        self._values = profile_values(
            self.profile, len(model.timeindex), name=self.name + " profile"
        )

    def ports(self):
        return {
            "out": Port(self.bus, lambda block, t: self._values[t], sign=-1)
        }

    def extract_results(self, block):
        return {"load": pd.Series(self._values)}
