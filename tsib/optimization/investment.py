# -*- coding: utf-8 -*-
"""
Investment decisions as composable objects: an InvestmentOption is
optionally attached to a component (component.investment = ...). Without
an attached investment the component is purely operational and no
investment variables are created at all.
"""

import abc

import pyomo.environ as pyomo


def annuity_factor(lifetime, wacc):
    """
    Annuity factor to distribute an investment over its lifetime with
    interest rate `wacc`. Falls back to straight-line depreciation for
    wacc == 0.
    """
    if wacc > 0:
        return ((1 + wacc) ** lifetime) * wacc / (((1 + wacc) ** lifetime) - 1)
    else:
        return 1.0 / lifetime


class InvestmentOption(abc.ABC):
    """
    Base class of investment decisions. `is_fixed` investments create no
    decision variables - this is the mechanism for "no investment" and
    for manually fixed scenarios, and also the intended pattern for
    freezing a perfect-foresight investment decision before subsequent
    myopic/rolling-horizon dispatch windows.
    """

    @property
    @abc.abstractmethod
    def is_fixed(self):
        raise NotImplementedError

    @abc.abstractmethod
    def build(self, model, block):
        """Adds the investment decision variables to the block."""
        raise NotImplementedError

    @abc.abstractmethod
    def annual_cost_terms(self, model, block):
        """Cost expressions [EUR per optimization horizon]: annualized
        investment scaled by the covered fraction of a year."""
        raise NotImplementedError


class ContinuousInvestment(InvestmentOption):
    """
    Continuous capacity decision (storage kWh, PV kWp, ...).

    Parameters
    ----------
    capex_per_unit: float, required
        Investment cost per capacity unit [EUR/unit].
    lifetime: float, required
        Economic lifetime [a].
    opex_fix_share: float, optional (default: 0.)
        Yearly fix operational cost as share of the investment.
    capacity: float, optional (default: None)
        If given, the capacity is fixed to this value and no decision
        variable is created.
    min_capacity, max_capacity: float, optional
        Bounds of the free capacity decision.
    """

    def __init__(
        self,
        capex_per_unit,
        lifetime,
        opex_fix_share=0.0,
        capacity=None,
        min_capacity=0.0,
        max_capacity=None,
    ):
        self.capex_per_unit = capex_per_unit
        self.lifetime = lifetime
        self.opex_fix_share = opex_fix_share
        self.fixed_capacity = capacity
        self.min_capacity = min_capacity
        self.max_capacity = max_capacity

    @property
    def is_fixed(self):
        return self.fixed_capacity is not None

    def build(self, model, block):
        if not self.is_fixed:
            block.capacity = pyomo.Var(
                within=pyomo.NonNegativeReals,
                bounds=(self.min_capacity, self.max_capacity),
            )

    def capacity_expr(self, block):
        """The capacity as float (fixed) or decision variable."""
        if self.is_fixed:
            return self.fixed_capacity
        return block.capacity

    def capacity_value(self, block):
        if self.is_fixed:
            return self.fixed_capacity
        return block.capacity.value

    def annual_cost_terms(self, model, block):
        specific_annual = self.capex_per_unit * (
            annuity_factor(self.lifetime, model.wacc) + self.opex_fix_share
        )
        return [
            specific_annual * self.capacity_expr(block) * model.timeindex.year_fraction
        ]

