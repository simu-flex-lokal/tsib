# -*- coding: utf-8 -*-
"""
Investment decisions as composable objects: an InvestmentOption is
optionally attached to a component (component.investment = ...). Without
an attached investment the component is purely operational and no
investment variables or Big-M constraints are created at all.
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


class DiscreteOptionInvestment(InvestmentOption):
    """
    "Choose exactly one option from a catalog" decision (insulation
    level, window quality, ventilation, control equipment).

    Parameters
    ----------
    name: str, required
        Label used to prefix the pyomo variables on the block (one
        component may hold several discrete investments).
    options: dict, required
        Option name -> dict with keys "capex" [EUR], "lifetime" [a],
        optionally "opex_fix_share" (yearly share of capex) and
        arbitrary further payload entries (e.g. heat transfer values).
    fixed_option: str, optional (default: None)
        If given, the choice is fixed and no binary variables are
        created - the mechanism for "no investment" and for manual
        scenarios.
    """

    def __init__(self, name, options, fixed_option=None):
        if not options:
            raise ValueError("DiscreteOptionInvestment requires at least one option")
        if fixed_option is not None and fixed_option not in options:
            raise ValueError(
                "fixed_option '{}' is not part of the options {}".format(
                    fixed_option, list(options)
                )
            )
        self.name = name
        self.options = options
        self.fixed_option = fixed_option

    @property
    def is_fixed(self):
        return self.fixed_option is not None

    @property
    def option_names(self):
        if self.is_fixed:
            return [self.fixed_option]
        return list(self.options)

    def _var_name(self):
        return self.name + "_select"

    def build(self, model, block):
        if self.is_fixed:
            return
        block.add_component(
            self._var_name(), pyomo.Var(list(self.options), within=pyomo.Binary)
        )
        select = getattr(block, self._var_name())
        # choose exactly one option (Schuetz et al. 2017 - eq. 2)
        block.add_component(
            self.name + "_choose_one",
            pyomo.Constraint(expr=sum(select[o] for o in self.options) == 1),
        )

    def selection(self, block, option):
        """Binary decision of `option` as variable, or as plain 1.0/0.0
        float if the investment is fixed."""
        if self.is_fixed:
            return 1.0 if option == self.fixed_option else 0.0
        return getattr(block, self._var_name())[option]

    def selection_value(self, block, option):
        sel = self.selection(block, option)
        return sel if isinstance(sel, float) else sel.value

    def selected_option(self, block):
        """Name of the chosen option after the solve."""
        if self.is_fixed:
            return self.fixed_option
        select = getattr(block, self._var_name())
        return max(self.options, key=lambda o: select[o].value)

    def value(self, block, values):
        """Expression of an option-dependent coefficient: values[o] of
        the chosen option."""
        if self.is_fixed:
            return values[self.fixed_option]
        return sum(values[o] * self.selection(block, o) for o in self.options)

    def annual_cost_terms(self, model, block):
        terms = []
        for opt_name in self.option_names:
            data = self.options[opt_name]
            annual = data["capex"] * (
                annuity_factor(data["lifetime"], model.wacc)
                + data.get("opex_fix_share", 0.0)
            )
            if annual != 0.0:
                terms.append(
                    annual * self.selection(block, opt_name) * model.timeindex.year_fraction
                )
        return terms

    def switched_flow(self, model, block, flow_name, coefficients, driver, big_m, small_m):
        """
        Reusable Big-M pattern for an option-dependent heat flow
        (generalized form of Schuetz et al. 2017 - eq. 8, 23-25):

            flow[o, t] == coefficients[o] * driver(block, t)   if o chosen
            flow[o, t] == 0                                    otherwise

        Parameters
        ----------
        flow_name: str, required
            Name of the created flow variable on the block.
        coefficients: dict, required
            Option name -> linear coefficient of the driver.
        driver: callable(block, t) -> pyomo expression, required
            e.g. the temperature difference (T_m[t] - T_e[t]).
        big_m, small_m: dict, required
            Option name -> upper/lower bound of the flow.

        Returns
        -------
        callable(block, t) -> total flow expression over all options
        """
        steps = model.timeindex.steps

        if self.is_fixed:
            coeff = coefficients[self.fixed_option]
            return lambda b, t, _c=coeff: _c * driver(b, t)

        options = list(self.options)
        block.add_component(flow_name, pyomo.Var(options, steps))
        flow = getattr(block, flow_name)

        # if an option is not chosen, force its flow to be zero
        # (Schuetz et al. 2017 - eq. 23-25)
        def flow_zero_ub(b, o, t):
            return flow[o, t] <= self.selection(b, o) * big_m[o]

        def flow_zero_lb(b, o, t):
            return flow[o, t] >= self.selection(b, o) * small_m[o]

        # if the option is chosen, enforce the linear flow equation
        # (Schuetz et al. 2017 - eq. 8, 23-25)
        def flow_active_ub(b, o, t):
            return (
                coefficients[o] * driver(b, t) - flow[o, t]
                <= (1 - self.selection(b, o)) * big_m[o]
            )

        def flow_active_lb(b, o, t):
            return (
                coefficients[o] * driver(b, t) - flow[o, t]
                >= (1 - self.selection(b, o)) * small_m[o]
            )

        for suffix, rule in [
            ("_zero_ub", flow_zero_ub),
            ("_zero_lb", flow_zero_lb),
            ("_active_ub", flow_active_ub),
            ("_active_lb", flow_active_lb),
        ]:
            block.add_component(
                flow_name + suffix, pyomo.Constraint(options, steps, rule=rule)
            )

        return lambda b, t: sum(flow[o, t] for o in options)
