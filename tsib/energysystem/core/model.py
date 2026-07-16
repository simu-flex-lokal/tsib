# -*- coding: utf-8 -*-
"""
EnergySystemModel: orchestrator which composes registered components
and buses into a single pyomo MILP, solves it and collects the results
per component.
"""

import logging

import pyomo.environ as pyomo

from .bus import Bus
from .component import Component
from .strategy import PerfectForesightStrategy
from .timeindex import TimeIndex
from . import solverutils


class EnergySystemModel(object):
    """
    Container of an energy system: components coupled via buses,
    optimized jointly over a TimeIndex.

    Parameters
    ----------
    timeindex: TimeIndex or pandas.DatetimeIndex, required
        Time discretization of the optimization horizon.
    wacc: float, optional (default: 0.08)
        Interest rate used to annualize all investments.
    strategy: SolveStrategy, optional (default: PerfectForesightStrategy)
    """

    def __init__(self, timeindex, wacc=0.08, strategy=None):
        if not isinstance(timeindex, TimeIndex):
            timeindex = TimeIndex(timeindex)
        self.timeindex = timeindex
        self.wacc = wacc
        self.strategy = strategy if strategy is not None else PerfectForesightStrategy()
        self.components = {}
        self.buses = {}
        self.pyomo_model = None
        self.solver_results = None

    def add(self, *items):
        """Registers Bus and Component instances with the model."""
        for item in items:
            if isinstance(item, Bus):
                if item.name in self.buses:
                    raise ValueError("Duplicate bus name '{}'".format(item.name))
                self.buses[item.name] = item
            elif isinstance(item, Component):
                if item.name in self.components:
                    raise ValueError(
                        "Duplicate component name '{}'".format(item.name)
                    )
                self.components[item.name] = item
            else:
                raise ValueError(
                    "Only Bus and Component instances can be added, got {}".format(
                        type(item)
                    )
                )
        return items[0] if len(items) == 1 else items

    def build(self):
        """
        Creates the pyomo model: one Block per component, generic bus
        balances, and the total cost objective as sum of all component
        objective terms.
        """
        # fail fast: free investment decisions require a strategy which
        # supports them (perfect foresight)
        if not self.strategy.supports_investment():
            for comp in self.components.values():
                if comp.has_free_investment():
                    raise ValueError(
                        "Component '{}' has a free investment decision, which "
                        "strategy {} does not support. Fix the investment "
                        "(fixed_option / fixed capacity) first, e.g. from a "
                        "preceding perfect-foresight run.".format(
                            comp.name, type(self.strategy).__name__
                        )
                    )

        M = pyomo.ConcreteModel()

        for bus in self.buses.values():
            bus._reset()

        # every component gets its own namespace
        for comp in self.components.values():
            M.add_component(comp.name, pyomo.Block())

        # register all ports at their buses
        for comp in self.components.values():
            for port in comp.ports().values():
                if port.bus.name not in self.buses:
                    raise ValueError(
                        "Component '{}' is connected to bus '{}' which was "
                        "not added to the model".format(comp.name, port.bus.name)
                    )
                port.bus._register(comp, port)

        # build components in three passes so that constraints can rely
        # on all parameters/variables being present
        for comp in self.components.values():
            comp.build_parameters(self, getattr(M, comp.name))
        for comp in self.components.values():
            comp.build_variables(self, getattr(M, comp.name))
        for comp in self.components.values():
            comp.build_constraints(self, getattr(M, comp.name))

        # generic energy balances
        for bus in self.buses.values():
            bus.build_balance_constraint(self, M)

        # strategy hook for additional state coupling
        self.strategy.apply_state_constraints(self, M)

        # total cost objective
        terms = []
        for comp in self.components.values():
            terms.extend(comp.objective_terms(self, getattr(M, comp.name)))
        M.obj = pyomo.Objective(expr=sum(terms), sense=pyomo.minimize)

        self.pyomo_model = M
        return M

    def solve(self, solver=None, tee=False, solverOpts=None):
        """
        Builds (if necessary) and solves the model.

        Parameters
        ----------
        solver: str, optional (default: $SOLVER or auto-detected)
        tee: bool, optional (default: False)
            Stream the solver log.
        """
        if self.pyomo_model is None:
            self.build()

        self.solver_results = solverutils.solve_model(
            self.pyomo_model, solver=solver, tee=tee, solverOpts=solverOpts
        )
        logging.info(
            "Energy system model solved: objective = {}".format(
                pyomo.value(self.pyomo_model.obj)
            )
        )
        return self.solver_results

    @property
    def objective_value(self):
        return pyomo.value(self.pyomo_model.obj)

    def results(self, component=None):
        """
        Collects the solved results per component.

        Parameters
        ----------
        component: str, optional (default: None)
            Name of a single component whose results to return;
            otherwise a dict over all components is returned.
        """
        if self.pyomo_model is None:
            raise RuntimeError("Model has not been solved yet")
        if component is not None:
            return self.components[component].extract_results(
                getattr(self.pyomo_model, component)
            )
        return {
            name: comp.extract_results(getattr(self.pyomo_model, name))
            for name, comp in self.components.items()
        }
