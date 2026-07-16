# -*- coding: utf-8 -*-
"""
Component abstraction of the energy system model. Every component gets
its own pyomo.Block and communicates with other components exclusively
via declared Ports on shared Buses.
"""

import abc

from pyomo.core.base.block import Block

# attribute names pyomo reserves on blocks (e.g. "load") cannot be used
# as component or bus names
PYOMO_RESERVED_NAMES = getattr(Block, "_Block_reserved_words", set())


def check_name(name, kind):
    if not name.isidentifier():
        raise ValueError(
            "{} name '{}' has to be a valid python identifier".format(kind, name)
        )
    if name in PYOMO_RESERVED_NAMES:
        raise ValueError(
            "{} name '{}' is a reserved pyomo attribute - "
            "please choose a different name".format(kind, name)
        )


class Port(object):
    """
    Declares an energy flow of a component into (sign=+1) or out of
    (sign=-1) a Bus.

    Parameters
    ----------
    bus: Bus, required
        The energy balance node the flow is connected to.
    flow: callable(block, t) -> pyomo expression, required
        Nonnegative flow expression on the component's block at step t.
    sign: int, optional (default: +1)
        +1 injects into the bus, -1 withdraws from the bus.
    """

    def __init__(self, bus, flow, sign=1):
        if sign not in (1, -1):
            raise ValueError("Port sign has to be +1 or -1")
        self.bus = bus
        self.flow = flow
        self.sign = sign


class Component(abc.ABC):
    """
    Base class of all energy system components. Subclasses declare
    their pyomo parameters/variables/constraints on the block handed in
    by EnergySystemModel.build() and expose their bus couplings via
    ports().

    Parameters
    ----------
    name: str, required
        Unique component name, also used as pyomo block name.
    """

    def __init__(self, name):
        check_name(name, "Component")
        self.name = name
        # optionally attached investment decision (composition, not
        # inheritance): None means fully parameterized operation
        self.investment = None

    def build_parameters(self, model, block):
        """Pre-calculations and parameters, no variables/constraints."""
        pass

    def build_variables(self, model, block):
        """Adds all decision variables to the component block."""
        pass

    def build_constraints(self, model, block):
        """Adds all constraints to the component block."""
        pass

    def objective_terms(self, model, block):
        """Cost expressions to be added to the model objective."""
        return []

    @abc.abstractmethod
    def ports(self):
        """dict of port name -> Port declaring all bus couplings."""
        raise NotImplementedError

    def extract_results(self, block):
        """Reads the solved variable values into plain python/pandas
        objects."""
        return {}

    # --- state hooks for future myopic/rolling-horizon chaining -----

    def state_variables(self, block):
        """dict of state name -> pyomo Var (indexed by time step) which
        couples consecutive solve windows (e.g. SOC, T_m)."""
        return {}

    def initial_state(self):
        """dict of state name -> initial value, or None if the state is
        free (perfect foresight with periodic wrap)."""
        return None

    def final_state(self, block):
        """dict of state name -> value at the last time step, to be
        passed as initial_state of a subsequent solve window."""
        states = {}
        for key, var in self.state_variables(block).items():
            last = max(var.index_set())
            states[key] = var[last].value
        return states

    def has_free_investment(self):
        """Whether an unfixed investment decision is attached (used by
        the strategy fail-fast guard)."""
        return self.investment is not None and not self.investment.is_fixed

    def __repr__(self):
        return "{}('{}')".format(type(self).__name__, self.name)
