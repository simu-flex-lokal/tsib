# -*- coding: utf-8 -*-
"""
Solve strategies. Only perfect foresight is implemented; myopic and
rolling-horizon are documented stubs whose interface requirements
(TimeIndex.slice(), Component.initial_state()/final_state()) are already
part of the framework.
"""

import abc


class SolveStrategy(abc.ABC):
    """
    Determines how the optimization horizon is solved.

    Attributes
    ----------
    wraps_state: bool
        If True, components close their state difference equations
        periodically (last step wraps around to the first step) within
        a single full-horizon solve.
    """

    wraps_state = False

    def apply_state_constraints(self, model, pyomo_model):
        """Hook to add strategy-specific state coupling constraints
        after all components are built."""
        pass

    @abc.abstractmethod
    def supports_investment(self):
        """Whether free investment decisions are allowed in the solve.
        Non-perfect-foresight strategies require investments decided
        upfront (fixed_option / fixed capacity)."""
        raise NotImplementedError


class PerfectForesightStrategy(SolveStrategy):
    """Single solve over the full horizon with periodic state wrap -
    the previous behavior of Building5R1C.sim5R1C."""

    wraps_state = True

    def supports_investment(self):
        return True


class MyopicStrategy(SolveStrategy):
    """
    Sequential solve of consecutive windows, each passing its final
    state as initial state to the next window (not implemented yet).

    Intended pattern: decide investments once in a perfect-foresight
    run, freeze them via fixed_option / fixed capacity, then dispatch
    myopically.
    """

    wraps_state = False

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "MyopicStrategy is a documented interface stub and not implemented yet"
        )

    def supports_investment(self):
        return False


class RollingHorizonStrategy(SolveStrategy):
    """
    Overlapping look-ahead windows re-solved while moving forward in
    time (not implemented yet). Requires TimeIndex.slice() and the
    Component.initial_state()/final_state() hooks, which already exist.
    """

    wraps_state = False

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "RollingHorizonStrategy is a documented interface stub and not implemented yet"
        )

    def supports_investment(self):
        return False
