# -*- coding: utf-8 -*-
"""
Time discretization of the energy system model.
"""

import pandas as pd


class TimeIndex(object):
    """
    Immutable wrapper around a pandas.DatetimeIndex which defines the
    time discretization of an EnergySystemModel.

    Parameters
    ----------
    times: pandas.DatetimeIndex, required
        Time stamps of the optimization horizon.
    step_size_h: float, optional (default: derived from the index)
        Step size in hours.
    """

    def __init__(self, times, step_size_h=None):
        if not isinstance(times, pd.DatetimeIndex):
            times = pd.DatetimeIndex(times)
        if len(times) < 2:
            raise ValueError("TimeIndex requires at least two time steps")
        if step_size_h is None:
            step_size_h = (times[1] - times[0]).total_seconds() / 3600.0
        self._times = times
        self._step_size_h = float(step_size_h)

    @property
    def times(self):
        return self._times

    @property
    def step_size_h(self):
        """Step size in hours."""
        return self._step_size_h

    @property
    def steps(self):
        """Integer positions of all time steps."""
        return range(len(self._times))

    @property
    def hours(self):
        """Total covered hours of the horizon."""
        return len(self._times) * self._step_size_h

    @property
    def year_fraction(self):
        """Share of a full year covered by the horizon, used to scale
        annualized investment cost against operational cost."""
        return self.hours / 8760.0

    def pairs(self):
        """Consecutive step tuples (t, t+1) for difference equations,
        excluding the wrap-around of the last step."""
        return [(t, t + 1) for t in range(len(self._times) - 1)]

    def slice(self, start, length):
        """Sub-horizon of `length` steps beginning at position `start`
        (interface for future rolling-horizon strategies)."""
        return TimeIndex(self._times[start : start + length], self._step_size_h)

    def __len__(self):
        return len(self._times)

    def __eq__(self, other):
        return (
            isinstance(other, TimeIndex)
            and self._step_size_h == other._step_size_h
            and self._times.equals(other._times)
        )

    def __repr__(self):
        return "TimeIndex({} steps of {} h from {})".format(
            len(self._times), self._step_size_h, self._times[0]
        )
