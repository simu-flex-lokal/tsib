# -*- coding: utf-8 -*-
"""
Unit tests of the energy system framework core: TimeIndex, Bus/Port
balances, EnergySystemModel orchestration and the solve strategy seam.
"""

import numpy as np
import pandas as pd
import pytest

import tsib.energysystem as es


def hourly_index(n, start="2010-01-01"):
    return pd.date_range(start, periods=n, freq="h", tz="UTC")


def test_timeindex():
    tix = es.TimeIndex(hourly_index(24))
    assert tix.step_size_h == 1.0
    assert len(tix) == 24
    assert tix.hours == 24.0
    assert list(tix.steps) == list(range(24))
    assert tix.pairs()[0] == (0, 1)
    assert tix.pairs()[-1] == (22, 23)

    sub = tix.slice(6, 12)
    assert len(sub) == 12
    assert sub.times[0] == tix.times[6]

    quarter = es.TimeIndex(pd.date_range("2010-01-01", periods=8, freq="15min"))
    assert quarter.step_size_h == 0.25
    assert quarter.hours == 2.0


def test_annuity_factor():
    # zero interest falls back to straight-line depreciation
    assert es.annuity_factor(20, 0.0) == pytest.approx(1.0 / 20)
    # standard annuity for 8% over 40 years
    a = es.annuity_factor(40, 0.08)
    assert a == pytest.approx(0.08 * 1.08 ** 40 / (1.08 ** 40 - 1))


def test_source_sink_balance():
    """A fixed demand supplied by a priced grid connection: the balance
    has to force import == demand and the objective the energy cost."""
    n = 24
    demand = pd.Series(np.linspace(1.0, 3.0, n), index=hourly_index(n))

    model = es.EnergySystemModel(hourly_index(n), wacc=0.05)
    bus = model.add(es.Bus("elec", carrier="electricity"))
    model.add(es.FixedDemand("demand", bus, demand))
    model.add(es.GridConnection("grid", bus, import_price=0.30))

    model.solve(tee=False)

    imported = model.results("grid")["import"]
    np.testing.assert_allclose(imported.values, demand.values, atol=1e-6)
    assert model.objective_value == pytest.approx(demand.sum() * 0.30, rel=1e-6)


def test_unconnected_bus_raises():
    model = es.EnergySystemModel(hourly_index(4))
    model.add(es.Bus("elec"))
    with pytest.raises(ValueError, match="no connected components"):
        model.build()


def test_strategy_stubs():
    with pytest.raises(NotImplementedError):
        es.MyopicStrategy()
    with pytest.raises(NotImplementedError):
        es.RollingHorizonStrategy()
    assert es.PerfectForesightStrategy().supports_investment()
    assert es.PerfectForesightStrategy().wraps_state


def test_investment_guard():
    """A strategy without investment support has to reject components
    with free investment decisions."""

    class NoInvestStrategy(es.SolveStrategy):
        wraps_state = False

        def supports_investment(self):
            return False

    n = 24
    tix = hourly_index(n)
    model = es.EnergySystemModel(tix, strategy=NoInvestStrategy())
    bus = model.add(es.Bus("elec"))
    model.add(es.FixedDemand("demand", bus, 1.0))
    model.add(es.GridConnection("grid", bus, import_price=0.30))
    model.add(
        es.PVGenerator(
            "pv",
            bus,
            specific_yield=pd.Series(0.5, index=tix),
            capacity=es.ContinuousInvestment(capex_per_unit=1000.0, lifetime=25),
        )
    )
    with pytest.raises(ValueError, match="free investment"):
        model.build()


def test_explicit_perfect_foresight_matches_default():
    """Passing PerfectForesightStrategy explicitly reproduces the
    default behavior."""
    n = 48
    tix = hourly_index(n)
    hour = pd.DatetimeIndex(tix).hour
    price = pd.Series(np.where(hour < 12, 0.05, 0.40), index=tix)

    def run(strategy=None):
        kwargs = {"strategy": strategy} if strategy is not None else {}
        model = es.EnergySystemModel(tix, **kwargs)
        bus = model.add(es.Bus("elec"))
        model.add(es.FixedDemand("demand", bus, 1.0))
        model.add(es.GridConnection("grid", bus, import_price=price))
        model.add(es.ElectricalStorage("battery", bus, capacity=10.0))
        model.solve(tee=False)
        return model.objective_value

    assert run(es.PerfectForesightStrategy()) == pytest.approx(run(), rel=1e-9)


def test_initial_state_without_wrap():
    """Without a periodic state wrap, a given initial state fixes the
    first step (window-chaining interface for future strategies)."""

    class NoWrapStrategy(es.SolveStrategy):
        wraps_state = False

        def supports_investment(self):
            return True

    n = 24
    tix = hourly_index(n)
    model = es.EnergySystemModel(tix, strategy=NoWrapStrategy())
    bus = model.add(es.Bus("elec"))
    model.add(es.FixedDemand("demand", bus, 1.0))
    model.add(es.GridConnection("grid", bus, import_price=0.30))
    battery = model.add(
        es.ElectricalStorage(
            "battery",
            bus,
            capacity=10.0,
            charge_efficiency=1.0,
            discharge_efficiency=1.0,
            initial_soc=5.0,
        )
    )
    model.solve(tee=False)

    results = model.results("battery")
    assert results["soc"].iloc[0] == pytest.approx(5.0, abs=1e-6)
    # final_state exposes the last-step state for the next window
    final = battery.final_state(getattr(model.pyomo_model, "battery"))
    assert "soc" in final


def test_duplicate_names_raise():
    model = es.EnergySystemModel(hourly_index(4))
    bus = model.add(es.Bus("elec"))
    model.add(es.FixedDemand("demand", bus, 1.0))
    with pytest.raises(ValueError, match="Duplicate component"):
        model.add(es.FixedDemand("demand", bus, 2.0))
