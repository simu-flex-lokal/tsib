# -*- coding: utf-8 -*-
"""
Solver handling of the building optimization.
"""

import pytest

from tsib.optimization import (
    build_system,
    presets,
    solve,
    solverutils,
    zone_results,
)

from conftest import golden_zone_cfg


#: a first attempt that cannot succeed, to drive the fallback chain
DOOMED = {"time_limit": 1e-6}


def _zone_system():
    return build_system(presets.heat_load_only(), golden_zone_cfg(n_steps=168))


def test_highs_retries_with_other_settings(monkeypatch):
    """HiGHS fails on the zone LP for some inputs - IPX in its basis
    construction, simplex and crossover in postsolve - without that being
    predictable or reproducible, so a failed attempt has to be retried
    rather than reported."""
    monkeypatch.setattr(solverutils, "HIGHS_FALLBACKS", (DOOMED, {}))

    es, nodes = _zone_system()
    model, results = solve(es, solver="highs")

    assert results.termination_condition.name == "optimal"
    # the solution of the successful attempt has to reach the model
    heat = zone_results(model, nodes["thermalzone"])["timeseries"]["Heating Load"]
    assert heat.sum() > 0


def test_highs_reports_every_attempt_when_all_fail(monkeypatch):
    monkeypatch.setattr(solverutils, "HIGHS_FALLBACKS", (DOOMED,))

    es, nodes = _zone_system()
    with pytest.raises(RuntimeError, match="no optimal solution"):
        solve(es, solver="highs")


def test_explicit_options_are_used_as_given(monkeypatch):
    """A caller who passes options wants those options, not a retry with
    something else."""
    monkeypatch.setattr(solverutils, "HIGHS_FALLBACKS", ({},))

    es, nodes = _zone_system()
    with pytest.raises(RuntimeError):
        solve(es, solver="highs", solverOpts=DOOMED)
