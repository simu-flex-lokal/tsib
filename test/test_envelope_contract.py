# -*- coding: utf-8 -*-
"""
The 5R1C contract: ten scalars and five time series, derived from a
building configuration without solving anything.

These are the numbers tsib hands to whatever computes the heat load. They
used to be obtainable only by instantiating the MILP zone component; the
arithmetic now stands on its own in `tsib.envelope.gains`, and this pins
it against the capture taken while the old component still existed
(data/golden/zone_contract_*, provenance in capture_meta.json).

If a term of the gain equations drifts, this fails - without building a
model, in a second, and independently of any solver.
"""

import json
import os

import numpy as np
import pandas as pd
import pytest

from conftest import GOLDEN_DIR, golden, golden_zone_cfg

import tsib
from tsib.envelope import ZONE_SERIES, zone_parameters


SCALARS = ["H_ms", "H_is", "H_door", "C_m", "max_load", "design_capacity"]


def contract(horizon):
    """The captured contract: (scalars, frame of the five series)."""
    with open(golden("zone_contract_params_{}.json".format(horizon))) as handle:
        scalars = json.load(handle)
    source = "zone_contract_{}.csv{}".format(
        horizon, ".gz" if horizon == "year" else ""
    )
    frame = pd.read_csv(golden(source), index_col=0)
    return scalars, frame


@pytest.fixture(scope="module")
def params_168h():
    return zone_parameters(golden_zone_cfg(n_steps=168))


def test_the_scalars_match_the_capture(params_168h):
    expected, _ = contract("168h")
    for key in SCALARS:
        assert params_168h[key] == pytest.approx(expected[key], abs=1e-9)


def test_the_envelope_coefficients_match_the_capture(params_168h):
    expected, _ = contract("168h")
    assert set(params_168h["H"]) == set(expected["H"])
    for element, value in expected["H"].items():
        assert params_168h["H"][element] == pytest.approx(value, abs=1e-9)


def test_the_series_match_the_capture(params_168h):
    _, frame = contract("168h")
    assert list(frame.columns) == ZONE_SERIES
    for key in ZONE_SERIES:
        values = np.asarray(params_168h[key], dtype=float)
        assert len(values) == 168
        np.testing.assert_allclose(values, frame[key].values, atol=1e-9)


def test_the_whole_year_matches_the_capture():
    """The short horizon truncates; the year exercises every season, the
    solar gains at both solstices and the full comfort band."""
    params = zone_parameters(golden_zone_cfg())
    expected, frame = contract("year")

    for key in SCALARS:
        assert params[key] == pytest.approx(expected[key], abs=1e-9)
    for key in ZONE_SERIES:
        np.testing.assert_allclose(
            np.asarray(params[key], dtype=float), frame[key].values, atol=1e-9
        )


def test_a_building_reports_its_own_contract():
    """`Building.zone_parameters()` is the same thing over `self.cfg`."""
    from conftest import example_building

    bdg, _ = example_building()
    params = bdg.zone_parameters()

    assert set(params) == set(SCALARS) | {"H"} | set(ZONE_SERIES)
    assert params["C_m"] > 0
    assert len(params["T_e"]) == len(bdg.cfg["weather"].index)
    # the design heat load is the default ceiling of the heating system
    assert params["max_load"] == pytest.approx(
        bdg.zone_config.calcDesignHeatLoad()
    )


def test_max_load_can_be_overridden():
    params = zone_parameters(golden_zone_cfg(n_steps=168), max_load=3.0)
    assert params["max_load"] == 3.0
    # the reported design capacity is a property of the envelope, not of
    # the installed system, so it does not follow
    assert params["design_capacity"] != 3.0


def test_the_ventilation_control_path_is_still_refused():
    cfg = golden_zone_cfg(n_steps=168)
    cfg["ventControl"] = True
    with pytest.raises(NotImplementedError):
        zone_parameters(cfg)


def test_the_capture_is_the_one_from_esmkit():
    """Provenance guard: the fixture was read off tsib's own zone at a
    named commit, so a re-capture is a deliberate act, not a side effect."""
    with open(os.path.join(GOLDEN_DIR, "capture_meta.json")) as handle:
        meta = json.load(handle)
    assert meta["captured_from"] == "tsib"
    assert meta["n_steps_short"] == 168
    assert meta["n_steps_year"] == 8760
