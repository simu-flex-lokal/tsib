# -*- coding: utf-8 -*-
"""
The plotting helpers tsib keeps: the profiles it initializes, and the
thermal zone against its comfort band.

Nothing here solves anything - the plot functions consume frames and result
dicts, so a hand-built dict of pandas Series exercises them completely and
in milliseconds. Plots of a solved energy system live in `esmkit.plotting`.

Assertions are on the returned Axes rather than on pixels - what matters is
that the right number of artists carry the right labels, that `window`
actually narrows the data, and that no function draws to a global figure or
calls plt.show().
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from tsib.envelope import comfort_bounds  # noqa: E402
from tsib.plotting import (  # noqa: E402
    plot_heatmap,
    plot_load_duration,
    plot_profiles,
    plot_zone,
    use_style,
)

N = 48
INDEX = pd.date_range("2010-01-01", periods=N, freq="h")


def series(values):
    return pd.Series(np.asarray(values, dtype=float), index=INDEX)


@pytest.fixture
def results():
    """A solved thermal zone, by hand - whatever produced it."""
    return {
        "thermalzone": {
            "timeseries": pd.DataFrame({
                "Heating Load": np.full(N, 4.0),
                "Cooling Load": np.zeros(N),
                "T_air": np.full(N, 21.5),
                "T_m": np.full(N, 21.0),
                "T_e": np.full(N, 2.0),
            }, index=INDEX),
            "static": {},
        },
    }


@pytest.fixture
def load():
    """A load profile with a distinct evening peak."""
    hours = INDEX.hour.values
    return series(np.where((hours >= 17) & (hours < 21), 3.5, 1.0))


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


# --- the rules ---------------------------------------------------------


def test_every_plot_returns_its_axes(results, load):
    fig, ax = plt.subplots()
    frame = pd.DataFrame({"Heating Load": load})

    assert plot_zone(results, band_cfg(), ax=ax) is ax
    assert plot_load_duration(load, ax=ax) is ax
    assert plot_profiles(frame, ax=ax) is ax
    assert plot_heatmap(load, ax=ax) is ax


def test_plots_create_their_own_axes_when_given_none(results):
    before = len(plt.get_fignums())
    ax = plot_zone(results, band_cfg())

    assert len(plt.get_fignums()) == before + 1
    assert ax.figure in [plt.figure(n) for n in plt.get_fignums()]


def test_window_narrows_the_data(results):
    full = plot_zone(results, band_cfg())
    windowed = plot_zone(results, band_cfg(), window=12)

    assert windowed.get_xlim()[1] - windowed.get_xlim()[0] < (
        full.get_xlim()[1] - full.get_xlim()[0]
    )


def test_window_accepts_a_date_range(results):
    ax = plot_zone(results, band_cfg(),
                   window=("2010-01-01", "2010-01-01 05:00"))
    line = ax.get_lines()[0]

    assert len(line.get_xdata()) == 6


def test_use_style_does_not_leak():
    before = plt.rcParams["axes.grid"]
    with use_style():
        pass

    assert plt.rcParams["axes.grid"] == before


# --- profiles ----------------------------------------------------------


def test_load_duration_is_sorted_descending(load):
    ax = plot_load_duration(load)
    values = ax.get_lines()[0].get_ydata()

    assert np.all(np.diff(values) <= 1e-9)


def test_load_duration_overlays_a_dict_of_scenarios(load):
    ax = plot_load_duration({"flat": load, "dynamic": load * 0.5})

    assert len(ax.get_lines()) == 2
    assert {t.get_text() for t in ax.get_legend().get_texts()} == {
        "flat", "dynamic"
    }


def test_heatmap_needs_a_datetime_index():
    with pytest.raises(TypeError, match="DatetimeIndex"):
        plot_heatmap(pd.Series(np.arange(10)))


def test_heatmap_lays_out_hours_against_days():
    index = pd.date_range("2010-01-01", periods=24 * 10, freq="h")
    ax = plot_heatmap(pd.Series(np.arange(len(index), dtype=float), index=index))
    image = ax.images[0]

    assert image.get_array().shape == (24, 10)
    assert ax.get_ylabel() == "hour of day"


def test_profiles_take_their_unit_from_the_building():
    frame = pd.DataFrame({"Heating Load": np.full(N, 3.0)}, index=INDEX)
    ax = plot_profiles(frame, units={"Heating Load": "kW_{th}"})

    assert "kW_{th}" in ax.get_ylabel()


def test_profiles_leave_the_unit_blank_when_columns_disagree():
    frame = pd.DataFrame({"Heating Load": np.full(N, 3.0),
                          "Photovoltaic 1": np.full(N, 0.4)}, index=INDEX)
    ax = plot_profiles(frame, units={"Heating Load": "kW_{th}",
                                     "Photovoltaic 1": "kW/kWp"})

    assert ax.get_ylabel() == ""
    assert len(ax.get_lines()) == 2


def test_profiles_resample_for_a_whole_year_view():
    index = pd.date_range("2010-01-01", periods=24 * 14, freq="h")
    frame = pd.DataFrame({"Heating Load": np.arange(len(index),
                                                    dtype=float)}, index=index)
    ax = plot_profiles(frame, resample="D")

    assert len(ax.get_lines()[0].get_ydata()) == 14


def test_profiles_report_an_unknown_column():
    frame = pd.DataFrame({"Heating Load": np.full(N, 3.0)}, index=INDEX)
    with pytest.raises(KeyError, match="No profile"):
        plot_profiles(frame, columns="Cooling Load")


# --- the comfort band, which the zone plot depends on -------------------


def band_cfg(**overrides):
    cfg = {
        "comfortT_lb": 21.0,
        "comfortT_ub": 24.0,
        "capControl": True,
        "occControl": False,
        "nightReduction": True,
        "occ_sleeping": np.where(INDEX.hour < 6, 1.0, 0.0),
        "occ_nothome": np.zeros(N),
    }
    cfg.update(overrides)
    return cfg


def test_night_reduction_lowers_the_floor_while_occupants_sleep():
    lower, upper = comfort_bounds(band_cfg(), N)

    night = INDEX.hour < 6
    assert lower[night].max() == pytest.approx(18.0)
    assert lower[~night].min() == pytest.approx(21.0)
    assert np.all(upper == 24.0)


def test_the_band_is_flat_with_every_control_switched_off():
    lower, upper = comfort_bounds(
        band_cfg(capControl=False, nightReduction=False), N
    )
    # without capControl the ceiling collapses onto the floor: the zone is
    # held at comfortT_lb and has no thermal flexibility at all
    assert np.all(lower == 21.0)
    assert np.all(upper == 21.0)


def test_occ_control_widens_the_band_while_nobody_is_home():
    cfg = band_cfg(occControl=True, occ_nothome=np.where(INDEX.hour == 10,
                                                         1.0, 0.0))
    lower, upper = comfort_bounds(cfg, N)

    away = INDEX.hour == 10
    assert lower[away].max() == pytest.approx(14.0)
    assert upper[away].min() == pytest.approx(30.0)


def test_zone_plot_draws_the_effective_band(results):
    ax = plot_zone(results, band_cfg())
    labels = [t.get_text() for t in ax.get_legend().get_texts()]

    assert "comfort band" in labels
    assert {"T_air", "T_m", "T_e"} <= set(labels)


def test_zone_plot_can_omit_the_band(results):
    with_band = plot_zone(results, band_cfg())
    without = plot_zone(results, band_cfg(), band=False)

    assert len(without.collections) < len(with_band.collections)
