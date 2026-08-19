# -*- coding: utf-8 -*-
"""
Looking at tsib results with `tsib.plotting`.

This is the worked example for `docs/plotting.md`. It solves one building
under two tariffs - flat and dynamic - and then draws the same year five
different ways.

The point is not that the functions render. It is that each view answers a
different question. Total imported energy barely moves between the two
tariffs; *when* the energy is drawn moves completely, and the annual peak
moves the wrong way. Which of those you see depends entirely on the view
you pick, which is why a tariff study needs more than one.

Every function takes `ax=` and returns the axes, so the figures below are
composed here rather than inside the library.

Run:
    SOLVER=highs uv run python examples/energysystem/plotting_demo.py
    SOLVER=highs uv run python examples/energysystem/plotting_demo.py --show
    SOLVER=highs uv run python examples/energysystem/plotting_demo.py --out figures/
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

import tsib
from tsib import plotting
from tsib.optimization import BuildingSystemParameters, build_spec


#: a week in January, where the heat pump dominates the grid draw
WEEK = ("2010-01-04", "2010-01-11")

SHEET = BuildingSystemParameters(
    equipment={
        "heat_pump": {"capacity_kw": 25.0},
        "pv": {"kwp": 8.0},
        "battery": {"capacity_kwh": 10.0, "power_kw": 3.0},
        "buffer": {"capacity_kwh": 20.0, "standby_loss_kW": 0.02},
    },
    tariff={"import": "@elecPrice", "export": 0.08},
    meta={"bus_id": "pylovo-42", "freq": "h"},
)


def dynamic_tariff(index):
    """Cheap at night, a peak in the evening - a crude day-ahead shape."""
    price = np.full(len(index), 0.28)
    price[index.hour < 6] = 0.18
    price[(index.hour >= 17) & (index.hour < 21)] = 0.55
    return price


def flat_tariff(index):
    """The same average price, with no signal to react to."""
    return np.full(len(index), dynamic_tariff(index).mean())


def solve_both():
    """
    The same building under a flat and a dynamic tariff.

    Only the price signal differs - same archetype, same equipment, same
    occupancy realization - so every difference in the figures is the
    building reacting to the tariff and nothing else. Both tariffs have the
    same mean, so the comparison is not confounded by one simply being
    cheaper.
    """
    runs = {}
    for label, tariff in (("flat tariff", flat_tariff),
                          ("dynamic tariff", dynamic_tariff)):
        bdg = tsib.Building(tsib.BuildingConfiguration({
            "ID": "DE.N.SFH.06.Gen.ReEx.001.001",
            "n_persons": 2,
            "seed": 42,
            "comfortT_lb": 21.0,
            "comfortT_ub": 24.0,
        }))
        bdg.cfg["elecPrice"] = tariff(bdg.cfg["weather"].index)
        runs[label] = (bdg, bdg.optimize(build_spec(SHEET)))
    return runs


# --- the figures --------------------------------------------------------


def figure_overview(bdg, results):
    """
    The everyday view: what happened, on which bus, over one week.

    Read it top down. The heat pump is the big negative block on the
    electricity bus and the big positive one on the heat bus; the zone and
    the hot water are what consume that heat; and the zone's air
    temperature shows how the comfort band was spent to make it possible.
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    plotting.plot_dispatch(results, "elec", ax=axes[0], window=WEEK)
    plotting.plot_dispatch(results, "heat", ax=axes[1], window=WEEK)
    plotting.plot_zone(results, bdg.cfg, ax=axes[2], window=WEEK)
    for ax in axes[:2]:
        ax.set_xlabel("")
    fig.tight_layout()
    return fig


def figure_grid_and_storage(bdg, results):
    """
    What the grid sees, and where the flexibility is parked.

    The shaded hours are the expensive ones. The exchange should collapse
    to zero inside them while the two storages - the battery and the heat
    buffer - discharge to cover the gap.
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    plotting.plot_grid_exchange(results, price=bdg.cfg["elecPrice"],
                                ax=axes[0], window=WEEK)
    plotting.plot_storage(results, ax=axes[1], window=WEEK)
    axes[0].set_xlabel("")
    fig.tight_layout()
    return fig


def figure_annual(results):
    """
    The whole year, two ways.

    The carpet shows *when* electricity is drawn across all 8760 hours: a
    bright band overnight, a hole in the evening peak, and negative
    (exporting) midday in summer. The load duration curve throws the time
    axis away and asks the question a grid connection actually cares about
    - how many hours sit near the peak.
    """
    net = plotting.net_grid_exchange(results).rename("net grid exchange")

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    plotting.plot_heatmap(net, ax=axes[0])
    plotting.plot_load_duration(net, ax=axes[1])
    fig.tight_layout()
    return fig


def figure_comparison(runs, price):
    """
    Flat against dynamic - the comparison the study is built on.

    `plot_load_duration` takes a dict of scenarios, so both years overlay
    on one axes: watch the head of the curve, where the dynamic tariff
    builds a *higher* peak than the flat one while shifting energy away
    from the expensive hours.

    Both runs are binned by the *dynamic* price, including the one that
    never saw it - that is what makes the two panels comparable. A flat
    bar chart means the building ignored the signal, a downward slope
    means it followed it.
    """
    nets = {label: plotting.net_grid_exchange(results)
            for label, (_, results) in runs.items()}

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    plotting.plot_load_duration(nets, ax=axes[0], title="Load duration")
    # the two price panels share a scale, otherwise matplotlib rescales each
    # one and the inversion between them stops being visible
    axes[2].sharey(axes[1])
    for ax, (label, (_, results)) in zip(axes[1:], runs.items()):
        plotting.plot_price_response(results, price, ax=ax,
                                     title="Price response - {}".format(label))
    fig.tight_layout()
    return fig


def figure_profiles(bdg):
    """
    The other half of tsib: the profiles it initialized before optimizing.

    The unit on the axis comes from `bdg.units`, not from this script.
    Note that "Heating Load" is absent - `optimize()` does not write it into
    `bdg.timeseries`, only `getHeatLoad()` does.
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    plotting.plot_profiles(bdg, columns=["Electricity Load", "Hot Water Load"],
                           ax=axes[0], window=WEEK,
                           title="Occupancy driven loads, one week")
    plotting.plot_profiles(bdg, columns=["Photovoltaic 1"], resample="D",
                           ax=axes[1], title="Specific PV yield, daily mean")
    fig.tight_layout()
    return fig


FIGURES = [
    ("overview", "dispatch per bus and the zone, one week"),
    ("grid_and_storage", "what the grid sees, and where the flexibility sits"),
    ("annual", "the whole year: carpet and load duration"),
    ("comparison", "flat tariff against dynamic"),
    ("profiles", "the initialized profiles"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="plots",
                        help="directory for the PNGs (default: plots)")
    parser.add_argument("--show", action="store_true",
                        help="open the figures instead of writing them")
    args = parser.parse_args()

    print("solving the building twice - the first run simulates the")
    print("occupancy profile and takes minutes\n")
    runs = solve_both()
    bdg, results = runs["dynamic tariff"]
    price = dynamic_tariff(bdg.cfg["weather"].index)
    expensive = price >= 0.5

    print("{:16s} {:>10s} {:>10s} {:>12s}".format(
        "tariff", "import", "peak", "in peak h"))
    for label, (_, run) in runs.items():
        imported = np.clip(plotting.net_grid_exchange(run).values, 0, None)
        print("{:16s} {:9.0f}  {:9.1f}  {:11.1%}".format(
            label, imported.sum(), imported.max(),
            imported[expensive].sum() / imported.sum()))
    print("\nunits: kWh and kW; the last column is the share of the import "
          "drawn\nin the {:.0%} of hours that are expensive.".format(
              expensive.mean()))

    with plotting.use_style():
        figures = {
            "overview": figure_overview(bdg, results),
            "grid_and_storage": figure_grid_and_storage(bdg, results),
            "annual": figure_annual(results),
            "comparison": figure_comparison(runs, price),
            "profiles": figure_profiles(bdg),
        }

    print()
    if args.show:
        plt.show()
        return

    os.makedirs(args.out, exist_ok=True)
    for name, description in FIGURES:
        path = os.path.join(args.out, name + ".png")
        figures[name].savefig(path, dpi=110)
        plt.close(figures[name])
        print("  {:22s} {}".format(os.path.basename(path), description))
    print("\nwrote {} figures to {}".format(len(FIGURES),
                                            os.path.abspath(args.out)))


if __name__ == "__main__":
    main()
