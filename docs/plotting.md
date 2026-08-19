# Plotting

`tsib.plotting` draws tsib results and the profiles tsib initializes. It needs no dependency
beyond `matplotlib`, which tsib already requires.

```python
import matplotlib.pyplot as plt
import tsib
from tsib import plotting

results = bdg.optimize(build_spec(params))

with plotting.use_style():
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    plotting.plot_dispatch(results, "elec", ax=a1, window=("2010-01-04", "2010-01-11"))
    plotting.plot_zone(results, bdg.cfg, ax=a2, window=("2010-01-04", "2010-01-11"))
    plt.show()
```

## The one rule

**Every function takes an optional `ax` and returns the axes it drew on.** One function draws one
thing; figures are composed by the caller. Nothing calls `plt.show()`, nothing writes a file, and
nothing touches global state outside `use_style()` — whether a plot ends up in a notebook, a
script or a PNG is not the library's decision.

Omit `ax` and the function creates its own figure, which is what you want at a prompt.

## A year does not fit on a line plot

8760 points render as a solid block. Four ways out, answering different questions:

| | how | question it answers |
|---|---|---|
| `window=` | `12`, `slice(0, 48)`, `("2010-01-04", "2010-01-11")`, `"2010-01"` | what happens on these days? |
| `resample=` | `"D"`, `"W"` (`plot_profiles`) | what does the year look like? |
| `plot_heatmap` | day-of-year × hour-of-day carpet | *when* does consumption happen, across the whole year? |
| `plot_load_duration` | sorted descending | how many hours sit near the peak? |

The last two are the ones a grid study needs. A tariff that moves consumption shows up in the
heatmap as a change in the vertical banding, and in the load duration curve as a steeper head —
often while total energy barely moves.

## Functions

### Results

| function | draws |
|---|---|
| `plot_dispatch(results, bus)` | every flow on one bus, stacked: supply above zero, demand below, net as a dotted line that should sit on zero |
| `plot_zone(results, cfg)` | `T_air` / `T_m` / `T_e` against the **effective** comfort band |
| `plot_storage(results)` | storage content of every storage in the system |
| `plot_grid_exchange(results, price=…)` | net exchange with the expensive hours shaded and the tariff on a second axis |
| `plot_price_response(results, price)` | mean import power per price bin |
| `plot_load_duration(series \| dict)` | load duration curve; a dict overlays scenarios |
| `plot_heatmap(series)` | day × hour carpet of an annual series |

### Initialization

| function | draws |
|---|---|
| `plot_profiles(bdg, columns=…)` | columns of `bdg.timeseries`, labelled from `bdg.units` |

### Derived series, no plotting involved

Both are useful on their own and carry the logic the plots sit on:

```python
plotting.bus_flows(results, "elec")        # DataFrame, one signed column per component
plotting.net_grid_exchange(results)        # Series [kW], import positive
```

`net_grid_exchange` is the single series a grid powerflow computation needs per building.

## Things worth knowing

**Signs.** In `bus_flows`, positive is energy *into* the bus — generation, import, storage
discharge — and negative is energy *out* of it. A component doing both nets to one column, so a
battery reads positive while discharging and negative while charging. In `net_grid_exchange`,
**positive means importing**.

**The thermal zone is not a flow.** `node_results` reports the zone as a `"timeseries"` frame, not
as `in_`/`out_` flows, so `bus_flows` reads its heating and cooling loads out of that frame. It
assumes the template's bus names (`heat`, `cool`); pass `spec=` if yours differ.

**The comfort band is not flat.** `comfortT_lb` / `comfortT_ub` are only the nominal band.
`plot_zone` draws the *effective* one from
[`optimization.comfort_bounds`](../tsib/optimization/zone5r1c.py) — the same function the
optimization constrains against, so the picture cannot drift from the model. With the default
`nightReduction=True` the floor drops while the occupants sleep, and with `capControl=False` the
ceiling collapses onto the floor and the zone has no thermal flexibility at all. A flat band
would hide both.

**The zone's loads are not in `plot_zone`.** They are flows on the heat bus, so
`plot_dispatch(results, "heat")` shows them together with whatever supplies them.

**Storage content has one value too many.** solph reports it on time *points*, without a time
index; `plot_storage` trims the last value and reattaches the model's index.

**`plot_price_response` plots mean power, not energy**, because price bins hold different numbers
of hours. A flat bar chart means the tariff changed nothing; a downward slope means the building
buys when it is cheap. The hour count per bin is annotated so a bin resting on few hours is not
over-read.

## Colours

Components are coloured by role, so a technology keeps its colour across figures: heat
`#c1440e`, PV `#e0a800`, grid `#4a4a4a`, battery `#2e6f9e`. A component the palette does not know
falls back to a cycle. `ROLE_COLORS` is a plain dict — assign into it to add your own.

## Seeing all of it at once

```bash
SOLVER=highs uv run python examples/energysystem/parameterization_demo.py --plots out/
```

writes a contact sheet of every function against a solved year.
