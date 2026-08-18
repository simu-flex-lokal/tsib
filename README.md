[![Build Status](https://github.com/simu-flex-lokal/tsib/actions/workflows/test.yml/badge.svg)](https://github.com/simu-flex-lokal/tsib/actions/workflows/test.yml)

# tsib - Time Series Initialization for Buildings

tsib is a python package that builds up on different databases and models for creating consistent demand and production time series of residential buildings. This could be either occupancy behavior, electricity demand or heat demand time series as well as photovoltaic (PV) and solar thermal production time series.

> **This is a fork.** It is maintained at FH Aachen, Institute NOWUM Energy,
> and has diverged substantially from the original [FZJ IEK-3 tsib](https://github.com/FZJ-IEK3-VSA/tsib),
> which was last updated in 2023. It is **not** the `tsib` package on PyPI — installing that
> gives you the original project, which is different software. Install from this repository
> (see [Installation](#installation)).
>
> What changed here: the package was modernized for Python 3.12-3.14 and current dependencies,
> and the in-house MILP layer was replaced by an energy system optimization built on
> [oemof-solph](https://github.com/oemof/oemof-solph). See [`docs/`](docs/) for the current
> architecture, and [Origin and attribution](#origin-and-attribution) for credit and licensing.


## Features
* flexible configuration of single buildings by different input arguments
* simple building definition based on an archetype building catalogue
* consideration of the occupancy behavior
* derivation of the electric device load or the demand for thermal comfort
* calculation of the heat load based on a thermal building model
* optimization of the building energy system - dispatch, flexibility and investment sizing - solving the thermal zone jointly with storage, PV and price signals
* provision of location specific time series for solar irradiation and temperature based on weather data


## Applied databases and models
tsib is a flexible tool which allows the use of different models and databases for the generation of time series for buildings. The following databases and models are included in tsib:
* [CREST](https://www.lboro.ac.uk/research/crest/demand-model/) demand model for the simulaton of the occupancy behavior
* [5R1C](https://www.sciencedirect.com/science/article/abs/pii/S0306261916314933) thermal building model 
* [pvlib](https://github.com/pvlib/pvlib-python) for solar irradiance calculation and photovoltaic simulation
* [TABULA/EPISCOPE](http://episcope.eu/) archetype building catalogue
* [DWD Testreferenzjahre](https://www.dwd.de/DE/leistungen/testreferenzjahre/testreferenzjahre.html)  for providing weather data
* [oemof-solph](https://github.com/oemof/oemof-solph) as the optimization framework for the building energy system


## Installation

This fork is not published on a package index, so install it from source. Clone a local copy of
the repository to your computer

	git clone https://github.com/simu-flex-lokal/tsib.git

and install it with [uv](https://docs.astral.sh/uv/) (recommended - the repository ships a
`uv.lock`)

	cd tsib
	uv sync --extra highs

or with pip

	cd tsib
	pip install '.[highs]'

tsib requires Python 3.12 or newer.

### Solver

The 5R1C thermal building model and every other energy system optimization are solved as a
(MI)LP, so tsib needs a solver. The free, open-source default is HiGHS, pulled in by the `highs`
extra used above. Gurobi is available the same way (`--extra gurobi` / `'.[gurobi]'`), and
separately installed cplex, scip or cbc installations are picked up as well.

Solvers are auto-detected in the order `gurobi, cplex, scip, cbc, highs` - commercial ones first,
HiGHS last as the free fallback. Set the `$SOLVER` environment variable to force one explicitly:

	SOLVER=highs python your_script.py

Note that glpk is not supported for this model.

### Development

	uv sync --group dev --extra highs
	uv run pytest


## Examples

This [jupyter notebook](examples/showcase.ipynb) shows the capabilites of tsib to create all relevant time series.

For the energy system side, [`EnergySystemDemo.ipynb`](examples/energysystem/EnergySystemDemo.ipynb)
walks through flexibility, PV and battery investment sizing, and a full-year whole-building
workflow, while [`chp_component.py`](examples/energysystem/chp_component.py) shows how to add your
own technology to the building block kit.

Further documentation lives in [`docs/`](docs/), in particular
[`docs/energysystem.md`](docs/energysystem.md).


## Origin and attribution

tsib was created at the [Institute of Energy and Climate Research - Techno-economic Systems
Analysis (IEK-3)](https://www.fz-juelich.de/en/iek/iek-3) of
[Forschungszentrum Jülich](https://www.fz-juelich.de/en) by Leander Kotzur, Timo Kannengießer,
Kevin Knosala, Peter Stenzel, Peter Markewitz, Martin Robinius and Detlef Stolten. The original
project lives at [FZJ-IEK3-VSA/tsib](https://github.com/FZJ-IEK3-VSA/tsib).

That original work was supported by the Helmholtz Association under the Joint Initiative
["Energy System 2050 - A Contribution of the Research Field Energy"](https://www.helmholtz.de/en/research/energy/energy_system_2050/).

If you use tsib in a published work, please [**cite the following publication**](http://juser.fz-juelich.de/record/858675),
which applies the original tsib to the creation of time series for residential buildings in
Germany.

This fork is developed independently at FH Aachen, Institute NOWUM Energy. It is not endorsed by
or affiliated with Forschungszentrum Jülich, and questions about it should go to this repository
rather than to the original authors.


## License

MIT - see [`LICENSE`](LICENSE) for the full notice.

Copyright for the original work is held by Leander Kotzur, Timo Kannengießer, Kevin Knosala,
Peter Stenzel, Peter Markewitz, Martin Robinius and Detlef Stolten (FZJ IEK-3); see
[Origin and attribution](#origin-and-attribution).

Modifications copyright (C) 2026 FH Aachen, Institute NOWUM Energy.
