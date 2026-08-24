# Parameter reference

Every input `tsib` accepts, what it is for, and what it actually changes downstream.

There are three levels, and the first two are easy to confuse:

1. **`BuildingConfiguration` kwargs** — *what the building is*: geometry, fabric, weather,
   occupants, comfort. This is the parameter space defined by `KWARG_TYPES` /
   `KWARG_DEFAULTS` at the top of `tsib/buildingconfig.py`.
2. **`BuildingSystemParameters`** — *what the building has*: which equipment is installed and
   how big it is. The exchange format, see [§9](#9-the-equipment-sheet).
3. **Spec / preset parameters** — *how it is wired*: the components and buses of the
   optimization itself. See [§8](#8-spec--preset-parameters).

The building configuration does **not** decide its own energy system. A building has PV because
its equipment sheet says so — `BuildingSystemParameters(equipment={"pv": {"kwp": 8.0}})` — or
because a preset builds one. See
[§7](#7-parameters-that-do-less-than-their-name-suggests).

## Minimal input

A TABULA/EPISCOPE archetype ID alone is enough — everything else has a default or is derived:

```python
import tsib
from tsib.optimization import presets

bdg = tsib.Building(tsib.BuildingConfiguration({"ID": "DE.N.SFH.05.Gen.ReEx.001.001"}))
results = bdg.optimize(presets.hp_pv_battery(pv_kwp=8.0, battery_kwh=10.0, hp_kw=25.0))
```

The only hard external requirement is a MILP solver (`uv sync --extra highs`, or `$SOLVER`).

## How the kwargs are validated

`BuildingConfiguration(kwargs)` is strict:

- a key not in `KWARG_TYPES` raises `ValueError`;
- a value of the wrong type raises `ValueError` (numpy scalars are accepted and converted to
  Python scalars);
- a value outside an enumerated list raises `ValueError`;
- a `NOT_IMPLEMENTED` parameter raises `NotImplementedError`;
- a key with value `None` is silently dropped, so `{"roofTilt": None}` means "use the default";
- a key that survives to the end unconsumed produces
  `UserWarning: Keyword X is not used for the building parameterization` — but only if its
  value differs from the default. This is how you find out a parameter was ignored.

---

## 1. Archetype selection

These pick a row out of the EPISCOPE/TABULA database (`tsib/data/episcope/episcope.csv`), which
supplies the geometry and the U-values of the construction. Either name the row directly with
`ID`, or describe the building and let `_get_typ_building` query for the best match.

| Parameter | Type | Default | What it does |
|---|---|---|---|
| `ID` | str | — | The archetype's `Code_BuildingVariant`, e.g. `DE.N.SFH.05.Gen.ReEx.001.001`. Takes `a_ref` (`A_C_Ref`), `buildingYear` (`Year1_Building`) and `n_apartments` from that row. **Short-circuits everything else in this table** — see [§7](#7-parameters-that-do-less-than-their-name-suggests). |
| `country` | `AT BE BG CY CZ DE DK ES FR GB GR HU IE IT NL NO PL RS SE SI XX` | `'DE'` | Query criterion against `Code_Country`. Selects the national typology, which is what determines construction practice and therefore U-values. |
| `buildingYear` | int | `1990` | Construction year. Query criterion against the `Year1_Building`–`Year2_Building` band, i.e. the insulation standard of its era. Also feeds the `T_sup` derivation. |
| `buildingType` | `AB` `SFH` `MFH` `TH` | — | Apartment block / single-family / multi-family / terraced house. Query criterion against `Code_BuildingSizeClass`; drives size and surface-to-volume ratio. |
| `surrounding` | `Detached` `Semi` `Terraced` | — | How many neighbours the building is attached to (`Code_AttachedNeighbours`: `B_Alone`/`B_N1`/`B_N2`). Fewer exposed walls means less transmission loss. |
| `a_ref` | float | — | Total heated living area [m²]. Used both as a query criterion and as the **scaling target**: `get_shape` rescales the archetype's areas by `ratio = a_ref / A_C_Ref` — walls, doors and vertical windows by `√ratio`, roof and floor by `1 + (ratio-1)/n_Storey`, horizontal windows by `ratio/n_Storey`. |
| `a_ref_app` | float | — | Living area of a single flat. Only used as `a_ref_app × n_apartments` when `a_ref` is not given. |
| `n_apartments` | int | — | Number of flats. Multiplies the household occupancy simulation (one tsorb profile per flat) and feeds the `T_sup` and `hasFirePlace` derivations. Inherited from the archetype if omitted. |
| `refurbished` | bool | `False` | **Not refurbishment optimization** — a query-year shifter. Shifts the searched year to `min(max(year+40, 1995), 2020)` so the query lands on a better-insulated archetype, and warns that it did. |
| `buildnew` | bool | `False` | Forces the query year to 2020, i.e. new-build construction standard. |

## 2. Geometry

Everything else about the shape (wall/roof/floor areas, window areas per orientation, shading
factors `F_sh_vert`/`F_sh_hor`/`F_f`/`F_w`, room height, air exchange rates) is derived from the
archetype row and is not user-settable.

| Parameter | Type | Default | What it does |
|---|---|---|---|
| `roofTilt` | float | `0.0` for flat-roof archetypes (`Code_RoofType == "FR"`), else `45.0` | Roof pitch [°]. Drives the plane-of-array irradiance for PV and solar thermal (`@pv_yield`), and selects the roof handling in `_get_renewable_profile`. |
| `roofOrientation` | float | `135.0` | Roof azimuth [°], 180 = south. Same effect: it decides how much sun the roof surfaces see and therefore the specific PV yield. |

## 3. Weather and location

| Parameter | Type | Default | What it does |
|---|---|---|---|
| `latitude` | float | `50.0` | Site latitude [°]. Selects the DWD test reference year region and drives the solar position calculation. |
| `longitude` | float | `8.0` | Site longitude [°]. Same, and also feeds the default occupancy seed. |
| `year` | int | `2010` | Calendar year the TRY is mapped onto. Sets the time index, and with it weekday/weekend patterns in the occupancy simulation. |
| `weatherData` | `pandas.DataFrame` | — | Your own weather instead of a TRY. Must contain columns `DHI`, `DNI`, `T`. `design_T_min` is then taken as the minimum of `T`. **Requires `weatherID`.** |
| `weatherID` | str | — | Identifier of that weather data. Mandatory companion of `weatherData`, because it goes into `IDentries` and therefore into the result cache key. |

Without `weatherData`, `getISO12831weather(longitude, latitude, year)` loads the matching DWD
Testreferenzjahr and derives the ISO 12831 design temperature. Passing `weatherData` while
latitude/longitude are still at their defaults raises a warning — the weather would then be
yours but the solar geometry would still be that of the default site.

## 4. Comfort and control

These define the comfort band, which is what makes the building *flexible*: `T_air` is a free
decision variable inside the band, so thermal mass can be charged when energy is cheap.

| Parameter | Type | Default | What it does |
|---|---|---|---|
| `comfortT_lb` | float | `21.0` | Lower comfort bound [°C]. With no smart thermostat this is where `T_air` is pinned. Warns if `comfortT_lb >= comfortT_ub - 0.5`, which makes the LP numerically nasty. |
| `comfortT_ub` | float | `24.0` | Upper comfort bound [°C]. A **hard** ceiling, so summer gains above it force cooling even in a building with no cooling device — see `model-deviations.md` item 4. |
| `capControl` | bool | `True` | Smart thermostat. **This is the flexibility switch.** At `False` the band collapses to `comfortT_lb` and the building has no storage value at all; at `True` it opens to the full `comfortT_lb … comfortT_ub`. |
| `nightReduction` | bool | `True` | Night setback. Lets the lower bound drop toward 18 °C in proportion to the share of occupants asleep. |
| `occControl` | bool | `False` | Occupancy-driven control. Lets the band widen toward 14 °C (lower) and 30 °C (upper) in proportion to the share of occupants away from home. |
| `ventControl` | bool | `False` | Controllable ventilation. **`True` raises `NotImplementedError`** in `ThermalZone5R1C.__init__` — the path was never validated and was not ported to solph. |

The three control flags enter the comfort constraints directly as 1/0 coefficients
(`ThermalZone5R1C.control`), so they are configuration, not decisions.

## 5. Occupancy and loads

Occupants matter twice: they are an internal heat gain (`Q_ig`), and they are the electricity
demand (`@elecLoad`). Both come from the stochastic CREST-derived model in `tsorb`.

| Parameter | Type | Default | What it does |
|---|---|---|---|
| `n_persons` | int | `2` | Occupants **per flat**. Scales internal gains and the appliance/lighting load, and feeds the default seed. |
| `elecLoad` | `pandas.Series` | — | A measured or otherwise external load profile for a single flat, bypassing the tsorb simulation. Sets `tsorb_device_load = False`. **Requires `elecLoadID`.** |
| `elecLoadID` | str | — | Identifier of that profile for the cache key. Otherwise the key is `"CREST_<n_persons>x<n_apartments>"`. |
| `varyoccupancy` | int | `1` | How many independent occupancy realizations to simulate, for studies that need a spread rather than one draw. Values `< 1` raise `ValueError`. |
| `mean_load` | bool | `False` | Use the mean hourly profile instead of the fluctuating minute-resolution one. Smoother, but it removes the peaks that size a battery. |
| `seed` | int | derived | Overrides the derived seed to get an independent stochastic realization of the *same* building; physical parameters are untouched. The default seed is built from `n_persons` + `longitude` + `A_ref`, truncated to 8 digits — so two identical buildings reproduce each other, by design. |

**The occupancy simulation runs on the first `optimize()` / `getLoad()` call** and takes minutes,
not seconds. It is what makes a cold first run slow.

## 6. Equipment and hot water

> **What is *not* here any more.** `existingHeatSupply`, `replaceHeatSupply`, `hasPhotovoltaic`,
> `hasSolarThermal`, `costdata`, `ownership` and `WACC` used to live in this section. They only
> ever fed the cache key — no code in `tsib/optimization/` read any of them — and the equipment
> sheet ([§9](#9-the-equipment-sheet)) now answers the question they pretended to answer. They
> were removed rather than kept as inert metadata. Interest rates live on the spec, per
> component (`wacc`, [§8](#8-spec--preset-parameters)).

| Parameter | Type | Default | What it does |
|---|---|---|---|
| `T_sup` | float | derived | Design supply temperature of the heat distribution system [°C]. **This one does matter**: it sets the heat pump COP profile (`@cop`, via `simHeatpump`) — a lower `T_sup` is a better COP. Also sets `T_ret = T_sup - 20`. Derivation: base 70, `+5` if `n_apartments > 6`, `-10` each for `buildingYear` above 1990 / 2000 / 2010, or a flat `40.0` if `floorHeating`. |
| `floorHeating` | bool | `False` | Underfloor heating, i.e. a low-temperature system. Only consulted when `T_sup` is *not* given explicitly, in which case it pins `T_sup = 40.0`. |
| `hotWaterElec` | bool | `False` | Hot water is provided electrically rather than by the heating system. Scales `hotWaterLoad` by `0.6` (BDEW correction), applied where the profile is created. |
| `hasFirePlace` | bool | derived | Wood stove present. Derived as `True` if `A_ref / n_apartments >= 100`, which then also sets `fireplaceSize = 10 kW × n_apartments`. Consumed by `simFireplace`, not by the optimization. |

## 7. Parameters that do less than their name suggests

- **`ID` short-circuits the archetype query.** `_get_typ_building` returns early, so `country`,
  `buildingType`, `surrounding`, `a_ref`, `a_ref_app`, `n_apartments`, `refurbished` and
  `buildnew` are never consumed. Non-default values then trigger the "not used" warning. If you
  pass an `ID`, you get that archetype's area and year — not the ones you asked for.
- **`refurbished` is a query shifter**, not a physical refurbishment. See §1.
- **`eastOrOverall`** (`'N'` / `'East'`) validates but is **never read anywhere in the
  codebase**. It is dead.
- **Declared but unusable**, all raising `NotImplementedError`: `n_storey`, `a_roof`,
  `thermalClass` (the value `'medium'` is used, but passing it explicitly raises).
- **Removed** (they now raise `ValueError` as unknown kwargs): `refurbishment`,
  `force_refurbishment`, `onlyEnergyInvest`, `windows_refurbished`, `walls_refurbished`,
  `roof_refurbished`. Envelope/control refurbishment optimization lives on the `refurbishment`
  branch.

## 8. Spec / preset parameters

The spec decides the topology. Time series are `"@key"` references resolved against the
building configuration, which is what keeps specs serializable and reusable across buildings.

### `presets.heat_load_only(heat_cost=0.08, cool_cost=0.02, **zone_kwargs)`

Pure heat load simulation: the thermal zone plus a priced heat and cool source. The absolute
price level is irrelevant to the resulting load as long as both are positive — they exist only
so the objective has something to minimize.

### `presets.hp_pv_battery(...)`

| Parameter | Default | What it does |
|---|---|---|
| `import_price` | `"@elecPrice"` | Grid import tariff [EUR/kWh], scalar or profile. Falls back to `DEFAULT_ELEC_PRICE = 0.35` — see the gotcha below. |
| `export_price` | `None` | Feed-in revenue. When `None` the export sink is omitted entirely, so surplus PV is curtailed rather than sold. |
| `cop` | `"@cop"` | Heat pump COP profile from `simHeatpump`. It is `0` below −20 °C, and those hours are handled by forcing heat output to zero, not by dividing by zero. |
| `specific_yield` | `"@pv_yield"` | PV yield [kW/kWp] from `simPhotovoltaic`. |
| `elec_demand` | `"@elecLoad"` | Household electricity demand from tsorb. |
| `pv_kwp`, `battery_kwh`, `hp_kw`, `buffer_kwh` | `None` | A **number** fixes the capacity; a **dict** (`capex_per_unit`, `lifetime`, `max_capacity`) makes it an investment decision; `None` omits the component — except the heat pump, which is always built. |
| `cool_cost` | `0.02` | Price of the cooling source [EUR/kWh]. |
| `wacc` | `0.06` | Interest rate used to annualize the investments in this spec. An assumption of the scenario, not a property of the building. |

`zone_kwargs` are forwarded to `ThermalZone5R1C`: `max_load` (defaults to the calculated design
heat load), `max_load_violation_penalty` (`100.0` EUR/kW), and `initial_T_m` (replaces the
periodic wrap of the mass node).

### `Building.optimize(spec, solver=None, tee=False, solverOpts=None)`

`solver` defaults to `$SOLVER`, then auto-detects gurobi → cplex → scip → cbc → highs.

## 9. The equipment sheet

`BuildingSystemParameters` (`tsib/optimization/parameterization.py`) says **what a building
has**, without saying how it is wired. `build_spec()` turns it into a `SystemSpec` using the
one template every tsib building shares, so a few hundred buildings of a synthetic grid are
described by parameters instead of by hand-authored topologies.

```python
from tsib.optimization import BuildingSystemParameters, build_spec

params = BuildingSystemParameters(
    equipment={
        "heat_pump": {"capacity_kw": 8.0},
        "pv":        {"kwp": 8.0},
        "battery":   {"capacity_kwh": 10.0, "power_kw": 3.0},
        "buffer":    {"capacity_kwh": 20.0},
    },
    tariff={"import": "@elecPrice", "export": 0.08},
    meta={"bus_id": "pylovo-42", "freq": "h"},
)
results = bdg.optimize(build_spec(params))
```

Runnable: [`examples/energysystem/parameterization_demo.py`](../examples/energysystem/parameterization_demo.py).

Equipment which is absent or `None` is **not built**. An unknown equipment key, or an unknown
parameter inside one, raises — a typo must not silently size nothing.

| equipment | parameters | becomes |
|---|---|---|
| `heat_pump` | `capacity_kw`, `cop` (default `"@cop"`), `wacc` | `heat_pump` between the `elec` and `heat` buses |
| `pv` | `profile` **or** `kwp`, `specific_yield`, `curtailable`, `wacc` | `pv` on the `elec` bus |
| `battery` | `capacity_kwh`, `power_kw`, `roundtrip_efficiency`, and the `_storage` passthroughs | `battery` on the `elec` bus |
| `buffer` | `capacity_kwh`, `power_kw`, `standby_loss_kW`, same passthroughs | `thermal_storage` on the `heat` bus |

Any `capacity_*` accepts a number (fixed) or an investment dict, exactly as the presets do —
`presets.capacity_params` is shared between the two.

**PV has two forms on purpose.** `kwp` against a specific yield is the tsib-internal form and
keeps PV sizeable as an investment. A pre-computed absolute `profile` [kW] is the form
*exchanged* with other models: it removes any chance of a second engine recomputing a different
yield from the same kWp. Given a `profile`, a `kwp` alongside it is provenance only and does not
scale it again.

**`meta` is descriptive and never reaches the model** — in particular the grid connection point.
Two identical archetypes at different Pylovo buses are the same building and must share a cache
entry. `meta["freq"]` records the resolution the sheet was written for; `check_index()` compares
it against a time index so an hourly building cannot drift into a 15 minute study unnoticed.

### What the sheet does not contain

Topology. Each engine keeps its own predefined building network, and only the equipment sheet
plus the input time series travel between them. tsib's template puts **space heating and hot
water on one `heat` bus** with a single `buffer`, so the two are not distinguishable by
temperature level; a model that separates them consumes the same `hotWaterLoad` series and
splits it itself.

A building with no heat-supplying equipment gets a priced non-electric `heat_supply` source
instead of an infeasible model — a gas-heated building still belongs in a grid study, it just
contributes household load, PV and battery rather than heat pump electricity.

### Adding new equipment

One registered builder, nothing else:

```python
from tsib.optimization import equipment

@equipment("ev_charger")
def _ev_charger(name, params, spec):
    spec.add_component(name, "battery", bus_in="elec", bus_out="elec",
                       capacity=params["capacity_kwh"])
```

The exchange format, the template and the solver are all unchanged by this.

### `required_inputs`

`required_inputs(spec)` returns every configuration key a spec will look up — the `"@…"`
references, resolved. It is what makes the contract between the building configuration and the
system inspectable *before* building, and it is what lets `Building.optimize` simulate only the
profiles a given spec actually reads (a `heat_load_only` run no longer pays for a PV
simulation).

```python
>>> params.required_inputs()
['cop', 'elecLoad', 'elecPrice', 'hotWaterLoad', 'pv_yield']
```

## 10. Two gotchas

**`elecPrice` is not a `BuildingConfiguration` kwarg.** Passing it raises `ValueError`.
`_optimization_config` falls back to a flat 0.35 EUR/kWh. To use a real tariff, either pass the
array into the preset (`presets.hp_pv_battery(import_price=my_array)`) or set
`bdg.cfg["elecPrice"] = ...` after constructing the `Building`.

**Fixed capacities can make the model infeasible.** A heat pump smaller than the design heat
load, combined with the hours where the COP is zero, has no feasible dispatch, and HiGHS reports
`A feasible solution was not found`. Check `bdg.zone_config.calcDesignHeatLoad()` before fixing
`hp_kw`, or pass an investment dict and let it size itself.

## Where the values come from

```
BuildingConfiguration(kwargs)
  ├── _get_typ_building   archetype row  (ID, or query by country/type/year/area)
  ├── _get_form           get_shape(): areas scaled to a_ref, window areas, shading factors
  ├── _get_fabric         get_fabric(): U-values, g_gl, air exchange, thermal class
  ├── _get_operation      weather, comfort band, control flags, occupancy, seed
  └── _get_equipment      hot water, T_sup, fireplace flag
        -> cfg (the full parameter dict) + IDentries (the cache key)
```

`IDentries` is what the TinyDB result cache in `tsib/data/results/db.json` keys on. Changing any
parameter that appears there gives the building a new identity and forces a recomputation.
