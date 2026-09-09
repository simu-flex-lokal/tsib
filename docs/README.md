# tsib documentation

| document | what it is | read it when |
|---|---|---|
| [`building-system.md`](building-system.md) | The three descriptions of a building, the spec tsib emits, and the ten scalars plus five series that are the thermal zone's whole contract with an energy system model. | You are handing a building to a model, or wondering where the optimization went. |
| [`parameters.md`](parameters.md) | Every `BuildingConfiguration` kwarg and equipment sheet entry: type, default, how it is derived, and what it actually affects downstream. | You are parameterizing a building and want to know which knob does what — or why one you set had no effect. |
| [`plotting.md`](plotting.md) | `tsib.plotting`: the functions, the four ways to look at a year (window, resample, heatmap, load duration), and the sign conventions. | You are plotting initialized profiles or the thermal zone. |

This index lists **user-facing documentation only** — reference material for people using or
extending `tsib`. Design notes, open problems, and feature ideas live in the untracked
`../backlog/` directory (kept out of GitHub) and are not documentation.

Runnable: [`examples/buildingsystem/system_export_demo.py`](../examples/buildingsystem/system_export_demo.py)
takes one archetype from an ID to a spec plus its input series on disk. The energy system model
itself lives in [`esmkit`](https://github.com/simu-flex-lokal/esmkit), whose `docs/` carry the
solving notebooks; the two are driven end to end from the `orchestrator/` working directory.

The 5R1C zone is pinned twice: `test/test_envelope_contract.py` compares the parameters tsib
derives against a capture from the pre-split model, and the golden fixtures in
[`../test/data/golden/`](../test/data/golden/) hold the solved heat load the forward model still
has to reproduce.
