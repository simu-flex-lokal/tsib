# tsib documentation

| document | what it is | read it when |
|---|---|---|
| [`energysystem.md`](energysystem.md) | The energy system framework: minimal example, architecture, the full MILP formulation, a step-by-step tutorial on writing a new component, and gotchas. | You are using or extending `tsib.energysystem`. |
| [`model-deviations.md`](model-deviations.md) | Register of four known deviations of the 5R1C thermal zone from its source publications. All deliberately **unfixed** to preserve parity with the pre-refactor model. | **Before changing any equation in `thermalzone5r1c.py`**, or if a result looks physically wrong. |
| [`open-problem-summer-overheating.md`](open-problem-summer-overheating.md) | Self-contained problem statement for the fictitious-cooling issue (deviation #4), written for readers who do not know this codebase. Includes evidence, candidate formulation, and open research questions. | You want to solve that problem, or hand it to someone who might. |

Runnable demonstrations live in [`../examples/energysystem/`](../examples/energysystem/):
`EnergySystemDemo.ipynb` (flexibility, PV+battery investment, envelope refurbishment, and a
full-year whole-building workflow) and `heatpump_component.py` (the component from the tutorial).
