# -*- coding: utf-8 -*-
"""
Solver handling for the energy system MILP: option management per solver
and auto-detection of an available solver. Ported unchanged from
tsib.thermal.utils.manageSolverOpts and Building5R1C.sim5R1C.
"""

import os

import pyomo.opt as opt
from pyomo.contrib import appsi


def manageSolverOpts(solver, solverOpts):
    """
    Adds solver specific options.

    Parameters
    ----------
    solver: str, required
        Solver name used by pyomo
    solverOpts: dict, required
        Other solveroptions

    Returns
    -------
    solverOpts: dict
    """

    defaultOpts = {}

    defaultOpts_gurobi = {
        "Threads": 3,
        "OptimalityTol": 1e-8,
        "Method": 2,  # interior point/barrier
        "Crossover": 0,  # skip crossover: it's numerically unstable on this model
        "Cuts": 0,  # no precut of solution spce
        "NodeMethod": 2,  # interior points
        "IntFeasTol": 1e-9,  # small values in order to avoid errors with BigM
    }

    # note: scip needs a param file, which is located in the dir of he model - so, no direct options can be given
    defaultOpts_scip = {}

    defaultOpts_cbc = {"primalT": 1e-3}

    defaultOpts_glpk = {}

    defaultOpts_cplex = {
        "threads": 3,
        "lp_method": 4,  # -> "lp method 4": force barrier
        "barrier_crossover_algorithm": -1,  # -> "barrier crossover algorithm -1": skip crossover, same numerical-stability fix as gurobi/highs
    }

    defaultOpts_highs = {}

    # append default options
    if solver == "gurobi":
        defaultOpts.update(defaultOpts_gurobi)
    elif solver == "scip":
        # note: scip needs a param file, which is located in the dir of he model - so, no direct options can be given
        defaultOpts.update(defaultOpts_scip)
    elif solver == "cplex":
        defaultOpts.update(defaultOpts_cplex)
    elif solver == "cbc":
        defaultOpts.update(defaultOpts_cbc)
    elif solver == "glpk":
        defaultOpts.update(defaultOpts_glpk)
        if "Threads" in solverOpts:
            solverOpts.pop("Threads")
        if "LogFile" in solverOpts:
            solverOpts.pop("LogFile")
    elif solver == "highs":
        # highs is solved separately via appsi.solvers.Highs(); these
        # options end up unused, but the solver name still needs to
        # pass validation here
        defaultOpts.update(defaultOpts_highs)
    else:
        raise ValueError(
            'Solver name unknown. Please use one of "gurobi", "scip", "cbc", "glpk", "cplex" or "highs".'
        )

    # just add default options if not defined in solverOpts
    for option in defaultOpts:
        if not option in solverOpts:
            solverOpts[option] = defaultOpts[option]

    return solverOpts


def detect_solver():
    """
    Determines the MILP solver to use: first the $SOLVER environment
    variable, otherwise the first available solver in performance
    priorization, with HiGHS last as the free, open-source fallback.

    Returns
    -------
    solver name as str
    """
    try:
        return os.environ["SOLVER"]
    except KeyError:
        DEFAULT_SOLVERS = ["gurobi", "cplex", "scip", "cbc", "highs"]
        for potential_solver in DEFAULT_SOLVERS:
            if opt.SolverFactory(potential_solver).available():
                return potential_solver
    raise LookupError(
        "No MILP solver found. Recommended: install the free, open-source "
        "HiGHS solver via `pip install tsib[highs]` (or `uv sync --extra highs` "
        "in this repo) - no license required. Alternatively, install a "
        "commercial solver (`pip install tsib[gurobi]`/`uv sync --extra gurobi`, "
        "which needs a Gurobi license) or install cplex/scip/cbc separately "
        "(e.g. `apt install coinor-cbc`), and declare it with the environment "
        "variable 'SOLVER' if it isn't auto-detected."
    )


def solve_model(pyomo_model, solver=None, tee=False, solverOpts=None):
    """
    Solves a pyomo model with the given or an auto-detected solver.

    Parameters
    ----------
    pyomo_model: pyomo.ConcreteModel, required
    solver: str, optional (default: $SOLVER or auto-detected)
    tee: bool, optional (default: False)
        Stream the solver log.
    solverOpts: dict, optional
        Additional solver options.

    Returns
    -------
    The solver results object.
    """
    if solver is None:
        solver = detect_solver()

    if solver == "glpk":
        raise ValueError(
            "Solver 'glpk' fails for the 5R1C optimization, although it is a MILP solver"
        )

    opts = manageSolverOpts(solver, dict(solverOpts) if solverOpts else {"Threads": 1, "LogFile": ""})

    if solver == "highs":
        highs = appsi.solvers.Highs()
        highs.config.stream_solver = tee
        highs_options = {
            "solver": "ipm",
            "simplex_scale_strategy": "off",
            "run_crossover": "off",
        }
        # only explicitly passed options reach HiGHS (the generic
        # Threads/LogFile defaults are not valid HiGHS options)
        if solverOpts:
            highs_options.update(solverOpts)
        highs.highs_options = highs_options
        results = highs.solve(pyomo_model)
    else:
        optprob = opt.SolverFactory(solver)
        optprob.options = opts
        results = optprob.solve(pyomo_model, tee=tee)

    return results
