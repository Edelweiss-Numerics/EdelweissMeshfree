# -*- coding: utf-8 -*-
#  ---------------------------------------------------------------------
#
#  _____    _      _              _
# | ____|__| | ___| |_      _____(_)___ ___
# |  _| / _` |/ _ \ \ \ /\ / / _ \ / __/ __|
# | |__| (_| |  __/ |\ V  V /  __/ \__ \__ \
# |_____\__,_|\___|_| \_/\_/_\___|_|___/___/
# |  \/  | ___  ___| |__  / _|_ __ ___  ___
# | |\/| |/ _ \/ __| '_ \| |_| '__/ _ \/ _ \
# | |  | |  __/\__ \ | | |  _| | |  __/  __/
# |_|  |_|\___||___/_| |_|_| |_|  \___|\___|
#
#
#  Unit of Strength of Materials and Structural Analysis
#  University of Innsbruck,
#
#  Research Group for Computational Mechanics of Materials
#  Institute of Structural Engineering, BOKU University, Vienna
#
#  2023 - today
#
#  This file is part of EdelweissMeshfree.
#
#  This library is free software; you can redistribute it and/or
#  modify it under the terms of the GNU Lesser General Public
#  License as published by the Free Software Foundation; either
#  version 2.1 of the License, or (at your option) any later version.
#
#  The full text of the license can be found in the file LICENSE.md at
#  the top level directory of EdelweissMeshfree.
#  ---------------------------------------------------------------------
"""
Test for the pure Python cell and material point implementation with linear elastic behavior.

This test performs simulations on a 2D rectangular domain [0, 200] x [0, 100] with 4x2 cells
and 8x4 material points using:
- PythonCell (bilinear Quad4) as the grid cell provider
- PythonMaterialPoint with linear elastic plane strain material
- NonlinearQuasistaticSolver for solving (10 equal time increments)

Four scenarios are tested, each with different Dirichlet boundary conditions:
1. Uniaxial compression: left fixed, right displaced by -10 in x
2. Vertical compression: bottom fixed, top displaced by -5 in y
3. Partial fixity: u=0 at left, v=0 at bottom, u=-10 at right
4. Shear: bottom fully fixed, top displaced by +10 in x
"""

import argparse

import edelweissfe.utils.performancetiming as performancetiming
import numpy as np
import pytest
from edelweissfe.journal.journal import Journal
from edelweissfe.linsolve.pardiso.pardiso import pardisoSolve
from edelweissfe.timesteppers.adaptivetimestepper import AdaptiveTimeStepper
from edelweissfe.utils.exceptions import StepFailed

from edelweissmeshfree.fieldoutput.fieldoutput import MPMFieldOutputController
from edelweissmeshfree.generators import (
    rectangulargridgenerator,
    rectangularmpgenerator,
)
from edelweissmeshfree.models.mpmmodel import MPMModel
from edelweissmeshfree.mpmmanagers.simplempmmanager import SimpleMaterialPointManager
from edelweissmeshfree.outputmanagers.ensight import (
    OutputManager as EnsightOutputManager,
)
from edelweissmeshfree.solvers.nqs import NonlinearQuasistaticSolver
from edelweissmeshfree.stepactions.dirichlet import Dirichlet


def run_sim(dirichlet_specs):
    """Run an MPM simulation with pure Python cells and material points using the nonlinear solver.

    Parameters
    ----------
    dirichlet_specs : list of (str, str, dict)
        Each entry is (name, node_set_key, values_dict) passed to Dirichlet.
        *values_dict* maps dof component index to prescribed value.
    """

    dimension = 2

    journal = Journal()

    mpmModel = MPMModel(dimension)

    rectangulargridgenerator.generateModelData(
        mpmModel,
        journal,
        x0=0.0,
        l=200.0,
        y0=0.0,
        h=100.0,
        nX=4,
        nY=2,
        cellProvider="PythonCell",
        cellType="Quad4",
    )

    linearElastic = {"material": "LinearElastic", "properties": np.array([30000.0, 0.3])}

    rectangularmpgenerator.generateModelData(
        mpmModel,
        journal,
        x0=12.5,
        l=175.0,
        y0=12.5,
        h=75.0,
        nX=8,
        nY=4,
        mpProvider="python",
        mpType="PlaneStrain",
        material=linearElastic,
    )

    mpmModel.prepareYourself(journal)
    mpmModel.nodeFields["displacement"].createFieldValueEntry("dU")

    allCells = mpmModel.cellSets["all"]
    allMPs = mpmModel.materialPointSets["all"]

    mpmManager = SimpleMaterialPointManager(allCells, allMPs)

    fieldOutputController = MPMFieldOutputController(mpmModel, journal)

    fieldOutputController.addPerMaterialPointFieldOutput(
        "displacement",
        allMPs,
        "displacement",
    )

    fieldOutputController.initializeJob()

    ensightOutput = EnsightOutputManager("ensight", mpmModel, fieldOutputController, journal, None)
    ensightOutput.updateDefinition(fieldOutput=fieldOutputController.fieldOutputs["displacement"], create="perNode")
    ensightOutput.initializeJob()

    dirichlets = [
        Dirichlet(name, mpmModel.nodeSets[node_set_key], "displacement", values, mpmModel, journal)
        for name, node_set_key, values in dirichlet_specs
    ]

    adaptiveTimeStepper = AdaptiveTimeStepper(0.0, 1.0, 0.1, 0.1, 1e-3, 1000, journal)

    nonlinearSolver = NonlinearQuasistaticSolver(journal)

    iterationOptions = {
        "max. iterations": 5,
        "critical iterations": 3,
        "allowed residual growths": 3,
    }

    linearSolver = pardisoSolve

    try:
        nonlinearSolver.solveStep(
            adaptiveTimeStepper,
            linearSolver,
            mpmModel,
            fieldOutputController,
            mpmManagers=[mpmManager],
            dirichlets=dirichlets,
            outputManagers=[ensightOutput],
            userIterationOptions=iterationOptions,
        )
    except StepFailed as e:
        journal.errorMessage(str(e), "StepFailed")
        raise

    finally:
        fieldOutputController.finalizeJob()
        ensightOutput.finalizeJob()

        prettytable = performancetiming.makePrettyTable()
        prettytable.min_table_width = journal.linewidth
        journal.printPrettyTable(prettytable, "Summary")

    return mpmModel


def run_sim_uniaxial():
    """Left fully fixed, right displaced by -10 in x."""
    return run_sim(
        [
            ("left", "rectangular_grid_left", {0: 0.0, 1: 0.0}),
            ("right", "rectangular_grid_right", {0: -10.0, 1: 0.0}),
        ]
    )


def run_sim_compression_y():
    """Bottom fully fixed, top displaced by -5 in y."""
    return run_sim(
        [
            ("bottom", "rectangular_grid_bottom", {0: 0.0, 1: 0.0}),
            ("top", "rectangular_grid_top", {0: 0.0, 1: -5.0}),
        ]
    )


def run_sim_partial_fixity():
    """Only u=0 at left, only v=0 at bottom, u=-10 at right."""
    return run_sim(
        [
            ("left", "rectangular_grid_left", {0: 0.0}),
            ("bottom", "rectangular_grid_bottom", {1: 0.0}),
            ("right", "rectangular_grid_right", {0: -10.0}),
        ]
    )


def run_sim_shear():
    """Bottom fully fixed, top displaced by +10 in x (shear loading)."""
    return run_sim(
        [
            ("bottom", "rectangular_grid_bottom", {0: 0.0, 1: 0.0}),
            ("top", "rectangular_grid_top", {0: 10.0, 1: 0.0}),
        ]
    )


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    """No matter where pytest is ran, we set the working dir
    to this testscript's parent directory"""

    monkeypatch.chdir(request.fspath.dirname)


def test_sim(assert_gold):
    """Uniaxial compression: left fixed, right displaced by -10 in x."""
    mpmModel = run_sim_uniaxial()
    res = np.array([mp.getResultArray("displacement") for mp in mpmModel.materialPoints.values()])
    assert_gold(res, np.loadtxt("gold.csv"))


def test_sim_compression_y(assert_gold):
    """Vertical compression: bottom fixed, top displaced by -5 in y."""
    mpmModel = run_sim_compression_y()
    res = np.array([mp.getResultArray("displacement") for mp in mpmModel.materialPoints.values()])
    assert_gold(res, np.loadtxt("gold_compression_y.csv"))


def test_sim_partial_fixity(assert_gold):
    """Partial fixity: u=0 at left, v=0 at bottom, u=-10 at right."""
    mpmModel = run_sim_partial_fixity()
    res = np.array([mp.getResultArray("displacement") for mp in mpmModel.materialPoints.values()])
    assert_gold(res, np.loadtxt("gold_partial_fixity.csv"))


def test_sim_shear(assert_gold):
    """Shear loading: bottom fully fixed, top displaced by +10 in x."""
    mpmModel = run_sim_shear()
    res = np.array([mp.getResultArray("displacement") for mp in mpmModel.materialPoints.values()])
    assert_gold(res, np.loadtxt("gold_shear.csv"))


_SCENARIOS = {
    "uniaxial": (run_sim_uniaxial, "gold.csv"),
    "compression_y": (run_sim_compression_y, "gold_compression_y.csv"),
    "partial_fixity": (run_sim_partial_fixity, "gold_partial_fixity.csv"),
    "shear": (run_sim_shear, "gold_shear.csv"),
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--create-gold", dest="create_gold", action="store_true", help="create the gold file(s).")
    parser.add_argument(
        "--scenario",
        choices=list(_SCENARIOS.keys()),
        default=None,
        help="which scenario to run (default: all)",
    )
    args = parser.parse_args()

    scenarios = [args.scenario] if args.scenario else list(_SCENARIOS.keys())

    for name in scenarios:
        run_fn, gold_file = _SCENARIOS[name]
        mpmModel = run_fn()
        if args.create_gold:
            gold = np.array([mp.getResultArray("displacement") for mp in mpmModel.materialPoints.values()])
            np.savetxt(gold_file, gold)
            print(f"Wrote {gold_file}")
