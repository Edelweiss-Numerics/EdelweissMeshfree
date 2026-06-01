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

Uniaxial compression: left fully fixed, right displaced by -10 in x.
Domain [0, 200] x [0, 100] with 4x2 cells and 8x4 material points using:
- PythonCell (bilinear Quad4) as the grid cell provider
- PythonMaterialPoint with linear elastic plane strain material
- NonlinearQuasistaticSolver (10 equal time increments)
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


def run_sim():
    """Left fully fixed, right displaced by -10 in x."""

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
        nX=20,
        nY=10,
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
        Dirichlet(
            "left", mpmModel.nodeSets["rectangular_grid_left"], "displacement", {0: 0.0, 1: 0.0}, mpmModel, journal
        ),
        Dirichlet(
            "right", mpmModel.nodeSets["rectangular_grid_right"], "displacement", {0: -10.0, 1: 0.0}, mpmModel, journal
        ),
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


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    """No matter where pytest is ran, we set the working dir
    to this testscript's parent directory"""

    monkeypatch.chdir(request.fspath.dirname)


def test_sim(assert_gold):
    """Uniaxial compression: left fixed, right displaced by -10 in x."""
    mpmModel = run_sim()
    ordered_mps = [mp for _, mp in sorted(mpmModel.materialPoints.items())]
    res = np.array([mp.getResultArray("displacement") for mp in ordered_mps])
    assert_gold(res, np.loadtxt("gold.csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--create-gold", dest="create_gold", action="store_true", help="create the gold file.")
    args = parser.parse_args()

    mpmModel = run_sim()
    if args.create_gold:
        ordered_mps = [mp for _, mp in sorted(mpmModel.materialPoints.items())]
        gold = np.array([mp.getResultArray("displacement") for mp in ordered_mps])
        np.savetxt("gold.csv", gold)
        print("Wrote gold.csv")
