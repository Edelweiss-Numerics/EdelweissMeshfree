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
#  Matthias Neuner |  matthias.neuner@boku.ac.at
#  Thomas Mader    |  thomas.mader@bokut.ac.at
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
"""Implicit quasi-static MPM with and without aggregation (AgMPM, Coombs, Bird and Pretti, CMAME 442 (2025) 118012).

A plane strain cantilever under its own weight, clamped on the left edge of the grid, on a background grid of unit
Displacement/Quad4 cells. The body does not align with the grid: the material points of its top and bottom rows, and
of its right end, lie in cells they cover by a small part only (small cuts), so the nodes of these cells are reached
through tiny basis function values. With aggregation (:class:`AggregatedMaterialPointMethod`), the exterior nodes of the
active domain take the values of interior cells extended to them.

Checks: aggregation is active and leaves the clamped degrees of freedom independent; it reduces the condition number of
the stiffness reduced to the free degrees of freedom by orders of magnitude; both runs converge to nearly the same tip
deflection; the material point displacements match the gold file of each run.
"""

import argparse

import numpy as np
import pytest
from edelweissfe.journal.journal import Journal
from edelweissfe.solvers.nonlinearimplicitstatic import NIST
from edelweissfe.stepactions.dirichlet import StepAction as Dirichlet
from edelweissfe.steps.adaptivestep import AdaptiveStep
from edelweissfe.steps.stepmanager import StepActionCollection
from edelweissfe.utils.exceptions import StepFailed

from edelweissmeshfree.constraints.aggregatedmpm import AggregatedMaterialPointMethod
from edelweissmeshfree.fieldoutput.fieldoutput import MPMFieldOutputController
from edelweissmeshfree.generators import (
    rectangulargridgenerator,
    rectangularmpgenerator,
)
from edelweissmeshfree.models.mpmmodel import MPMModel
from edelweissmeshfree.mpmmanagers.smartmpmmanager import SmartMaterialPointManager
from edelweissmeshfree.numerics.aggregation import reducedStiffness
from edelweissmeshfree.solvers.meshfreeconvergencecriterion import (
    jobInfoWithMeshfreeTolerances,
    meshfreeCriterionOptions,
)
from edelweissmeshfree.stepactions.bodyload import BodyLoad

#: E, nu, density
LINEAR_ELASTIC = {"material": "LINEARELASTIC/HUGHES-WINGET", "properties": np.array([1.0e4, 0.3, 1.0])}
GRAVITY = np.array([0.0, -0.1])


def run_sim(aggregate: bool):
    dimension = 2
    journal = Journal()

    mpmModel = MPMModel(dimension)

    rectangulargridgenerator.generateModelData(
        mpmModel,
        journal,
        x0=0.0,
        l=10.0,
        y0=-1.0,
        h=5.0,
        nX=10,
        nY=5,
        cellProvider="LagrangianMarmotCell",
        cellType="Displacement/Quad4",
    )
    # 33 x 12 material points at x = 0.125 ... 8.005, y = 0.995 ... 2.005: the rows y = 0.995 and y = 2.005, and the
    # column x = 8.005, lie 0.005 inside the cells below, above and right of the body's core (small cuts)
    rectangularmpgenerator.generateModelData(
        mpmModel,
        journal,
        x0=0.125,
        l=7.88,
        y0=0.995,
        h=1.01,
        nX=33,
        nY=12,
        mpProvider="marmot",
        mpType="Displacement/PlaneStrain",
        material=LINEAR_ELASTIC,
    )

    mpmModel.prepareYourself(journal)

    allCells = mpmModel.cellSets["all"]
    allMPs = mpmModel.materialPointSets["all"]

    mpmModel.mpmManagers.append(SmartMaterialPointManager(allCells, allMPs, dimension, options={"KDTreeLevels": 3}))

    if aggregate:
        mpmModel.multiPointConstraints["agmpm"] = AggregatedMaterialPointMethod("agmpm", mpmModel, journal)

    fieldOutputController = MPMFieldOutputController(mpmModel, journal)
    fieldOutputController.addPerMaterialPointFieldOutput("displacement", allMPs, "displacement")
    fieldOutputController.initializeJob()

    clamp = Dirichlet(
        "clamp", mpmModel.nodeSets["rectangular_grid_left"], "displacement", {0: 0.0, 1: 0.0}, mpmModel, journal
    )
    gravity = BodyLoad("gravity", mpmModel, journal, mpmModel.cells.values(), "BodyForce", GRAVITY)

    # the stiffness reduced to the free degrees of freedom, before the step
    stiffness = reducedStiffness(mpmModel, [clamp], aggregate=aggregate)

    stepActions = StepActionCollection()
    stepActions["dirichlet"][clamp.name] = clamp
    stepActions["bodyforce"][gravity.name] = gravity

    jobInfo = jobInfoWithMeshfreeTolerances()
    step = AdaptiveStep(
        0,
        mpmModel,
        fieldOutputController,
        journal,
        jobInfo,
        NIST(jobInfo, journal, linsolver="pardiso", **meshfreeCriterionOptions),
        [],
        stepActions,
        startInc=0.5,
        maxInc=0.5,
        minInc=1e-2,
        maxNumInc=20,
        maxIter=10,
        criticalIter=5,
        maxGrowIter=3,
    )

    try:
        step.solve()
    except StepFailed as e:
        journal.errorMessage(str(e), "StepFailed")
        raise
    finally:
        fieldOutputController.finalizeJob()

    return mpmModel, stiffness


def displacements(mpmModel):
    return np.array([mp.getResultArray("displacement") for mp in mpmModel.materialPoints.values()])


def tipDeflection(mpmModel):
    return displacements(mpmModel)[:, 1].min()


def goldFile(aggregate: bool):
    return "gold_aggregated.csv" if aggregate else "gold_standard.csv"


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    """No matter where pytest is ran, we set the working dir
    to this testscript's parent directory"""

    monkeypatch.chdir(request.fspath.dirname)


@pytest.mark.parametrize("aggregate", [False, True], ids=["standard", "aggregated"])
def test_sim(assert_gold, aggregate):
    mpmModel, stiffness = run_sim(aggregate)

    if aggregate:
        aggregation = stiffness.aggregation
        # at the start, 27 active cells: 8 interior (the core row), 19 boundary; the exterior nodes are those of the
        # rows y = 0 and y = 3 and of the column x = 9
        assert len(aggregation.interiorCells) == 8
        assert len(aggregation.boundaryCells) == 19
        assert len(aggregation.exteriorNodes) == 22
        # the clamped nodes (0, 0) and (0, 3) stay independent: 2 x (22 - 2) aggregated degrees of freedom
        assert stiffness.nExteriorDofs == 40
        assert stiffness.conditionNumber < 1e5
        # the deflection moves material points out of the cells of the top row near the tip: the solver aggregates
        # anew on the new active cells
        assert len(mpmModel.multiPointConstraints["agmpm"].aggregation.activeCells) == 22
    else:
        assert stiffness.conditionNumber > 1e6

    assert -0.06 < tipDeflection(mpmModel) < -0.02

    assert_gold(displacements(mpmModel), np.loadtxt(goldFile(aggregate)))


def test_aggregation_reduces_the_condition_number():
    """Fig. 7 of the paper, in short: aggregation removes the ill-conditioning of the small cuts."""

    _, standard = run_sim(False)
    _, aggregated = run_sim(True)
    assert aggregated.conditionNumber < 1e-2 * standard.conditionNumber


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--create-gold", dest="create_gold", action="store_true", help="create the gold files.")
    args = parser.parse_args()

    for aggregate in (False, True):
        mpmModel, stiffness = run_sim(aggregate)
        print(
            "aggregate={:}: condition number {:.3e} ({:} free dof, {:} aggregated), tip deflection {:.6e}".format(
                aggregate,
                stiffness.conditionNumber,
                len(stiffness.freeDofs),
                stiffness.nExteriorDofs,
                tipDeflection(mpmModel),
            )
        )
        if args.create_gold:
            np.savetxt(goldFile(aggregate), displacements(mpmModel))
