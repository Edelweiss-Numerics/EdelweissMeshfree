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
"""Implicit quasi-static MPM with ghost stabilisation (Coombs, IJNME 124 (2023) 4841-4875), alone and on top of the
aggregation (AgMPM).

The cantilever of example 190: a plane strain cantilever under its own weight, clamped on the left edge of the grid,
on a background grid of unit Displacement/Quad4 cells, with small cuts in its top and bottom rows and at its right end.
The ghost penalty (:class:`GhostStabilisation`, gamma = 0.01 E, see :data:`GHOST`) penalises the jumps of the
normal derivative of the displacement over the faces of the boundary cells.

Checks: the ghost faces of the initial active domain; ghost stabilisation reduces the condition number of the
stiffness reduced to the free degrees of freedom by orders of magnitude, alone and with aggregation; the runs converge
to nearly the deflection of the unstabilised run (example 190); the material point displacements match the gold file
of each run.
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
from edelweissmeshfree.constraints.ghoststabilisation import GhostStabilisation
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

E = 1.0e4
#: E, nu, density
LINEAR_ELASTIC = {"material": "LINEARELASTIC/HUGHES-WINGET", "properties": np.array([E, 0.3, 1.0])}
GRAVITY = np.array([0.0, -0.1])
#: the ghost penalty parameter of the displacement field. Coombs (Section 5.6) takes gamma_k = E, which on this coarse
#: grid (h = 1 = the depth of the beam, every cell of the top and bottom rows a boundary cell) stiffens the beam by a
#: factor 2.4: the penalty also acts on the kinks of the piecewise linear interpolant of the bending deflection, an
#: energy of order gamma h^3 / (E d^3) relative to the bending energy. 0.01 E already bounds the condition number.
GHOST = {"displacement": 0.01 * E}


def run_sim(aggregate: bool, ghost: bool = True):
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
    # as example 190: 33 x 12 material points with small cuts in the rows y = 0.995 and y = 2.005, and the column
    # x = 8.005
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
    if ghost:
        mpmModel.constraints["ghost"] = GhostStabilisation("ghost", mpmModel, GHOST, journal)

    fieldOutputController = MPMFieldOutputController(mpmModel, journal)
    fieldOutputController.addPerMaterialPointFieldOutput("displacement", allMPs, "displacement")
    fieldOutputController.initializeJob()

    clamp = Dirichlet(
        "clamp", mpmModel.nodeSets["rectangular_grid_left"], "displacement", {0: 0.0, 1: 0.0}, mpmModel, journal
    )
    gravity = BodyLoad("gravity", mpmModel, journal, mpmModel.cells.values(), "BodyForce", GRAVITY)

    # the stiffness reduced to the free degrees of freedom, before the step
    stiffness = reducedStiffness(mpmModel, [clamp], aggregate=aggregate, ghost=GHOST if ghost else None)

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
    return "gold_aggregated_ghost.csv" if aggregate else "gold_ghost.csv"


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    """No matter where pytest is ran, we set the working dir
    to this testscript's parent directory"""

    monkeypatch.chdir(request.fspath.dirname)


@pytest.mark.parametrize("aggregate", [False, True], ids=["ghost", "aggregated_ghost"])
def test_sim(assert_gold, aggregate):
    mpmModel, stiffness = run_sim(aggregate)

    ghost = mpmModel.constraints["ghost"]
    assert ghost.active
    # after the deflection (22 active cells, as in example 190): 18 boundary cells, 29 ghost faces
    assert len(ghost.boundaryCells) == 18
    assert len(ghost.faces) == 29
    # unstabilised: 1.05e7 (example 190)
    assert stiffness.conditionNumber < 1e5

    assert -0.06 < tipDeflection(mpmModel) < -0.02

    assert_gold(displacements(mpmModel), np.loadtxt(goldFile(aggregate)))


def test_ghost_reduces_the_condition_number():
    """Table 1 of the paper, in short: the ghost penalty removes the ill-conditioning of the small cuts, alone and on
    top of the aggregation; the deflection changes little."""

    standard, kStandard = run_sim(False, ghost=False)
    ghost, kGhost = run_sim(False)
    aggregatedGhost, kAggregatedGhost = run_sim(True)
    assert kGhost.conditionNumber < 1e-2 * kStandard.conditionNumber
    assert kAggregatedGhost.conditionNumber < 1e-2 * kStandard.conditionNumber
    assert abs(tipDeflection(ghost) / tipDeflection(standard) - 1) < 0.05


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--create-gold", dest="create_gold", action="store_true", help="create the gold files.")
    args = parser.parse_args()

    for aggregate, ghost in ((False, False), (True, False), (False, True), (True, True)):
        mpmModel, stiffness = run_sim(aggregate, ghost)
        g = mpmModel.constraints.get("ghost")
        print(
            "aggregate={:} ghost={:}: condition number {:.3e} ({:} free dof), tip deflection {:.6e}{:}".format(
                aggregate,
                ghost,
                stiffness.conditionNumber,
                len(stiffness.freeDofs),
                tipDeflection(mpmModel),
                "" if g is None else ", {:} boundary cells, {:} ghost faces".format(len(g.boundaryCells), len(g.faces)),
            )
        )
        if args.create_gold and ghost:
            np.savetxt(goldFile(aggregate), displacements(mpmModel))
