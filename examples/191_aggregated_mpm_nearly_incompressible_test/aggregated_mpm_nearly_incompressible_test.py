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
"""Volumetric locking of implicit quasi-static MPM with and without aggregation (AgMPM), Displacement/Quad4 versus
DisplacementFBar/Quad4 cells.

A plane strain cantilever (length 10, depth 1, E = 12e3, compressible NeoHooke with nu = 0.4999),
clamped on the left edge of the grid, on a background grid of Quad4 cells of size 0.25 aligned with the beam, 2 x 2
material points per cell, under a small dead tip load (geometrically linear: deflection / length < 1e-3).
Reference: Timoshenko beam theory in plane strain (E' = E / (1 - nu^2), Cowper's shear coefficient), which a fine grid
(h = 0.0625) of DisplacementFBar cells reproduces within 2 %.

Checks: the standard cell locks (deflection < 10 % of the reference); the F-bar cell without aggregation does not
(within 3 %); with aggregation, the F-bar cell does not lock either (within 20 %), but only because the cells of an
aggregate evaluate the F-bar dilatation at one common point (AggregatedMaterialPointMethod.commonProjectionPoint,
the centroid of the material points of the aggregate); without it, it locks like the standard cell; the material
point displacements match the gold file of each of the four variants.

Why the common point: an aggregated boundary cell carries the polynomial extension of its interior cell (its exterior
nodes are slaved to it), but would still impose its own volumetric constraint at the centre of its own material
points, so the aggregate carries two volumetric constraints on (essentially) one bilinear field and loses its bending
mode (tip deflection 0.014 x reference). With the common point: 1.17 x reference at h = 0.25 (here the beam is four
cells deep, two of which are aggregated, so the volumetric constraint is coarser than the grid) and 0.98 x reference
at h = 0.125.
"""

import argparse

import numpy as np
import pytest
from edelweissfe.journal.journal import Journal
from edelweissfe.solvers.nonlinearimplicitstatic import NIST
from edelweissfe.stepactions.dirichlet import StepAction as Dirichlet
from edelweissfe.steps.adaptivestep import AdaptiveStep
from edelweissfe.steps.stepmanager import StepActionCollection

from edelweissmeshfree.constraints.aggregatedmpm import AggregatedMaterialPointMethod
from edelweissmeshfree.fieldoutput.fieldoutput import MPMFieldOutputController
from edelweissmeshfree.config.mplibrary import getMaterialPointClass
from edelweissmeshfree.generators import rectangulargridgenerator
from edelweissmeshfree.models.mpmmodel import MPMModel
from edelweissmeshfree.mpmmanagers.smartmpmmanager import SmartMaterialPointManager
from edelweissmeshfree.sets.materialpointset import MaterialPointSet
from edelweissmeshfree.solvers.meshfreeconvergencecriterion import (
    jobInfoWithMeshfreeTolerances,
    meshfreeCriterionOptions,
)
from edelweissmeshfree.stepactions.distributedload import (
    MaterialPointPointWiseDistributedLoad,
)

E, NU, LENGTH, DEPTH, TIPLOAD = 12.0e3, 0.4999, 10.0, 1.0, 0.03
#: bulk and shear modulus, density
NEOHOOKE = {
    "material": "COMPRESSIBLENEOHOOKE",
    "properties": np.array([E / (3 * (1 - 2 * NU)), E / (2 * (1 + NU)), 1.0]),
}
VARIANTS = [
    ("Displacement/Quad4", False),
    ("Displacement/Quad4", True),
    ("DisplacementFBar/Quad4", False),
    ("DisplacementFBar/Quad4", True),
]
IDS = ["standard", "standard-aggregated", "fbar", "fbar-aggregated"]


def timoshenkoTipDeflection():
    """Plane strain Timoshenko cantilever under a tip load (per unit thickness), Cowper's shear coefficient."""
    planeStrainModulus, shearModulus = E / (1 - NU**2), E / (2 * (1 + NU))
    shearCoefficient = 10 * (1 + NU) / (12 + 11 * NU)
    bending = TIPLOAD * LENGTH**3 / (3 * planeStrainModulus * DEPTH**3 / 12)
    return bending + TIPLOAD * LENGTH / (shearCoefficient * shearModulus * DEPTH)


class DeadPointLoad(MaterialPointPointWiseDistributedLoad):
    """A dead point force carried by material points (the library's point-wise 'Pressure' load follows the rotation;
    here, at a geometrically linear load level, only the plain force matters)."""

    def __init__(self, name, model, journal, materialPoints, force):
        super().__init__(name, model, journal, materialPoints, "Pressure", np.asarray(force, dtype=float))

    def assembleLoad(self, P, K, U_np, dU, timeStep, locations):
        for mp in self.mpSet:
            _, force = self.getCurrentMaterialPointLoad(mp, timeStep)
            x = np.asarray(mp.getCenterCoordinates(), dtype=float).ravel()
            for cell in mp.assignedCells:
                N = np.asarray(cell.getInterpolationVector(x)).ravel()
                P[cell] += -np.outer(N, force).ravel()


def run_sim(cellType: str, aggregate: bool, commonProjectionPoint: bool = True):
    dimension = 2
    journal = Journal()

    mpmModel = MPMModel(dimension)

    # grid [0, 11] x [-1, 2] of 0.25 cells; the beam [0, 10] x [0, 1] fills 40 x 4 cells
    rectangulargridgenerator.generateModelData(
        mpmModel,
        journal,
        x0=0.0,
        l=11.0,
        y0=-1.0,
        h=3.0,
        nX=44,
        nY=12,
        cellProvider="LagrangianMarmotCell",
        cellType=cellType,
    )
    # 80 x 8 material points at the centres of a regular subdivision of the beam (2 x 2 per cell), volume = area / 640
    materialPointFactory = getMaterialPointClass("marmot")
    nX, nY = 80, 8
    dx, dy = LENGTH / nX, DEPTH / nY
    for i in range(nX):
        for j in range(nY):
            number = i * nY + j + 1
            xy = np.array([[(i + 0.5) * dx, (j + 0.5) * dy]])
            mpmModel.materialPoints[number] = materialPointFactory(
                "Displacement/PlaneStrain", number, xy, dx * dy, NEOHOOKE
            )
    mpmModel.materialPointSets["all"] = MaterialPointSet("all", list(mpmModel.materialPoints.values()))

    mpmModel.prepareYourself(journal)

    allCells = mpmModel.cellSets["all"]
    allMPs = mpmModel.materialPointSets["all"]

    mpmModel.mpmManagers.append(SmartMaterialPointManager(allCells, allMPs, dimension, options={"KDTreeLevels": 3}))

    if aggregate:
        mpmModel.multiPointConstraints["agmpm"] = AggregatedMaterialPointMethod(
            "agmpm", mpmModel, journal, commonProjectionPoint=commonProjectionPoint
        )

    fieldOutputController = MPMFieldOutputController(mpmModel, journal)
    fieldOutputController.addPerMaterialPointFieldOutput("displacement", allMPs, "displacement")
    fieldOutputController.initializeJob()

    # the two material points of the last column next to the neutral axis carry the tip load
    tip = sorted(
        [mp for mp in mpmModel.materialPoints.values() if mp.getCenterCoordinates()[0] > LENGTH - 0.1],
        key=lambda mp: abs(mp.getCenterCoordinates()[1] - DEPTH / 2),
    )[:2]

    clamp = Dirichlet(
        "clamp", mpmModel.nodeSets["rectangular_grid_left"], "displacement", {0: 0.0, 1: 0.0}, mpmModel, journal
    )
    tipLoad = DeadPointLoad("tipload", mpmModel, journal, tip, [0.0, -TIPLOAD / 2])

    stepActions = StepActionCollection()
    stepActions["dirichlet"][clamp.name] = clamp
    stepActions["distributedload"][tipLoad.name] = tipLoad

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
        startInc=1.0,
        maxInc=1.0,
        minInc=1e-2,
        maxNumInc=10,
        maxIter=10,
        criticalIter=5,
        maxGrowIter=3,
    )

    try:
        step.solve()
    finally:
        fieldOutputController.finalizeJob()

    return mpmModel, tip


def displacements(mpmModel):
    return np.array([mp.getResultArray("displacement") for mp in mpmModel.materialPoints.values()])


def tipDeflection(tip):
    return -np.mean([mp.getResultArray("displacement")[1] for mp in tip])


def goldFile(cellType: str, aggregate: bool):
    return "gold_{:}{:}.csv".format(cellType.split("/")[0].lower(), "_aggregated" if aggregate else "")


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    """No matter where pytest is ran, we set the working dir
    to this testscript's parent directory"""

    monkeypatch.chdir(request.fspath.dirname)


@pytest.mark.parametrize("cellType, aggregate", VARIANTS, ids=IDS)
def test_sim(assert_gold, cellType, aggregate):
    mpmModel, _ = run_sim(cellType, aggregate)

    assert_gold(displacements(mpmModel), np.loadtxt(goldFile(cellType, aggregate)), atol=1e-10)


def test_standard_cell_locks():
    _, tip = run_sim("Displacement/Quad4", False)
    assert tipDeflection(tip) < 0.1 * timoshenkoTipDeflection()


def test_fbar_cell_is_free_of_locking():
    _, tip = run_sim("DisplacementFBar/Quad4", False)
    assert tipDeflection(tip) == pytest.approx(timoshenkoTipDeflection(), rel=0.03)


def test_aggregated_fbar_cell_is_free_of_locking():
    _, tip = run_sim("DisplacementFBar/Quad4", True)
    assert tipDeflection(tip) == pytest.approx(timoshenkoTipDeflection(), rel=0.2)


def test_aggregated_fbar_cell_locks_without_common_projection_point():
    _, tip = run_sim("DisplacementFBar/Quad4", True, commonProjectionPoint=False)
    assert tipDeflection(tip) < 0.1 * timoshenkoTipDeflection()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--create-gold", dest="create_gold", action="store_true", help="create the gold files.")
    args = parser.parse_args()

    reference = timoshenkoTipDeflection()
    for (cellType, aggregate), name in zip(VARIANTS, IDS):
        mpmModel, tip = run_sim(cellType, aggregate)
        print(
            "{:20}: tip deflection {:.6e} = {:.3f} x Timoshenko".format(
                name, tipDeflection(tip), tipDeflection(tip) / reference
            )
        )
        if args.create_gold:
            np.savetxt(goldFile(cellType, aggregate), displacements(mpmModel))
