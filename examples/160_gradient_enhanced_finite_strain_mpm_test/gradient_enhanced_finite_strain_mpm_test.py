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
"""MPM with the gradient-enhanced finite-strain cell, material point and materials.

A plane strain block (20 x 4) of material points on a background grid of GradientEnhancedFiniteStrain/Quad4
cells is compressed by 2 % from the right, with two of Marmot's gradient-enhanced finite-strain materials:

- GradientEnhancedCompressibleNeoHookeDamage, loaded to twenty times its damage threshold;
- GradientEnhancedFiniteStrainDruckerPrager, which yields on the Drucker-Prager cone, with the dilatant plastic
  flow driving the implicit-gradient damage.

The grid is taller than the block, so that the lateral expansion keeps every material point inside it. Checks:
the imposed shortening is reached, the material-specific mechanism is active (damage / plastic flow), and the
material point displacements match the gold file of each material.

The boundary conditions (left edge fixed in x, one grid node fixed in y, free top and bottom) admit a homogeneous
solution, which the Quad4 grid represents exactly. For the Drucker-Prager material (with hardening and a
nonlocal damage that is not over-nonlocal, so that the homogeneous state stays stable) the test verifies it
against the single material point solution: a linear displacement field, a uniform nonlocal field, and the
lateral stretch of the homogeneous state. On the Eulerian grid each of the ten increments shortens the current
length by 0.2 %, so the axial stretch is 0.998^10.
"""

import argparse

import numpy as np
import pytest
from edelweissfe.config.linsolve import getLinSolverByName
from edelweissfe.journal.journal import Journal
from edelweissfe.timesteppers.adaptivetimestepper import AdaptiveTimeStepper
from edelweissfe.utils.exceptions import StepFailed

from edelweissmeshfree.fieldoutput.fieldoutput import MPMFieldOutputController
from edelweissmeshfree.generators import (
    rectangulargridgenerator,
    rectangularmpgenerator,
)
from edelweissmeshfree.models.mpmmodel import MPMModel
from edelweissmeshfree.mpmmanagers.smartmpmmanager import SmartMaterialPointManager
from edelweissmeshfree.solvers.nqs import NonlinearQuasistaticSolver
from edelweissmeshfree.stepactions.dirichlet import Dirichlet

KAPPA0 = 1e-3
SHORTENING = 0.4

MATERIALS = {
    # K, G, kappa0, kappaF, l, rho
    "neohooke": {
        "material": "GradientEnhancedCompressibleNeoHookeDamage",
        "properties": np.array([3500.0, 1500.0, KAPPA0, 1e-2, 2.0, 1.0]),
    },
    # K, G, c0, phi, psi, H, epsF, omegaMax, l, m, rho
    "druckerprager": {
        "material": "GradientEnhancedFiniteStrainDruckerPrager",
        "properties": np.array([3500.0, 1500.0, 5.0, 30.0, 10.0, 100.0, 0.02, 0.99, 2.0, 1.0, 1.0]),
    },
}


def run_sim(materialName):
    dimension = 2

    journal = Journal()

    mpmModel = MPMModel(dimension)

    rectangulargridgenerator.generateModelData(
        mpmModel,
        journal,
        x0=0.0,
        l=20.0,
        y0=-1.0,
        h=6.0,
        nX=10,
        nY=3,
        cellProvider="LagrangianMarmotCell",
        cellType="GradientEnhancedFiniteStrain/Quad4",
    )

    material = MATERIALS[materialName]

    rectangularmpgenerator.generateModelData(
        mpmModel,
        journal,
        x0=0.1,
        l=19.8,
        y0=0.1,
        h=3.8,
        nX=20,
        nY=8,
        mpProvider="marmot",
        mpType="GradientEnhancedFiniteStrain/PlaneStrain",
        material=material,
    )

    mpmModel.prepareYourself(journal)
    mpmModel.nodeFields["displacement"].createFieldValueEntry("dU")
    mpmModel.nodeFields["nonlocal damage"].createFieldValueEntry("dU")

    allCells = mpmModel.cellSets["all"]
    allMPs = mpmModel.materialPointSets["all"]

    mpmManager = SmartMaterialPointManager(allCells, allMPs, dimension, options={"KDTreeLevels": 3})

    fieldOutputController = MPMFieldOutputController(mpmModel, journal)
    fieldOutputController.addPerMaterialPointFieldOutput("displacement", allMPs, "displacement")
    fieldOutputController.addPerMaterialPointFieldOutput("nonlocal damage", allMPs, "nonlocal damage")
    fieldOutputController.initializeJob()

    dirichletLeft = Dirichlet(
        "left", mpmModel.nodeSets["rectangular_grid_left"], "displacement", {0: 0.0}, mpmModel, journal
    )
    dirichletFix = Dirichlet(
        "fix", mpmModel.nodeSets["rectangular_grid_leftBottom"], "displacement", {1: 0.0}, mpmModel, journal
    )
    dirichletRight = Dirichlet(
        "right", mpmModel.nodeSets["rectangular_grid_right"], "displacement", {0: -SHORTENING}, mpmModel, journal
    )

    adaptiveTimeStepper = AdaptiveTimeStepper(0.0, 1.0, 1e-1, 1e-1, 1e-3, 100, journal)

    nonlinearSolver = NonlinearQuasistaticSolver(journal)

    iterationOptions = {"max. iterations": 12, "critical iterations": 6, "allowed residual growths": 3}

    linearSolver = getLinSolverByName("pardiso", {})

    try:
        nonlinearSolver.solveStep(
            adaptiveTimeStepper,
            linearSolver,
            mpmModel,
            fieldOutputController,
            mpmManagers=[mpmManager],
            dirichlets=[dirichletLeft, dirichletFix, dirichletRight],
            userIterationOptions=iterationOptions,
        )
    except StepFailed as e:
        journal.errorMessage(str(e), "StepFailed")
        raise
    finally:
        fieldOutputController.finalizeJob()

    return mpmModel


def result(mpmModel, name):
    return np.array([mp.getResultArray(name) for mp in mpmModel.materialPoints.values()])


def goldFile(materialName):
    return "gold_" + materialName + ".csv"


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    """No matter where pytest is ran, we set the working dir
    to this testscript's parent directory"""

    monkeypatch.chdir(request.fspath.dirname)


# the homogeneous plane strain state after ten increments of 0.2 % shortening each, computed for a single material
# point of GradientEnhancedFiniteStrainDruckerPrager with the properties above (tau_yy = 0, nonlocal = local field)
HOMOGENEOUS_DRUCKERPRAGER = {"lateral stretch": 1.0164877609, "nonlocal field": 3.920030e-03, "alphaP": 2.210223e-02}


def checkHomogeneousDruckerPrager(mpmModel, u, n):
    x = np.array([mp.getCenterCoordinates()[:2] for mp in mpmModel.materialPoints.values()])
    X = x - u[:, :2]
    A = np.c_[X, np.ones(len(X))]
    for i in range(2):
        c = np.linalg.lstsq(A, u[:, i], rcond=None)[0]
        assert np.abs(A @ c - u[:, i]).max() < 1e-8, "the displacement field must be homogeneous"
        if i == 0:
            assert abs(c[0] - (0.998**10 - 1.0)) < 1e-8, "axial stretch of the imposed shortening"
        else:
            assert abs(c[1] - (HOMOGENEOUS_DRUCKERPRAGER["lateral stretch"] - 1.0)) < 1e-6, "lateral stretch"
    assert np.abs(n - HOMOGENEOUS_DRUCKERPRAGER["nonlocal field"]).max() < 1e-8, "uniform nonlocal field"
    alphaP = result(mpmModel, "alphaP")
    assert np.abs(alphaP - HOMOGENEOUS_DRUCKERPRAGER["alphaP"]).max() < 1e-7, "uniform hardening variable"


@pytest.mark.parametrize("materialName", MATERIALS.keys())
def test_sim(assert_gold, materialName):
    mpmModel = run_sim(materialName)
    u = result(mpmModel, "displacement")
    n = result(mpmModel, "nonlocal damage")

    assert u[:, 0].min() < -0.9 * SHORTENING, "the right end must follow the imposed shortening"
    if materialName == "neohooke":
        assert n.max() > 5 * KAPPA0, "the nonlocal field must drive the material well into damage"
    else:
        assert result(mpmModel, "alphaP").max() > 0.0, "the material must yield"
        assert n.max() > 0.0, "the plastic flow must drive the nonlocal field"
        checkHomogeneousDruckerPrager(mpmModel, u, n)

    assert_gold(u, np.loadtxt(goldFile(materialName)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--create-gold", dest="create_gold", action="store_true", help="create the gold files.")
    args = parser.parse_args()

    for materialName in MATERIALS:
        mpmModel = run_sim(materialName)
        u = result(mpmModel, "displacement")
        n = result(mpmModel, "nonlocal damage")
        print(materialName, "max nonlocal field", n.max(), " min u_x", u[:, 0].min())
        if args.create_gold:
            np.savetxt(goldFile(materialName), u)
