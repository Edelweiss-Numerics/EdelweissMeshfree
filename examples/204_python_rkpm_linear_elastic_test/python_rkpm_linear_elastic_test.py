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
Test for the pure Python RKPM particle implementation with linear elastic behavior
and Variationally Consistent Integration (VCI).

Uniaxial tension: left fully fixed (penalty weak Dirichlet), right displaced by +0.01 in y.
Domain [-1, 3] x [-1, 0] with 20x5 kernel functions and 20x5 RKPM particles using:
- PythonKernelFunction (cubic B-Spline with boxed support)
- PythonRKPMApproximation (reproducing kernel, completeness order 1)
- PythonParticle with linear elastic plane strain material
- Variationally Consistent Integration for Galerkin exactness
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

from edelweissmeshfree.constraints.particlepenaltyweakdirichtlet import (
    ParticlePenaltyWeakDirichlet,
)
from edelweissmeshfree.fieldoutput.fieldoutput import MPMFieldOutputController
from edelweissmeshfree.generators.rectangularkernelfunctiongridgenerator import (
    generateRectangularKernelFunctionGrid,
)
from edelweissmeshfree.generators.rectangularparticlegridgenerator import (
    generateRectangularParticleGrid,
)
from edelweissmeshfree.meshfree.approximations.python.pythonrkpmapproximation import (
    PythonRKPMApproximation,
)
from edelweissmeshfree.meshfree.kernelfunctions.python.pythonkernelfunction import (
    PythonKernelFunction,
)
from edelweissmeshfree.meshfree.particlekerneldomain import ParticleKernelDomain
from edelweissmeshfree.meshfree.vci import (
    BoundaryParticleDefinition,
    VariationallyConsistentIntegrationManager,
)
from edelweissmeshfree.models.mpmmodel import MPMModel
from edelweissmeshfree.outputmanagers.ensight import (
    OutputManager as EnsightOutputManager,
)
from edelweissmeshfree.particlemanagers.kdbinorganizedparticlemanager import (
    KDBinOrganizedParticleManager,
)
from edelweissmeshfree.particles.python.pythonparticle import PythonParticle
from edelweissmeshfree.solvers.nqs import NonlinearQuasistaticSolver


def run_sim():
    """Left fully fixed, right displaced by +0.01 in y (uniaxial tension)."""

    dimension = 2

    journal = Journal()

    theModel = MPMModel(dimension)

    x0 = -1.0
    y0 = -1.0
    height = 1.0
    length = 4.0
    nX = 20
    nY = 5
    supportRadius = 0.5

    # Create kernel function grid using pure Python B-Spline kernel functions
    def theMeshfreeKernelFunctionFactory(node):
        return PythonKernelFunction(node, supportRadius=supportRadius)

    theModel = generateRectangularKernelFunctionGrid(
        theModel,
        journal,
        theMeshfreeKernelFunctionFactory,
        x0=x0,
        y0=y0,
        h=height,
        l=length,
        nX=nX,
        nY=nY,
    )

    # Define RKPM approximation with completeness order 1 (linear reproduction)
    theApproximation = PythonRKPMApproximation(dimension, completenessOrder=1)

    # Material: linear elastic plane strain
    E = 200.0
    nu = 0.3
    theMaterial = {"material": "LinearElastic", "properties": np.array([E, nu])}

    # Create particle factory
    def theParticleFactory(number, coordinates, volume):
        return PythonParticle(
            "Displacement/PlaneStrain/Point",
            number,
            coordinates,
            volume,
            theApproximation,
            theMaterial,
        )

    theModel = generateRectangularParticleGrid(
        theModel,
        journal,
        theParticleFactory,
        x0=x0,
        y0=y0,
        h=height,
        l=length,
        nX=nX,
        nY=nY,
    )

    # Create particle-kernel domain
    theParticleKernelDomain = ParticleKernelDomain(
        list(theModel.particles.values()), list(theModel.meshfreeKernelFunctions.values())
    )

    # Create particle manager (organizes connectivity)
    theParticleManager = KDBinOrganizedParticleManager(
        theParticleKernelDomain, dimension, journal, bondParticlesToKernelFunctions=True
    )

    # Register domain and prepare model
    theModel.particleKernelDomains["my_all_with_all"] = theParticleKernelDomain
    theModel.prepareYourself(journal)

    # Field output
    fieldOutputController = MPMFieldOutputController(theModel, journal)
    fieldOutputController.addPerParticleFieldOutput(
        "displacement",
        theModel.particleSets["all"],
        "displacement",
    )
    fieldOutputController.initializeJob()

    ensightOutput = EnsightOutputManager("ensight", theModel, fieldOutputController, journal, None)
    ensightOutput.updateDefinition(fieldOutput=fieldOutputController.fieldOutputs["displacement"], create="perNode")
    ensightOutput.initializeJob()

    # Boundary conditions (penalty weak Dirichlet)
    dirichletLeft = ParticlePenaltyWeakDirichlet(
        "left", theModel, theModel.particleSets["rectangular_grid_left"], "displacement", {0: 0.0, 1: 0.0}, 1e6
    )
    dirichletRight = ParticlePenaltyWeakDirichlet(
        "right", theModel, theModel.particleSets["rectangular_grid_right"], "displacement", {0: 0.0, 1: 0.01}, 1e6
    )

    # VCI boundary definitions
    theBoundary = [
        BoundaryParticleDefinition(
            theModel.particleSets["rectangular_grid_left"], np.array([-1.0, 0.0]) * height / nY, 0
        ),
        BoundaryParticleDefinition(
            theModel.particleSets["rectangular_grid_right"], np.array([1.0, 0.0]) * height / nY, 0
        ),
        BoundaryParticleDefinition(
            theModel.particleSets["rectangular_grid_bottom"], np.array([0.0, -1.0]) * length / nX, 0
        ),
        BoundaryParticleDefinition(
            theModel.particleSets["rectangular_grid_top"], np.array([0.0, 1.0]) * length / nX, 0
        ),
    ]

    vciManager = VariationallyConsistentIntegrationManager(
        list(theModel.particles.values()), list(theModel.meshfreeKernelFunctions.values()), theBoundary
    )

    # Time stepping and solver
    adaptiveTimeStepper = AdaptiveTimeStepper(0.0, 1.0, 1e-1, 1e-1, 1e-3, 1000, journal)
    nonlinearSolver = NonlinearQuasistaticSolver(journal)

    iterationOptions = {
        "max. iterations": 15,
        "critical iterations": 3,
        "allowed residual growths": 5,
    }

    try:
        nonlinearSolver.solveStep(
            adaptiveTimeStepper,
            pardisoSolve,
            theModel,
            fieldOutputController,
            outputManagers=[ensightOutput],
            particleManagers=[theParticleManager],
            constraints=[dirichletLeft, dirichletRight],
            userIterationOptions=iterationOptions,
            vciManagers=[vciManager],
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

    return theModel


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    """No matter where pytest is ran, we set the working dir
    to this testscript's parent directory"""

    monkeypatch.chdir(request.fspath.dirname)


def test_sim(assert_gold):
    """Uniaxial tension with RKPM: left fixed, right displaced by +0.01 in y."""
    theModel = run_sim()
    ordered_particles = [p for _, p in sorted(theModel.particles.items())]
    res = np.array([p.getResultArray("displacement") for p in ordered_particles])
    assert_gold(res, np.loadtxt("gold.csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--create-gold", dest="create_gold", action="store_true", help="create the gold file.")
    args = parser.parse_args()

    theModel = run_sim()
    if args.create_gold:
        ordered_particles = [p for _, p in sorted(theModel.particles.items())]
        gold = np.array([p.getResultArray("displacement") for p in ordered_particles])
        np.savetxt("gold.csv", gold)
        print("Wrote gold.csv")
