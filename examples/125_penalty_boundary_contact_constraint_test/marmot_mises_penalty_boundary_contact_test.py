# -*- coding: utf-8 -*-
#  ---------------------------------------------------------------------
#
#  _____    _      _              _         __  __ ____  __  __
# | ____|__| | ___| |_      _____(_)___ ___|  \/  |  _ \|  \/  |
# |  _| / _` |/ _ \ \ \ /\ / / _ \ / __/ __| |\/| | |_) | |\/| |
# | |__| (_| |  __/ |\ V  V /  __/ \__ \__ \ |  | |  __/| |  | |
# |_____\__,_|\___|_| \_/\_/ \___|_|___/___/_|  |_|_|   |_|  |_|
#
#
#  Unit of Strength of Materials and Structural Analysis
#  University of Innsbruck,
#  2023 - today
#
#  Matthias Neuner matthias.neuner@uibk.ac.at
#
#  This file is part of EdelweissMPM.
#
#  This library is free software; you can redistribute it and/or
#  modify it under the terms of the GNU Lesser General Public
#  License as published by the Free Software Foundation; either
#  version 2.1 of the License, or (at your option) any later version.
#
#  The full text of the license can be found in the file LICENSE.md at
#  the top level directory of EdelweissMPM.
#  ---------------------------------------------------------------------

import argparse

import edelweissfe.utils.performancetiming as performancetiming
import numpy as np
import pytest
from edelweissfe.journal.journal import Journal
from edelweissfe.linsolve.pardiso.pardiso import pardisoSolve
from edelweissfe.timesteppers.adaptivetimestepper import AdaptiveTimeStepper
from edelweissfe.utils.exceptions import StepFailed

from edelweissmeshfree.constraints.particlelagrangianweakdirichlet import (
    ParticleLagrangianWeakDirichletOnParticleSetFactory,
)
from edelweissmeshfree.constraints.particlepenaltycartesianboundary import (
    ParticlePenaltyContactCartesianBoundaryConstraintFactory,
)
from edelweissmeshfree.fieldoutput.fieldoutput import MPMFieldOutputController
from edelweissmeshfree.generators.rectangularkernelfunctiongridgenerator import (
    generateRectangularKernelFunctionGrid,
)
from edelweissmeshfree.generators.rectangularquadparticlegridgenerator import (
    generateRectangularQuadParticleGrid,
)
from edelweissmeshfree.meshfree.approximations.marmot.marmotmeshfreeapproximation import (
    MarmotMeshfreeApproximationWrapper,
)
from edelweissmeshfree.meshfree.kernelfunctions.marmot.marmotmeshfreekernelfunction import (
    MarmotMeshfreeKernelFunctionWrapper,
)
from edelweissmeshfree.meshfree.particlekerneldomain import ParticleKernelDomain
from edelweissmeshfree.models.mpmmodel import MPMModel
from edelweissmeshfree.outputmanagers.ensight import (
    OutputManager as EnsightOutputManager,
)
from edelweissmeshfree.particlemanagers.kdbinorganizedparticlemanager import (
    KDBinOrganizedParticleManager,
)
from edelweissmeshfree.particles.marmot.marmotparticlewrapper import (
    MarmotParticleWrapper,
)
from edelweissmeshfree.solvers.nqs import NonlinearQuasistaticSolver

# from edelweissmeshfree.generators.rectangularparticlegridgenerator import (
#     generateRectangularParticleGrid,
# )


def run_sim(no_limit=False):
    dimension = 2

    E = 200
    nu = 0.25
    K = E / (3 * (1 - 2 * nu))
    G = E / (2 * (1 + nu))

    timeScalingFactor = 2e-0
    massDensityScalingFactor = 1.0 / (timeScalingFactor**2)

    particleSize = 1.0 / 10
    supportRadius = particleSize * 2.0

    x0Slab = 0
    y0Slab = 0
    heightSlab = 1
    lengthSlab = 6
    nXSlab = int(lengthSlab / particleSize)
    nYSlab = int(heightSlab / particleSize)

    # set nump linewidth to 200:
    np.set_printoptions(linewidth=200)
    # set 2 digits after comma:
    np.set_printoptions(precision=2)
    # and let's print all the array:
    np.set_printoptions(threshold=np.inf)

    theJournal = Journal()

    theModel = MPMModel(dimension)

    def theMeshfreeKernelFunctionFactory(node):
        return MarmotMeshfreeKernelFunctionWrapper(node, "BSplineBoxed", supportRadius=supportRadius, continuityOrder=3)

    theModel = generateRectangularKernelFunctionGrid(
        theModel,
        theJournal,
        theMeshfreeKernelFunctionFactory,
        x0=x0Slab,
        y0=y0Slab,
        h=heightSlab,
        l=lengthSlab,
        nX=nXSlab,
        nY=nYSlab,
        name="slab",
    )

    # let's define the type of approximation: We would like to have a reproducing kernel approximation of completeness order 1
    theApproximation = MarmotMeshfreeApproximationWrapper(
        "ReproducingKernelImplicitGradient", dimension, completenessOrder=1
    )

    theMaterial = {
        "material": "FiniteStrainJ2Plasticity",
        "properties": np.array([K, G, 20e10, 2010, 0, 0, 1, 1e-9 * massDensityScalingFactor]),
    }

    def TheSlabFactory(number, vertexCoordinates, volume):
        return MarmotParticleWrapper(
            "DisplacementSQCNIxNSNI/PlaneStrain/Quad",
            number,
            vertexCoordinates,
            volume,
            theApproximation,
            theMaterial,
        )

    theModel = generateRectangularQuadParticleGrid(
        theModel,
        theJournal,
        TheSlabFactory,
        x0=x0Slab,
        y0=y0Slab,
        h=heightSlab,
        l=lengthSlab,
        nX=nXSlab,
        nY=nYSlab,
        name="slab",
    )

    # let's create the particle kernel domain
    theParticleKernelDomain = ParticleKernelDomain(
        list(theModel.particles.values()), list(theModel.meshfreeKernelFunctions.values())
    )

    # for Semi-Lagrangian particle methods, we assoicate a particle with a kernel function.
    theParticleManager = KDBinOrganizedParticleManager(
        theParticleKernelDomain, dimension, theJournal, bondParticlesToKernelFunctions=True
    )

    # let's print some details
    print(theParticleManager)

    # We now create a bundled model.
    # We need this model to create the dof manager
    theModel.particleKernelDomains["my_all_with_all"] = theParticleKernelDomain

    dirichletLeft = ParticleLagrangianWeakDirichletOnParticleSetFactory(
        "left", theModel.surfaces["slab_left"], "displacement", {0: 3}, theModel, location="face"
    )
    theModel.constraints.update(dirichletLeft)

    dirichletBottom = ParticleLagrangianWeakDirichletOnParticleSetFactory(
        "bottom", theModel.surfaces["slab_bottom"], "displacement", {1: 0.0}, theModel, location="face"
    )
    theModel.constraints.update(dirichletBottom)

    boundaryRight = ParticlePenaltyContactCartesianBoundaryConstraintFactory(
        "right",
        7.0,
        0,
        theModel.particleSets["slab_all"],
        "displacement",
        theModel,
        location="face",
        faceIDs=2,
        penaltyParameter=1e7,
    )
    theModel.constraints.update(boundaryRight)

    theModel.prepareYourself(theJournal)
    theJournal.printPrettyTable(theModel.makePrettyTableSummary(), "summary")

    fieldOutputController = MPMFieldOutputController(theModel, theJournal)

    fieldOutputController.addPerParticleFieldOutput(
        "displacement",
        theModel.particleSets["all"],
        "displacement",
    )
    fieldOutputController.addPerParticleFieldOutput(
        "velocity",
        theModel.particleSets["all"],
        "velocity",
    )
    fieldOutputController.addPerParticleFieldOutput(
        "acceleration",
        theModel.particleSets["all"],
        "acceleration",
    )
    fieldOutputController.addPerParticleFieldOutput(
        "particle acceleration",
        theModel.particleSets["all"],
        "acceleration",
    )
    fieldOutputController.addPerParticleFieldOutput(
        "vertex displacements",
        theModel.particleSets["all"],
        "vertex displacements",
        f_x=lambda x: np.pad(np.reshape(x, (-1, 2)), ((0, 0), (0, 1)), mode="constant", constant_values=0),
    )
    fieldOutputController.addPerParticleFieldOutput(
        "deformation gradient",
        theModel.particleSets["all"],
        "deformation gradient",
    )

    fieldOutputController.initializeJob()

    ensightOutput = EnsightOutputManager("ensight", theModel, fieldOutputController, theJournal, None)
    # ensightOutput.minDTForOutput = 1e-3
    ensightOutput.updateDefinition(
        fieldOutput=fieldOutputController.fieldOutputs["acceleration"], create="perElement", name="acceleration"
    )
    ensightOutput.updateDefinition(
        fieldOutput=fieldOutputController.fieldOutputs["velocity"], create="perElement", name="velocity"
    )
    ensightOutput.updateDefinition(
        fieldOutput=fieldOutputController.fieldOutputs["vertex displacements"],
        create="perNode",
        name="vertex displacements",
    )
    ensightOutput.initializeJob()

    incSize = 1e-1
    adaptiveTimeStepper = AdaptiveTimeStepper(
        0.0, 1.0, incSize, incSize, incSize / 1e8, 5000 if not no_limit else 10000, theJournal, increaseFactor=1.5
    )

    nonlinearSolver = NonlinearQuasistaticSolver(theJournal)

    iterationOptions = dict()

    iterationOptions["max. iterations"] = 10
    iterationOptions["critical iterations"] = 7
    iterationOptions["allowed residual growths"] = 2
    iterationOptions["default relative flux residual tolerance"] = 1e-6
    iterationOptions["default absolute flux residual tolerance"] = 1e-12
    iterationOptions["default absolute field correction tolerance"] = 1e-9

    linearSolver = pardisoSolve

    try:
        nonlinearSolver.solveStep(
            adaptiveTimeStepper,
            linearSolver,
            theModel,
            fieldOutputController,
            outputManagers=[ensightOutput],
            particleManagers=[theParticleManager],
            constraints=theModel.constraints.values(),
            userIterationOptions=iterationOptions,
            # predictor=QuadraticPredictor(theJournal)
        )

        adaptiveTimeStepper = AdaptiveTimeStepper(
            theModel.time,
            1.0,
            incSize,
            incSize,
            incSize / 1e8,
            5000 if not no_limit else 10000,
            theJournal,
            increaseFactor=1.5,
        )

        dirichletLeft = ParticleLagrangianWeakDirichletOnParticleSetFactory(
            "left", theModel.surfaces["slab_left"], "displacement", {0: -6}, theModel, location="face"
        )
        theModel.constraints.update(dirichletLeft)
        theModel.scalarVariables.clear()
        theModel._createAndAssignScalarVariableForConstraints(theJournal)
        # theModel.prepareYourself(theJournal)

        nonlinearSolver.solveStep(
            adaptiveTimeStepper,
            linearSolver,
            theModel,
            fieldOutputController,
            outputManagers=[ensightOutput],
            particleManagers=[theParticleManager],
            constraints=theModel.constraints.values(),
            userIterationOptions=iterationOptions,
            # predictor=QuadraticPredictor(theJournal)
        )

    except StepFailed as e:
        theJournal.message(f"Step failed: {str(e)}", "error")
        if not no_limit:
            theJournal.message(
                "This is an expected behaviour for this test. Rerun with --no-limit to run until the end.", "error"
            )

    finally:
        fieldOutputController.finalizeJob()
        ensightOutput.finalizeJob()

        prettytable = performancetiming.makePrettyTable()
        prettytable.min_table_width = theJournal.linewidth
        theJournal.printPrettyTable(prettytable, "Summary")

    return theModel, fieldOutputController


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    """No matter where pytest is ran, we set the working dir
    to this testscript's parent directory"""

    monkeypatch.chdir(request.fspath.dirname)


def test_sim():

    # disable plots and suppress warnings
    import matplotlib

    matplotlib.use("Agg")
    import warnings

    warnings.filterwarnings("ignore")

    theModel, fieldOutputController = run_sim()

    res = fieldOutputController.fieldOutputs["displacement"].getLastResult()

    gold = np.loadtxt("gold.csv")

    assert np.isclose(np.copy(res.flatten() - gold.flatten()), 0.0, rtol=1e-12).all()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--create-gold", dest="create_gold", action="store_true", help="create the gold file.")
    parser.add_argument(
        "--no-limit", dest="no_limit", action="store_true", help="do not limit the number of time increments."
    )
    args = parser.parse_args()

    theModel, fieldOutputController = run_sim(args.no_limit)
    res = fieldOutputController.fieldOutputs["displacement"].getLastResult()

    if args.create_gold:
        np.savetxt("gold.csv", res.flatten())
