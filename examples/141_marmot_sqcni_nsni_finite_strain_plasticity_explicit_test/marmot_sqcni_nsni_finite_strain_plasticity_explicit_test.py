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
from edelweissfe.timesteppers.adaptivetimestepper import AdaptiveTimeStepper
from edelweissfe.utils.exceptions import StepFailed

from edelweissmpm.constraints.explicit.particlepenaltycartesianboundaryexplicit import (
    ParticleExplicitPenaltyCartesianBoundaryConstraintFactory,
)
from edelweissmpm.fieldoutput.fieldoutput import MPMFieldOutputController
from edelweissmpm.generators.rectangularkernelfunctiongridgenerator import (
    generateRectangularKernelFunctionGrid,
)
from edelweissmpm.generators.rectangularquadparticlegridgenerator import (
    generateRectangularQuadParticleGrid,
)
from edelweissmpm.meshfree.approximations.marmot.marmotmeshfreeapproximation import (
    MarmotMeshfreeApproximationWrapper,
)
from edelweissmpm.meshfree.kernelfunctions.marmot.marmotmeshfreekernelfunction import (
    MarmotMeshfreeKernelFunctionWrapper,
)
from edelweissmpm.meshfree.particlekerneldomain import ParticleKernelDomain
from edelweissmpm.models.mpmmodel import MPMModel
from edelweissmpm.outputmanagers.ensight import OutputManager as EnsightOutputManager
from edelweissmpm.particlemanagers.kdbinorganizedparticlemanager import (
    KDBinOrganizedParticleManager,
)
from edelweissmpm.particles.marmot.marmotparticlewrapper import MarmotParticleWrapper
from edelweissmpm.solvers.explicitmultiphysicssolver import ExplicitMultiphysicsSolver

# from edelweissmpm.generators.rectangularparticlegridgenerator import (
#     generateRectangularParticleGrid,
# )


def run_sim():
    dimension = 2

    # set nump linewidth to 200:
    np.set_printoptions(linewidth=200)
    # set 2 digits after comma:
    np.set_printoptions(precision=2)
    # and let's print all the array:
    np.set_printoptions(threshold=np.inf)

    theJournal = Journal()

    theModel = MPMModel(dimension)

    particleSize = 100.0 / 10
    supportRadius = particleSize * 3.0

    heightProjectile = 200
    lengthProjectile = 100
    nXProjectile = int(lengthProjectile / particleSize)
    nYProjectile = int(heightProjectile / particleSize)
    # place projectile above plate, centered in x direction
    x0Projectile = 0
    y0Projectile = 0

    def theMeshfreeKernelFunctionFactory(node):
        return MarmotMeshfreeKernelFunctionWrapper(node, "BSplineBoxed", supportRadius=supportRadius, continuityOrder=3)

    theModel = generateRectangularKernelFunctionGrid(
        theModel,
        theJournal,
        theMeshfreeKernelFunctionFactory,
        x0=x0Projectile,
        y0=y0Projectile,
        h=heightProjectile,
        l=lengthProjectile,
        nX=nXProjectile,
        nY=nYProjectile,
        name="projectile",
    )

    # let's define the type of approximation: We would like to have a reproducing kernel approximation of completeness order 1
    theApproximation = MarmotMeshfreeApproximationWrapper("ReproducingKernel", dimension, completenessOrder=1)

    # We need a dummy material for the material point
    E = 78200
    nu = 0.3
    K = E / (3 * (1 - 2 * nu))
    G = E / (2 * (1 + nu))

    timeScalingFactor = 1e-0
    massDensityScalingFactor = 1.0 / (timeScalingFactor**2)

    vProjectileInitial = -400 * 1e3 * timeScalingFactor * 1

    tMax = 2.0e-4

    theMaterialProjectile = {
        "material": "FiniteStrainJ2Plasticity",
        "properties": np.array([K, G, 290, 290, 1, 0, 1, 1.0e-8 * massDensityScalingFactor]),
    }

    def TheProjectileFactory(number, vertexCoordinates, volume):
        return MarmotParticleWrapper(
            "DisplacementSQCNIxNSNI/PlaneStrain/Quad",
            number,
            vertexCoordinates,
            volume,
            theApproximation,
            theMaterialProjectile,
        )

    theModel = generateRectangularQuadParticleGrid(
        theModel,
        theJournal,
        TheProjectileFactory,
        x0=x0Projectile,
        y0=y0Projectile,
        h=heightProjectile,
        l=lengthProjectile,
        nX=nXProjectile,
        nY=nYProjectile,
        name="projectile",
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

    wallBottom = ParticleExplicitPenaltyCartesianBoundaryConstraintFactory(
        "wall_bottom", y0Projectile - 1, 1, theModel.particleSets["projectile_all"], 1e8, theModel
    )

    theModel.constraints.update(wallBottom)

    theModel.prepareYourself(theJournal)
    theJournal.printPrettyTable(theModel.makePrettyTableSummary(), "summary")

    fieldOutputController = MPMFieldOutputController(theModel, theJournal)

    fieldOutputController.addPerParticleFieldOutput(
        "displacement",
        theModel.particleSets["projectile_all"],
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
        "vertex displacements",
        theModel.particleSets["all"],
        "vertex displacements",
        reshape_to_dimensions=2,
    )
    fieldOutputController.addPerParticleFieldOutput(
        "deformation gradient",
        theModel.particleSets["all"],
        "deformation gradient",
    )

    fieldOutputController.initializeJob()

    ensightOutput = EnsightOutputManager("ensight", theModel, fieldOutputController, theJournal, None)
    # ensightOutput.updateDefinition(fieldOutput=fieldOutputController.fieldOutputs["displacement_projectile"], create="perElement")
    ensightOutput.updateDefinition(fieldOutput=fieldOutputController.fieldOutputs["acceleration"], create="perElement")
    ensightOutput.updateDefinition(fieldOutput=fieldOutputController.fieldOutputs["velocity"], create="perElement")
    ensightOutput.updateDefinition(
        fieldOutput=fieldOutputController.fieldOutputs["vertex displacements"],
        name="vertex displacements",
        create="perNode",
    )
    ensightOutput.updateDefinition(
        fieldOutput=fieldOutputController.fieldOutputs["deformation gradient"], create="perElement"
    )
    ensightOutput.initializeJob()

    incSize = 1e-3
    adaptiveTimeStepper = AdaptiveTimeStepper(
        0.0, tMax, incSize, incSize, incSize / 1e8, 2000, theJournal, increaseFactor=1.5
    )

    # nonlinearSolver = NQSParallelForMarmot(theJournal)
    nonlinearSolver = ExplicitMultiphysicsSolver(theJournal)

    # for each particle, set velocity in y direction to -0.1
    for p in theModel.particleSets["projectile_all"]:
        v = p.getResultArray("velocity")
        v[1] = vProjectileInitial
        p.acceptStateAndPosition()

    try:
        nonlinearSolver.solveStep(
            adaptiveTimeStepper,
            theModel,
            fieldOutputController,
            outputManagers=[ensightOutput],
            particleManagers=[theParticleManager],
        )

    except StepFailed as e:
        theJournal.message(f"Step failed: {str(e)}", "error")
        raise

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
    theModel, fieldOutputController = run_sim()
    res = fieldOutputController.fieldOutputs["displacement"].getLastResult()

    parser = argparse.ArgumentParser()
    parser.add_argument("--create-gold", dest="create_gold", action="store_true", help="create the gold file.")
    args = parser.parse_args()

    if args.create_gold:
        np.savetxt("gold.csv", res.flatten())
