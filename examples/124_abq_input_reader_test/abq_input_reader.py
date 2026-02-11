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

import argparse

import edelweissfe.utils.performancetiming as performancetiming
import numpy as np
import pytest
from edelweissfe.journal.journal import Journal
from edelweissfe.linsolve.pardiso.pardiso import pardisoSolve
from edelweissfe.surfaces.entitybasedsurface import EntityBasedSurface
from edelweissfe.timesteppers.adaptivetimestepper import AdaptiveTimeStepper

from edelweissmeshfree.constraints.particlelagrangianweakdirichlet import (
    ParticleLagrangianWeakDirichletOnParticleSetFactory,
)
from edelweissmeshfree.fieldoutput.fieldoutput import MPMFieldOutputController
from edelweissmeshfree.generators.abqinpfilegenerator import (
    generateKernelFunctionGridFromInputFile,
)
from edelweissmeshfree.meshfree.approximations.marmot.marmotmeshfreeapproximation import (
    MarmotMeshfreeApproximationWrapper,
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
from edelweissmeshfree.stepactions.particledistributedload import (
    ParticleDistributedLoad,
)

# import gstools


# from edelweissmeshfree.solvers.nqsmparclength import NonlinearQuasistaticMarmotArcLengthSolver


def run_sim(
    inputFilePath="test_potato.inp",
    name="potato",
    supportRadiusFactor=1.1,
    outputName="_potato",
    particleType="GradientEnhancedMicropolarSQCNIxSDI/PlaneStrain/Quad",
):
    dimension = 2

    # set nump linewidth to 200:
    np.set_printoptions(linewidth=200)
    # set 2 digits after comma:
    np.set_printoptions(precision=2)
    # and let's print all the array:
    np.set_printoptions(threshold=np.inf)

    theJournal = Journal()

    theModel = MPMModel(dimension)

    kernelFunctionSpecifier = {
        "kernelFunction": "BSplineBoxed",
        "continuityOrder": 3,
        "supportRadiusFactor": supportRadiusFactor,
    }

    theApproximation = MarmotMeshfreeApproximationWrapper(
        "ReproducingKernelImplicitGradient", dimension, completenessOrder=1
    )

    # =====================================================================
    #                             MATERIAL
    # =====================================================================
    theMaterial = {
        "material": "GMDamagedShearNeoHooke",
        # E, nu, GcToG, lb, lt, polarRatio
        "properties": np.array([4.5e3, 0.2, 1, 1, 1, 1.4999, 1.0]),
    }

    # =====================================================================
    #                       PARTICLE FACTORY & MODEL
    # =====================================================================

    def TheParticleFactory(number, vertexCoordinates, volume):
        return MarmotParticleWrapper(
            particleType,
            number,
            vertexCoordinates,
            volume,
            theApproximation,
            theMaterial,
        )

    theModel = generateKernelFunctionGridFromInputFile(
        inputFilePath=inputFilePath,
        journal=theJournal,
        model=theModel,
        kernelFunctionSpecifier=kernelFunctionSpecifier,
        particleFactoryCallback=TheParticleFactory,
        firstKernelFunctionNumber=1,
        firstParticleNumber=1,
        name=name,
    )

    # =====================================================================
    #                      SET PARTICLE PROPERTIES
    # =====================================================================

    for particle in theModel.particles.values():
        particle.setProperty("newmark-beta beta", 0.0)
        particle.setProperty("newmark-beta gamma", 0.0)
        particle.setProperty("VCI order", 0)

        if "NSNI" in particleType:
            particle.setProperty("stabilize angular momentum", 1.0)

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

    # =====================================================================
    #                      SET DIRICHLET BCs
    # =====================================================================

    dirichletLeftBottom = ParticleLagrangianWeakDirichletOnParticleSetFactory(
        "leftBottom",
        theModel.particleSets[f"{name}_b_leftBottom"],
        "displacement",
        {0: 0, 1: 0},
        theModel,
        location="center",
    )
    dirichletRightBottom = ParticleLagrangianWeakDirichletOnParticleSetFactory(
        "rightBottom",
        theModel.particleSets[f"{name}_b_rightBottom"],
        "displacement",
        {0: 0, 1: 0},
        theModel,
        location="center",
    )
    theModel.constraints.update(dirichletLeftBottom)
    theModel.constraints.update(dirichletRightBottom)
    theModel.prepareYourself(theJournal)

    theJournal.printPrettyTable(theModel.makePrettyTableSummary(), "summary")

    # =====================================================================
    #                      SET FIELD OUTPUT
    # =====================================================================

    fieldOutputController = MPMFieldOutputController(theModel, theJournal)

    fieldOutputController.addPerParticleFieldOutput(
        "displacement",
        theModel.particleSets["all"],
        "displacement",
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
    fieldOutputController.addPerParticleFieldOutput(
        "stress",
        theModel.particleSets["all"],
        "stress",
    )
    fieldOutputController.addPerParticleFieldOutput(
        "microrotation",
        theModel.particleSets["all"],
        "microrotation",
    )

    fieldOutputController.initializeJob()

    # =====================================================================
    #                      ENSIGHT OUTPUT
    # =====================================================================

    ensightOutput = EnsightOutputManager(outputName, theModel, fieldOutputController, theJournal, None)
    ensightOutput.updateDefinition(fieldOutput=fieldOutputController.fieldOutputs["displacement"], create="perElement")
    ensightOutput.updateDefinition(
        fieldOutput=fieldOutputController.fieldOutputs["vertex displacements"],
        create="perNode",
    )
    ensightOutput.updateDefinition(
        fieldOutput=fieldOutputController.fieldOutputs["deformation gradient"], create="perElement"
    )
    ensightOutput.updateDefinition(fieldOutput=fieldOutputController.fieldOutputs["stress"], create="perElement")
    ensightOutput.updateDefinition(fieldOutput=fieldOutputController.fieldOutputs["microrotation"], create="perElement")
    ensightOutput.initializeJob()

    # =====================================================================
    #                      SOLVER SETUP
    # =====================================================================

    incSize = 1
    adaptiveTimeStepper = AdaptiveTimeStepper(
        currentTime=theModel.time,
        stepLength=1,
        startIncrement=incSize,
        maxIncrement=incSize,
        minIncrement=incSize / 1,
        maxNumberIncrements=5000,
        journal=theJournal,
    )

    nonlinearSolver = NonlinearQuasistaticSolver(theJournal)
    iterationOptions = nonlinearSolver.validOptions.copy()

    iterationOptions["max. iterations"] = 15
    iterationOptions["critical iterations"] = 3
    iterationOptions["allowed residual growths"] = 10

    iterationOptions["default relative flux residual tolerance alt."] = 5e-2
    iterationOptions["default absolute field correction tolerance"] = 1e10
    iterationOptions["spec. relative flux residual tolerances"]["nonlocal damage"] = 1e10
    iterationOptions["spec. absolute flux residual tolerances"]["nonlocal damage"] = 1e10
    iterationOptions["spec. relative flux residual tolerances"]["micro rotation"] = 1e10
    iterationOptions["spec. absolute flux residual tolerances"]["micro rotation"] = 1e10
    iterationOptions["spec. relative field correction tolerances"]["nonlocal damage"] = 1e10
    iterationOptions["spec. absolute field correction tolerances"]["nonlocal damage"] = 1e10
    # iterationOptions["spec. relative flux residual tolerances"]["nonlocal damage"] = 1e10
    # iterationOptions["spec. absolute flux residual tolerances"]["nonlocal damage"] = 1e10
    iterationOptions["spec. relative field correction tolerances"]["micro rotation"] = 1e10
    iterationOptions["spec. absolute field correction tolerances"]["micro rotation"] = 1e10

    linearSolver = pardisoSolve

    from edelweissmeshfree.meshfree.vci import (
        BoundaryParticleDefinition,
        VariationallyConsistentIntegrationManager,
    )

    # =====================================================================
    #                      VCI SETUP
    # =====================================================================

    theBoundary = [
        BoundaryParticleDefinition(theModel.particleSets[f"{name}__s_wholeBoundary_S1"], np.empty(2), 1),
        BoundaryParticleDefinition(theModel.particleSets[f"{name}__s_wholeBoundary_S2"], np.empty(2), 2),
        BoundaryParticleDefinition(theModel.particleSets[f"{name}__s_wholeBoundary_S3"], np.empty(2), 3),
        BoundaryParticleDefinition(theModel.particleSets[f"{name}__s_wholeBoundary_S4"], np.empty(2), 4),
    ]

    vciManager = VariationallyConsistentIntegrationManager(
        list(theModel.particles.values()), list(theModel.meshfreeKernelFunctions.values()), theBoundary
    )

    # =====================================================================
    #                      DISTRIBUTED LOADS
    # =====================================================================
    surfacePressure = EntityBasedSurface(
        name="surfacePressure",
        faceToEntities={
            1: list(theModel.particleSets[f"{name}__s_top_S1"]),
            3: list(theModel.particleSets[f"{name}__s_top_S3"]),
        },
    )

    pressure_S1_S3 = ParticleDistributedLoad(
        name="pressure_S1_S3",
        model=theModel,
        journal=theJournal,
        particleSurface=surfacePressure,
        distributedLoadType="pressure",
        loadVector=np.array([-10]),
        # surfaceID=1,
        f_t=lambda t: t,
    )
    # pressure_S3 = ParticleDistributedLoad(
    #    name="pressure_S3",
    #    model=theModel,
    #    journal=theJournal,
    #    particles=theModel.particleSets[f"{name}__s_top_S3"],
    #    distributedLoadType="pressure",
    #    loadVector=np.array([-10]),
    #    surfaceID=3,
    #    f_t=lambda t: t,
    # )

    # =====================================================================
    #                      SOLVE STEPS
    # =====================================================================

    try:
        theJournal.printSeperationLine()
        theJournal.message("Loading", "Step 1")
        nonlinearSolver.solveStep(
            adaptiveTimeStepper,
            linearSolver,
            theModel,
            fieldOutputController,
            outputManagers=[ensightOutput],
            particleManagers=[theParticleManager],
            constraints=theModel.constraints.values(),
            userIterationOptions=iterationOptions,
            particleDistributedLoads=[pressure_S1_S3],  # , pressure_S3],
            vciManagers=[vciManager],
        )

        prettytable = performancetiming.makePrettyTable()
        theJournal.printPrettyTable(prettytable, "Summary Step 1")
        performancetiming.times.clear()

    finally:
        fieldOutputController.finalizeJob()
        ensightOutput.finalizeJob()

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

    assert np.isclose(res.flatten(), gold.flatten(), rtol=1e-12).all()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--create-gold", dest="create_gold", action="store_true", help="create the gold file.")
    args = parser.parse_args()

    theModel, fieldOutputController = run_sim()
    res = fieldOutputController.fieldOutputs["displacement"].getLastResult()

    if args.create_gold:
        np.savetxt("gold.csv", res.flatten())
