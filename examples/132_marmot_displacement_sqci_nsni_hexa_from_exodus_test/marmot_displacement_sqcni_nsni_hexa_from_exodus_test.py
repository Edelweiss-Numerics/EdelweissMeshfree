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

from edelweissmeshfree.fieldoutput.fieldoutput import MPMFieldOutputController
from edelweissmeshfree.meshfree.approximations.marmot.marmotmeshfreeapproximation import (
    MarmotMeshfreeApproximationWrapper,
)
from edelweissmeshfree.meshfree.kernelfunctions.marmot.marmotmeshfreekernelfunction import (
    MarmotMeshfreeKernelFunctionWrapper,
)
from edelweissmeshfree.meshfree.particlekerneldomain import ParticleKernelDomain
from edelweissmeshfree.models.mpmmodel import MPMModel
from edelweissmeshfree.outputmanagers.ensight import OutputManager as EnsightOutputManager
from edelweissmeshfree.particlemanagers.kdbinorganizedparticlemanager import (
    KDBinOrganizedParticleManager,
)
from edelweissmeshfree.particles.marmot.marmotparticlewrapper import MarmotParticleWrapper
from edelweissmeshfree.solvers.nqs import NonlinearQuasistaticSolver


def run_sim():
    dimension = 3

    # set nump linewidth to 200:
    np.set_printoptions(linewidth=200)
    # set 2 digits after comma:
    np.set_printoptions(precision=2)
    # and let's print all the array:
    np.set_printoptions(threshold=np.inf)

    theJournal = Journal()

    theModel = MPMModel(dimension)

    # let's define the type of approximation: We would like to have a reproducing kernel approximation of completeness order 1
    theApproximation = MarmotMeshfreeApproximationWrapper("ReproducingKernel", dimension, completenessOrder=1)

    E = 400
    nu = 0.3
    K = E / (3 * (1 - 2 * nu))
    G = E / (2 * (1 + nu))

    # We need a dummy material for the material point
    theMaterial = {
        "material": "FiniteStrainJ2Plasticity",
        "properties": np.array([K, G, 2e2, 2e2, 1e0, 1e0, 1, 1e-7]),
    }

    def TheParticleFactory(number, vertexCoordinates):
        return MarmotParticleWrapper(
            "DisplacementSQCNIxNSNI/3D/Hexa",
            number,
            vertexCoordinates,
            0.0,
            theApproximation,
            theMaterial,
        )

    from edelweissmeshfree.generators.particlesfromexodus import generateParticlesFromExodus

    theModel = generateParticlesFromExodus(
        theModel, theJournal, "brick_coarse.e", {"hexahedron": TheParticleFactory}, "mesh_particles", 1
    )

    def theMeshfreeKernelFunctionFactory(node, characteristicLength):
        return MarmotMeshfreeKernelFunctionWrapper(
            node, "BSplineBoxed", supportRadius=characteristicLength, continuityOrder=3
        )

    from edelweissmeshfree.generators.kernelmatchingtoparticlegenerator import (
        generateKernelMatchingToParticle,
    )

    theModel = generateKernelMatchingToParticle(
        theModel,
        theJournal,
        theMeshfreeKernelFunctionFactory,
        theModel.particleSets["mesh_particles_all"],
        supportScalingFactor=2.2,
    )

    # let's create the particle kernel domain
    theParticleKernelDomain = ParticleKernelDomain(
        list(theModel.particles.values()), list(theModel.meshfreeKernelFunctions.values())
    )

    # for Semi-Lagrangian particle methods, we assoicate a particle with a kernel function.
    theParticleManager = KDBinOrganizedParticleManager(
        theParticleKernelDomain,
        dimension,
        theJournal,
        bondParticlesToKernelFunctions=True,
    )

    # let's print some details
    print(theParticleManager)

    # We now create a bundled model.
    # We need this model to create the dof manager
    theModel.particleKernelDomains["my_all_with_all"] = theParticleKernelDomain

    from edelweissmeshfree.constraints.particlelagrangianweakdirichlet import (
        ParticleLagrangianWeakDirichletOnParticleSetFactory,
    )

    dirchletBottom = ParticleLagrangianWeakDirichletOnParticleSetFactory(
        "bottom", theModel.surfaces["mesh_particles_sideset_bottom"], "displacement", {1: 0}, theModel, location="face"
    )

    dirichletXSymmetry = ParticleLagrangianWeakDirichletOnParticleSetFactory(
        "xsym", theModel.surfaces["mesh_particles_sideset_xsym"], "displacement", {0: 0}, theModel, location="face"
    )

    dirichletZSymmetry = ParticleLagrangianWeakDirichletOnParticleSetFactory(
        "zsym", theModel.surfaces["mesh_particles_sideset_zsym"], "displacement", {2: 0}, theModel, location="face"
    )

    dirichletLoadTop = ParticleLagrangianWeakDirichletOnParticleSetFactory(
        "top", theModel.surfaces["mesh_particles_sideset_load"], "displacement", {1: -5}, theModel, location="face"
    )

    theModel.constraints.update(dirchletBottom)
    theModel.constraints.update(dirichletXSymmetry)
    theModel.constraints.update(dirichletZSymmetry)
    theModel.constraints.update(dirichletLoadTop)

    theModel.constraintSets["bottom"] = dirchletBottom
    theModel.constraintSets["xsym"] = dirichletXSymmetry
    theModel.constraintSets["zsym"] = dirichletZSymmetry
    theModel.constraintSets["top"] = dirichletLoadTop

    theModel.prepareYourself(theJournal)
    theJournal.printPrettyTable(theModel.makePrettyTableSummary(), "summary")

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
        f_x=lambda x: np.reshape(x, (-1, 3)),
    )
    fieldOutputController.addPerParticleFieldOutput(
        "deformation gradient",
        theModel.particleSets["all"],
        "deformation gradient",
    )

    fieldOutputController.initializeJob()

    ensightOutput = EnsightOutputManager(
        "ensight", theModel, fieldOutputController, theJournal, None, intermediateSaveInterval=2
    )
    ensightOutput.updateDefinition(fieldOutput=fieldOutputController.fieldOutputs["displacement"], create="perElement")
    ensightOutput.updateDefinition(
        fieldOutput=fieldOutputController.fieldOutputs["vertex displacements"],
        create="perNode",
    )
    ensightOutput.updateDefinition(
        fieldOutput=fieldOutputController.fieldOutputs["deformation gradient"], create="perElement"
    )
    ensightOutput.initializeJob()

    dirichlets = theModel.constraints.values()

    incSize = 2e-2
    adaptiveTimeStepper = AdaptiveTimeStepper(
        0.0,
        1.0,
        incSize,
        incSize,
        incSize / 100,
        2,
        theJournal,
    )

    # nonlinearSolver = NQSParallelForMarmot(theJournal)
    nonlinearSolver = NonlinearQuasistaticSolver(theJournal)

    iterationOptions = {
        "default relative flux residual tolerance": 1e-8,
        "default relative flux residual tolerance alt.": 1e-6,
        "default relative field correction tolerance": 1e-9,
        "default absolute flux residual tolerance": 1e-14,
        "default absolute field correction tolerance": 1e-14,
        "max. iterations": 20,
    }

    linearSolver = pardisoSolve

    # pressureTop = ParticleDistributedLoad(
    #     "pressureTop",
    #     theModel,
    #     theJournal,
    #     theModel.surfaces["mesh_particles_sideset_load"],
    #     "pressure",
    #     np.array([-1e3]),
    # )

    from edelweissmeshfree.numerics.predictors.quadraticpredictor import QuadraticPredictor

    try:
        nonlinearSolver.solveStep(
            adaptiveTimeStepper,
            linearSolver,
            theModel,
            fieldOutputController,
            outputManagers=[ensightOutput],
            particleManagers=[theParticleManager],
            constraints=dirichlets,
            userIterationOptions=iterationOptions,
            # particleDistributedLoads=[pressureTop],
            predictor=QuadraticPredictor(),
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

    assert np.isclose(np.copy(res.flatten() - gold.flatten()), 0.0, rtol=1e-7).all()


if __name__ == "__main__":
    theModel, fieldOutputController = run_sim()
    res = fieldOutputController.fieldOutputs["displacement"].getLastResult()

    parser = argparse.ArgumentParser()
    parser.add_argument("--create-gold", dest="create_gold", action="store_true", help="create the gold file.")
    args = parser.parse_args()

    if args.create_gold:
        np.savetxt("gold.csv", res.flatten())
