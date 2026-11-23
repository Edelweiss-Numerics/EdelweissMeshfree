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

"""A slab of RKPM particles pushed against a wall and pulled back: the penalty contact of particle faces with a
Cartesian boundary (:mod:`edelweissmeshfree.constraints.particlepenaltycartesianboundary`), implicit.

The slab, 6 x 1, rests on a frictionless floor (its bottom held vertically). Its left face is pushed 3 to the right,
against a wall at x = 7: the slab, compressed from 6 to 4, presses on the wall, and spreads vertically. Then the left
face is pulled back by 6: the slab leaves the wall and comes to rest, unstressed, 3 left of where it started.

The test asserts what this is about: the wall pushes back with the force of the left face, the floor carries nothing,
the faces at the wall enter it by their force over the penalty and no further, the compression and the vertical
spreading are uniform, and after the release the slab is back at rest without stress. The force of the wall and the
vertical stretch are compared with the gold.
"""

import argparse

import edelweissfe.utils.performancetiming as performancetiming
import numpy as np
import pytest
from edelweissfe.config.linsolve import getLinSolverByName
from edelweissfe.journal.journal import Journal
from edelweissfe.timesteppers.adaptivetimestepper import AdaptiveTimeStepper

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
from edelweissmeshfree.solvers.deprecated.nqs import NonlinearQuasistaticSolver

# from edelweissmeshfree.generators.rectangularparticlegridgenerator import (
#     generateRectangularParticleGrid,
# )


#: The relative tolerance of the gold: far above the sensitivity of the force and the stretch to round-off, which changes
#: them by less than 1e-13 (another number of threads, increments perturbed by 1e-12).
GOLD_TOLERANCE = 1e-4

#: The position of the wall.
WALL = 7.0

#: The penalty of the wall, per face of a particle.
WALL_PENALTY = 1e7


def stateOfTheSlab(model: MPMModel, fieldOutputController: MPMFieldOutputController) -> dict:
    """What the example is about, at the end of a step: the reactions of the left face, the floor and the wall, where
    the faces at the wall are, the displacements of the particles' centers, with their reference coordinates, and the
    vertical stretch of the slab, the least-squares fit of :math:`u_y = (\\lambda - 1) Y`.

    Parameters
    ----------
    model
        The model.
    fieldOutputController
        The field outputs, with the displacement of every particle.

    Returns
    -------
    dict
        The state.
    """

    displacement = np.array(fieldOutputController.fieldOutputs["displacement"].getLastResult())[:, :2]
    centers = np.array([particle.getCenterCoordinates() for particle in model.particles.values()])
    constraintsOf = {
        side: [constraint for name, constraint in model.constraints.items() if name.startswith(side)]
        for side in ("left", "bottom", "right")
    }
    referenceCoordinates = centers - displacement
    facesAtTheWall = [particle for particles in model.surfaces["slab_right"].values() for particle in particles]
    return dict(
        leftReaction=sum(constraint.reactionForce[0] for constraint in constraintsOf["left"]),
        floorReaction=sum(constraint.reactionForce[1] for constraint in constraintsOf["bottom"]),
        wallForce=sum(constraint.reactionForce for constraint in constraintsOf["right"]),
        largestWallForceOfAFace=max(constraint.reactionForce for constraint in constraintsOf["right"]),
        xOfTheFacesAtTheWall=np.array([particle.getFaceCoordinates(2)[0] for particle in facesAtTheWall]),
        displacement=displacement,
        referenceCoordinates=referenceCoordinates,
        verticalStretch=1.0 + np.linalg.lstsq(referenceCoordinates[:, 1:2], displacement[:, 1], rcond=None)[0][0],
    )


def run_sim():
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
            "Displacement/SQCNIxNSNI/PlaneStrain/Quad",
            number,
            vertexCoordinates,
            0.0,
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
    # We need this model to build the discrete system
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
        penaltyParameter=WALL_PENALTY,
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
    ensightOutput.createPerElementOutput(fieldOutputController.fieldOutputs["acceleration"], name="acceleration")
    ensightOutput.createPerElementOutput(fieldOutputController.fieldOutputs["velocity"], name="velocity")
    ensightOutput.createPerNodeOutput(
        fieldOutputController.fieldOutputs["vertex displacements"], name="vertex displacements"
    )
    ensightOutput.initializeJob()

    incSize = 1e-1
    adaptiveTimeStepper = AdaptiveTimeStepper(
        0.0, 1.0, incSize, incSize, incSize / 1e8, 5000, theJournal, increaseFactor=1.5
    )

    nonlinearSolver = NonlinearQuasistaticSolver(theJournal)

    iterationOptions = dict()

    iterationOptions["max. iterations"] = 10
    iterationOptions["critical iterations"] = 7
    iterationOptions["allowed residual growths"] = 2
    iterationOptions["default relative flux residual tolerance"] = 1e-6
    iterationOptions["default absolute flux residual tolerance"] = 1e-12
    iterationOptions["default absolute field correction tolerance"] = 1e-9

    linearSolver = getLinSolverByName("pardiso", {})

    try:
        # push the left face 3 to the right, against the wall
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

        compressed = stateOfTheSlab(theModel, fieldOutputController)

        # pull it back by 6, away from the wall
        adaptiveTimeStepper = AdaptiveTimeStepper(
            theModel.time,
            1.0,
            incSize,
            incSize,
            incSize / 1e8,
            5000,
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

        released = stateOfTheSlab(theModel, fieldOutputController)

    finally:
        fieldOutputController.finalizeJob()
        ensightOutput.finalizeJob()

        prettytable = performancetiming.makePrettyTable()
        prettytable.min_table_width = theJournal.linewidth
        theJournal.printPrettyTable(prettytable, "Summary")

    return compressed, released


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    """No matter where pytest is ran, we set the working dir
    to this testscript's parent directory"""

    monkeypatch.chdir(request.fspath.dirname)


def test_sim(assert_gold):

    # disable plots and suppress warnings
    import matplotlib

    matplotlib.use("Agg")
    import warnings

    warnings.filterwarnings("ignore")

    compressed, released = run_sim()

    # pushed against the wall: the wall pushes back with the force of the left face, the frictionless floor carries
    # nothing
    wallForce = compressed["wallForce"]
    assert wallForce > 0.0
    assert compressed["leftReaction"] == pytest.approx(-wallForce, rel=1e-10)
    assert abs(compressed["floorReaction"]) < 1e-10 * wallForce
    # the faces at the wall enter it by their force over the penalty, and no further
    penetration = compressed["xOfTheFacesAtTheWall"] - WALL
    assert penetration.min() > 0.0
    assert penetration.max() <= compressed["largestWallForceOfAFace"] / WALL_PENALTY * (1 + 1e-6)

    # compressed uniformly from 6 to 4, between x = 3 and the wall, and spread uniformly upwards: u_y = (stretch - 1) Y;
    # to the accuracy of the particles
    X, u = compressed["referenceCoordinates"], compressed["displacement"]
    np.testing.assert_allclose(u[:, 0], 3.0 - X[:, 0] / 3.0, rtol=0, atol=5e-3)
    verticalStretch = compressed["verticalStretch"]
    np.testing.assert_allclose(u[:, 1], (verticalStretch - 1.0) * X[:, 1], rtol=0, atol=5e-3)

    # pulled back: the slab has left the wall and is at rest without stress, moved 3 to the left
    assert abs(released["wallForce"]) < 1e-10 * wallForce
    assert abs(released["leftReaction"]) < 1e-10 * wallForce
    assert released["xOfTheFacesAtTheWall"].max() < WALL
    np.testing.assert_allclose(released["displacement"] - [-3.0, 0.0], 0.0, rtol=0, atol=2e-3)

    assert_gold(np.array([wallForce, verticalStretch]), np.loadtxt("gold.csv"), rtol=GOLD_TOLERANCE, atol=0.0)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--create-gold", dest="create_gold", action="store_true", help="create the gold file.")
    args = parser.parse_args()

    compressed, _ = run_sim()
    print(f"force of the wall {compressed['wallForce']:.10g}, vertical stretch {compressed['verticalStretch']:.10g}")

    if args.create_gold:
        np.savetxt("gold.csv", [compressed["wallForce"], compressed["verticalStretch"]])
