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
"""A strip of RKPM particles, held at its left end and moved at its right end, with Lagrange multipliers.

One ParticleLagrangianWeakDirichlet per particle constrains the ends: on the left end the bottom left vertex of every
particle, on the right end the center of every particle. Nothing else acts on the strip, so the reaction forces of the
two ends -- the sums of the multipliers of their constraints -- balance. The displacements of the particles and the
reaction forces are compared with the gold files.
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
from edelweissmeshfree.particlemanagers.kdbinorganizedparticlemanager import (
    KDBinOrganizedParticleManager,
)
from edelweissmeshfree.particles.marmot.marmotparticlewrapper import (
    MarmotParticleWrapper,
)
from edelweissmeshfree.solvers.deprecated.nqs import NonlinearQuasistaticSolver

#: The left end is held in place, ...
LEFT = {0: 0.0, 1: 0.0}
#: ... and the right end is moved.
RIGHT = {0: 0.4, 1: -0.2}


def constrainEnds(model: MPMModel) -> tuple[list, list]:
    """One constraint per particle of an end: at the bottom left vertex on the left end, at the center on the right end.

    Parameters
    ----------
    model
        The model with the particles.

    Returns
    -------
    tuple[list, list]
        The constraints of the left end, and those of the right end.
    """

    left = ParticleLagrangianWeakDirichletOnParticleSetFactory(
        "left", model.particleSets["rectangular_grid_left"], "displacement", LEFT, model, location="vertex", vertexID=0
    )
    right = ParticleLagrangianWeakDirichletOnParticleSetFactory(
        "right", model.particleSets["rectangular_grid_right"], "displacement", RIGHT, model, location="center"
    )

    return list(left.values()), list(right.values())


def run_sim() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deform the strip.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        The displacements of the particles, and the reaction forces of the left and of the right end.
    """

    dimension = 2

    theJournal = Journal()

    theModel = MPMModel(dimension)

    x0 = -1
    y0 = -1
    height = 1
    length = 4
    nX = 4 * 4
    nY = 1 * 4
    supportRadius = 0.5

    def theMeshfreeKernelFunctionFactory(node):
        return MarmotMeshfreeKernelFunctionWrapper(node, "BSplineBoxed", supportRadius=supportRadius, continuityOrder=3)

    theModel = generateRectangularKernelFunctionGrid(
        theModel, theJournal, theMeshfreeKernelFunctionFactory, x0=x0, y0=y0, h=height, l=length, nX=nX, nY=nY
    )

    theApproximation = MarmotMeshfreeApproximationWrapper(
        "ReproducingKernelImplicitGradient", dimension, completenessOrder=1
    )

    E = 20000
    nu = 0.3
    K = E / (3 * (1 - 2 * nu))
    G = E / (2 * (1 + nu))
    theMaterial = {
        "material": "FiniteStrainJ2Plasticity",
        "properties": np.array([K, G, 20e10, 20e10, 0, 0, 1, 1e-9]),
    }

    def TheParticleFactory(number, vertexCoordinates, volume):
        return MarmotParticleWrapper(
            "DisplacementSQCNIxSDI/PlaneStrain/Quad",
            number,
            vertexCoordinates,
            volume,
            theApproximation,
            theMaterial,
        )

    theModel = generateRectangularQuadParticleGrid(
        theModel, theJournal, TheParticleFactory, x0=x0, y0=y0, h=height, l=length, nX=nX, nY=nY
    )

    theParticleKernelDomain = ParticleKernelDomain(
        list(theModel.particles.values()), list(theModel.meshfreeKernelFunctions.values())
    )

    theParticleManager = KDBinOrganizedParticleManager(
        theParticleKernelDomain, dimension, theJournal, bondParticlesToKernelFunctions=True
    )

    theModel.particleKernelDomains["my_all_with_all"] = theParticleKernelDomain

    # the constraints are part of the model before it is prepared, which gives them their multipliers
    leftConstraints, rightConstraints = constrainEnds(theModel)
    for constraint in leftConstraints + rightConstraints:
        theModel.constraints[constraint.name] = constraint

    theModel.prepareYourself(theJournal)
    theJournal.printPrettyTable(theModel.makePrettyTableSummary(), "summary")

    fieldOutputController = MPMFieldOutputController(theModel, theJournal)

    fieldOutputController.addPerParticleFieldOutput(
        "displacement",
        theModel.particleSets["all"],
        "displacement",
    )

    fieldOutputController.initializeJob()

    incSize = 0.25
    adaptiveTimeStepper = AdaptiveTimeStepper(0.0, 1.0, incSize, incSize, incSize, 20, theJournal)

    nonlinearSolver = NonlinearQuasistaticSolver(theJournal)

    iterationOptions = {
        "default relative flux residual tolerance": 1e-8,
        "default relative flux residual tolerance alt.": 1e-3,
        "default relative field correction tolerance": 1e-8,
        "default absolute flux residual tolerance": 1e-14,
        "default absolute field correction tolerance": 1e-14,
    }

    linearSolver = getLinSolverByName("pardiso", {})

    try:
        nonlinearSolver.solveStep(
            adaptiveTimeStepper,
            linearSolver,
            theModel,
            fieldOutputController,
            particleManagers=[theParticleManager],
            constraints=theModel.constraints.values(),
            userIterationOptions=iterationOptions,
        )

    finally:
        fieldOutputController.finalizeJob()

        prettytable = performancetiming.makePrettyTable()
        prettytable.min_table_width = theJournal.linewidth
        theJournal.printPrettyTable(prettytable, "Summary")

    displacement = fieldOutputController.fieldOutputs["displacement"].getLastResult()
    leftReaction = np.sum([constraint.reactionForce for constraint in leftConstraints], axis=0)
    rightReaction = np.sum([constraint.reactionForce for constraint in rightConstraints], axis=0)

    return displacement, leftReaction, rightReaction


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

    displacement, leftReaction, rightReaction = run_sim()

    # only the two ends act on the strip: their reactions balance, to the tolerance of the Newton iterations
    assert np.linalg.norm(rightReaction) > 0.0, "the balance is vacuous without a reaction"
    assert_gold(leftReaction, -rightReaction, rtol=1e-6, atol=0.0)

    assert_gold(displacement, np.loadtxt("gold.csv"), atol=1e-8)
    assert_gold(np.array([leftReaction, rightReaction]), np.loadtxt("gold_reactions.csv"), rtol=1e-6, atol=0.0)


if __name__ == "__main__":
    displacement, leftReaction, rightReaction = run_sim()
    print("reaction forces: left end {:}, right end {:}".format(leftReaction, rightReaction))

    parser = argparse.ArgumentParser()
    parser.add_argument("--create-gold", dest="create_gold", action="store_true", help="create the gold files.")
    args = parser.parse_args()

    if args.create_gold:
        np.savetxt("gold.csv", displacement.flatten())
        np.savetxt("gold_reactions.csv", np.array([leftReaction, rightReaction]))
