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
"""A symmetric particle projectile falls onto an analytic wall: once the whole of it, once only its right half, held on
the symmetry plane by a support
(:class:`~edelweissmeshfree.constraints.particlepenaltyweakdirichtlet.ParticlePenaltyWeakDirichlet` holding the
horizontal displacement of the vertices on the plane at zero). The half projectile reproduces the right half of the
whole one; without the support, it spreads about its own center instead.

The support has to be stiff against the material it holds: a penalty of the wall's (1e5 per point, about twice the
stiffness of a particle) lets the plane give way and spring back, and the half projectile is then 10 % off the whole
one, whatever the particle size, since the stiffness of a particle in plane strain does not change with its size. With
1e6, below the stable limit of the time increment, the deviation is a few %: the largest deviation is 0.26 of a
largest displacement of 6.6 here. A refinement study found it shrinking with the particle size, 0.28, 0.090 and 0.033
for particles of 10, 5 and 2.5 mm, when the support still distributed its force with the shape functions at the
vertices; it now distributes it with the vector that moves the vertices, which makes it conservative.
"""

import argparse

import numpy as np
import pytest
from edelweissfe.journal.journal import Journal
from edelweissfe.solvers.nonlinearexplicitdynamicparallel import NEDParallel
from edelweissfe.steps.adaptivestepforexplicitsimulations import (
    AdaptiveStepForExplicitSimulations,
)
from edelweissfe.steps.stepmanager import StepActionCollection

from edelweissmeshfree.constraints.explicit.particlepenaltycartesianboundaryexplicit import (
    ParticleExplicitPenaltyCartesianBoundaryConstraintFactory,
)
from edelweissmeshfree.constraints.particlepenaltyweakdirichtlet import (
    ParticlePenaltyWeakDirichlet,
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

PENALTY = 1e5
#: The penalty of the support on the symmetry plane, see the module documentation.
SUPPORT_PENALTY = 1e6
PARTICLE_SIZE = 10.0
HALF_WIDTH = 50.0
HEIGHT = 100.0
VELOCITY = -400e3


def run_sim(model: str) -> np.ndarray:
    """Let the projectile fall onto the wall.

    Parameters
    ----------
    model
        "whole" for the whole projectile, "half" for its right half on the support, "half without support" for its
        right half with a free symmetry plane.

    Returns
    -------
    numpy.ndarray
        The displacements of the particles, one row per particle, column after column from the left.
    """

    dimension = 2
    theJournal = Journal(verbose=False)
    theModel = MPMModel(dimension)

    x0 = -HALF_WIDTH if model == "whole" else 0.0
    length = 2 * HALF_WIDTH if model == "whole" else HALF_WIDTH
    nX, nY = int(length / PARTICLE_SIZE), int(HEIGHT / PARTICLE_SIZE)
    grid = dict(x0=x0, y0=1.0, h=HEIGHT, l=length, nX=nX, nY=nY, name="projectile")

    def theKernelFunctionFactory(node):
        return MarmotMeshfreeKernelFunctionWrapper(
            node, "BSplineBoxed", supportRadius=3.0 * PARTICLE_SIZE, continuityOrder=3
        )

    generateRectangularKernelFunctionGrid(theModel, theJournal, theKernelFunctionFactory, **grid)

    theApproximation = MarmotMeshfreeApproximationWrapper("ReproducingKernel", dimension, completenessOrder=1)
    E, nu = 78200.0, 0.3
    K, G = E / (3 * (1 - 2 * nu)), E / (2 * (1 + nu))
    theMaterial = {
        "material": "FiniteStrainJ2Plasticity",
        "properties": np.array([K, G, 290e10, 290e10, 1, 0, 1, 1.0e-8]),
    }

    def theParticleFactory(number, vertexCoordinates, volume):
        return MarmotParticleWrapper(
            "Displacement/SQCNIxNSNI/PlaneStrain/Quad", number, vertexCoordinates, 0.0, theApproximation, theMaterial
        )

    generateRectangularQuadParticleGrid(theModel, theJournal, theParticleFactory, **grid)

    theParticleKernelDomain = ParticleKernelDomain(
        list(theModel.particles.values()), list(theModel.meshfreeKernelFunctions.values())
    )
    theParticleManager = KDBinOrganizedParticleManager(
        theParticleKernelDomain, dimension, theJournal, bondParticlesToKernelFunctions=True
    )
    theModel.particleKernelDomains["projectile"] = theParticleKernelDomain

    theModel.constraints.update(
        ParticleExplicitPenaltyCartesianBoundaryConstraintFactory(
            "wall", 0.0, 1, theModel.particleSets["projectile_bottom"], "displacement", theModel, location="vertex",
            vertexIDs=[0, 1, 2, 3], penaltyParameter=PENALTY,
        )
    )  # fmt: skip

    if model == "half":
        # the vertices 0 and 3 of a particle in the left column are on the symmetry plane x = 0
        theModel.constraints["symmetry"] = ParticlePenaltyWeakDirichlet(
            "symmetry", theModel, list(theModel.particleSets["projectile_left"]), "displacement", {0: 0.0}, SUPPORT_PENALTY,
            constrain=[0, 3],
        )  # fmt: skip

    theModel.prepareYourself(theJournal)

    fieldOutputController = MPMFieldOutputController(theModel, theJournal)
    fieldOutputController.addPerParticleFieldOutput("displacement", theModel.particleSets["all"], "displacement")
    fieldOutputController.initializeJob()

    for particle in theModel.particleSets["all"]:
        particle.getResultArray("velocity")[1] = VELOCITY
        particle.acceptStateAndPosition()

    theModel.particleManagers.append(theParticleManager)
    incrementSize = 3e-3
    step = AdaptiveStepForExplicitSimulations(
        0,
        theModel,
        fieldOutputController,
        theJournal,
        {},
        NEDParallel({}, theJournal, **{"output-frequency": 1}),
        [],
        StepActionCollection(),
        stepLength=2.0e-4,
        startInc=incrementSize,
        maxInc=incrementSize,
        minInc=incrementSize / 1e8,
        maxNumInc=101,
    )

    try:
        step.solve()
    finally:
        fieldOutputController.finalizeJob()

    return fieldOutputController.fieldOutputs["displacement"].getLastResult().reshape(len(theModel.particles), -1)


def rightHalfOfTheWhole(uWhole: np.ndarray) -> np.ndarray:
    """The displacements of the particles of the whole projectile right of the symmetry plane, in the order of the half
    projectile's: the generator numbers the particles column after column from the left."""

    return uWhole[len(uWhole) // 2 :]


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    """No matter where pytest is ran, we set the working dir to this testscript's parent directory."""

    monkeypatch.chdir(request.fspath.dirname)


def deviations(uHalf: np.ndarray, uWhole: np.ndarray) -> tuple[float, float, float]:
    """How far a half projectile is from the right half of the whole one: the largest deviation of a displacement, and
    the deviations of the mean horizontal and the mean vertical displacement."""

    return (
        np.abs(uHalf - uWhole).max(),
        abs(uHalf[:, 0].mean() - uWhole[:, 0].mean()),
        abs(uHalf[:, 1].mean() - uWhole[:, 1].mean()),
    )


def test_sim(assert_gold):
    uWhole = rightHalfOfTheWhole(run_sim("whole"))
    uHalf = run_sim("half")
    uHalfWithoutSupport = run_sim("half without support")

    largest, meanHorizontal, meanVertical = deviations(uHalf, uWhole)
    largestWithoutSupport, meanHorizontalWithoutSupport, _ = deviations(uHalfWithoutSupport, uWhole)

    # Without the support, the half projectile spreads about its own center, not away from the plane: its mean
    # horizontal displacement is zero. The support restores it.
    assert meanHorizontal < 0.05 * meanHorizontalWithoutSupport
    assert largest < 0.15 * largestWithoutSupport
    assert meanVertical < 0.005 * abs(uWhole[:, 1].mean())

    assert_gold(uHalf.flatten(), np.loadtxt("gold.csv"), atol=1e-12)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--create-gold", dest="create_gold", action="store_true", help="create the gold file.")
    args = parser.parse_args()

    uWhole = rightHalfOfTheWhole(run_sim("whole"))
    uHalf = run_sim("half")
    uHalfWithoutSupport = run_sim("half without support")
    print("mean displacement of the whole: {:}".format(uWhole[:, :2].mean(axis=0)))
    print("half, deviations (largest, mean horizontal, mean vertical): {:}".format(deviations(uHalf, uWhole)))
    print("half without support, deviations: {:}".format(deviations(uHalfWithoutSupport, uWhole)))

    if args.create_gold:
        np.savetxt("gold.csv", uHalf.flatten())
