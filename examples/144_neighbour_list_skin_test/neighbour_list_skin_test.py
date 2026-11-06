# -*- coding: utf-8 -*-
#  ---------------------------------------------------------------------
#
#  Unit of Strength of Materials and Structural Analysis
#  University of Innsbruck,
#
#  Research Group for Computational Mechanics of Materials
#  Institute of Structural Engineering, BOKU University, Vienna
#
#  Matthias Neuner |  matthias.neuner@boku.ac.at
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
"""A neighbour list skin must not change the answer.

With a skin, the particle manager searches for the kernel functions that cover a particle *or come
within the skin of covering it*, and reuses that answer for as long as accumulated motion cannot have
carried anything across the remaining margin. The kernel functions this admits early evaluate to
exactly zero at the particle, and the reproducing-kernel reconstruction discards anything that does --
so the shape functions, and therefore the whole simulation, must come out exactly as they would have
without a skin.

That is a strong claim, so this test makes it the assertion: the same impact problem is run with the
skin off and on, and the two displacement fields have to be *equal*, not merely close.

The same impact problem checks the restart of the explicit solver: stopped after a few increments and resumed from the
restart checkpoint of the last one, it has to end in the very displacement field of the run that was not interrupted.
"""

import numpy as np
import pytest
from edelweissfe.journal.journal import Journal
from edelweissfe.outputmanagers.restart import OutputManager as RestartOutputManager
from edelweissfe.outputmanagers.restart import RestartOutputManagerSchema
from edelweissfe.steps.adaptivestep import AdaptiveStep
from edelweissfe.steps.stepmanager import StepActionCollection
from edelweissfe.utils.checkpoint import ResumeCheckpoint

from edelweissmeshfree.constraints.explicit.particlepenaltycartesianboundaryexplicit import (
    ParticleExplicitPenaltyCartesianBoundaryConstraintFactory,
)
from edelweissmeshfree.fieldoutput.fieldoutput import MPMFieldOutputController
from edelweissmeshfree.generators.boxhexaparticlegridgenerator import (
    generateBoxHexaParticleGrid,
)
from edelweissmeshfree.generators.kernelmatchingtoparticlegenerator import (
    generateKernelMatchingToParticle,
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
from edelweissmeshfree.solvers.meshfreeexplicitdynamic import MeshfreeExplicitDynamic

BAR_WIDTH = 4.0
BAR_LENGTH = 8.0
N_PARTICLES_ACROSS = 4
N_PARTICLES_ALONG = 8

INCREMENT_SIZE = 1e-4
N_INCREMENTS = 8


def run_sim(
    neighbourListSkinFraction: float = 0.0,
    restartBaseName: str = None,
    resumeFrom: str = None,
    maxNumInc: int = N_INCREMENTS,
):
    """Run a small impact problem and return the final displacement field.

    Parameters
    ----------
    neighbourListSkinFraction
        The skin handed to the particle manager, as a fraction of the smallest support half-width.
    restartBaseName
        The base name of the restart checkpoints written after every increment; None writes none.
    resumeFrom
        The restart checkpoint the step is resumed from; None for a fresh start.
    maxNumInc
        The increment cap, on which the step ends.

    Returns
    -------
    np.ndarray
        The displacement field at the end of the step.
    """

    dimension = 3

    theJournal = Journal(verbose=False)
    theModel = MPMModel(dimension)

    theApproximation = MarmotMeshfreeApproximationWrapper("ReproducingKernel", dimension, completenessOrder=1)

    timeScalingFactor = 5e-5
    theMaterial = {
        "material": "FiniteStrainJ2Plasticity",
        "properties": np.array([65166.7, 30076.9, 290.0, 290.0, 1.0, 0, 1, 2.7e-9 / timeScalingFactor**2]),
    }

    def theParticleFactory(number, vertexCoordinates, volume):
        # A nodally integrated particle derives its volume from its vertices and requires zero here.
        del volume

        return MarmotParticleWrapper(
            "Displacement/RS-SNNIxNSNI/3D/Hexa",
            number,
            vertexCoordinates,
            0.0,
            theApproximation,
            theMaterial,
        )

    theModel = generateBoxHexaParticleGrid(
        theModel,
        theJournal,
        theParticleFactory,
        name="bar",
        x0=-BAR_WIDTH / 2,
        y0=-BAR_WIDTH / 2,
        z0=-BAR_LENGTH / 2,
        l=BAR_WIDTH,
        h=BAR_WIDTH,
        t=BAR_LENGTH,
        nX=N_PARTICLES_ACROSS,
        nY=N_PARTICLES_ACROSS,
        nZ=N_PARTICLES_ALONG,
    )

    def theKernelFunctionFactory(node, characteristicLength):
        return MarmotMeshfreeKernelFunctionWrapper(
            node, "BSplineBoxed", supportRadius=characteristicLength, continuityOrder=3
        )

    theModel = generateKernelMatchingToParticle(
        theModel,
        theJournal,
        theKernelFunctionFactory,
        theModel.particleSets["bar_all"],
        supportScalingFactor=2.4,
    )

    theParticleKernelDomain = ParticleKernelDomain(
        list(theModel.particles.values()), list(theModel.meshfreeKernelFunctions.values())
    )
    theModel.particleKernelDomains["bar_with_bar"] = theParticleKernelDomain

    theParticleManager = KDBinOrganizedParticleManager(
        theParticleKernelDomain,
        dimension,
        theJournal,
        bondParticlesToKernelFunctions=True,
        neighbourListSkinFraction=neighbourListSkinFraction,
    )

    impactVelocity = -373e3 * timeScalingFactor
    for particle in theModel.particles.values():
        velocity = particle.getResultArray("velocity")
        velocity[2] = impactVelocity
        particle.acceptStateAndPosition()

    # Put the wall two increments of flight below the lowest particle centre, so that the run spends
    # most of its increments in contact rather than in free flight.
    particleHeight = BAR_LENGTH / N_PARTICLES_ALONG
    wallPositionZ = -BAR_LENGTH / 2 + particleHeight / 2 - 2 * abs(impactVelocity) * INCREMENT_SIZE

    theWallConstraints = ParticleExplicitPenaltyCartesianBoundaryConstraintFactory(
        "wall",
        wallPositionZ,
        2,
        theModel.particleSets["bar_all"],
        "displacement",
        theModel,
        location="center",
        doProximityCheck=True,
        penaltyParameter=1e5,
    )
    theModel.constraints.update(theWallConstraints)

    theModel.prepareYourself(theJournal)
    theModel.particleManagers.append(theParticleManager)

    # A resumed run is built as the checkpointed one, and then takes over the model's state from the checkpoint.
    resumeCheckpoint = None
    if resumeFrom is not None:
        resumeCheckpoint = ResumeCheckpoint(resumeFrom)
        resumeCheckpoint.restoreModel(theModel, theJournal)

    fieldOutputController = MPMFieldOutputController(theModel, theJournal)
    fieldOutputController.addPerParticleFieldOutput("displacement", theModel.particleSets["all"], "displacement")
    fieldOutputController.initializeJob()

    outputManagers = []
    if restartBaseName is not None:
        outputManagers.append(
            RestartOutputManager(
                "restart",
                theModel,
                fieldOutputController,
                theJournal,
                None,
                configuration=RestartOutputManagerSchema(writeInterval=1, baseName=restartBaseName),
            )
        )

    # The step ends on its increment cap, which is how it is meant to end.
    step = AdaptiveStep(
        0,
        theModel,
        fieldOutputController,
        theJournal,
        {},
        MeshfreeExplicitDynamic({}, theJournal),
        outputManagers,
        StepActionCollection(),
        startInc=INCREMENT_SIZE,
        maxInc=INCREMENT_SIZE,
        minInc=INCREMENT_SIZE / 100,
        maxNumInc=maxNumInc,
    )

    try:
        step.solve(resumeCheckpoint)
    finally:
        fieldOutputController.finalizeJob()

    return np.copy(fieldOutputController.fieldOutputs["displacement"].getLastResult())


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    """No matter where pytest is run, set the working dir to this test script's parent directory."""

    monkeypatch.chdir(request.fspath.dirname)


def test_sim():
    withoutSkin = run_sim(0.0)
    withSkin = run_sim(0.05)

    assert np.array_equal(withoutSkin, withSkin), (
        "A neighbour list skin changed the result. It must not: the kernel functions it admits early "
        "evaluate to zero at the particle and are discarded by the reconstruction."
    )


def test_restart(tmp_path):
    """Stopped after a few increments and resumed from the restart checkpoint of the last one, the step ends in the
    displacement field of the run that was not interrupted: the checkpoint carries the half-step velocity, the net
    force and the lumped mass the next increment starts from."""

    uninterrupted = run_sim()

    restartBaseName = str(tmp_path / "restart")
    run_sim(restartBaseName=restartBaseName, maxNumInc=N_INCREMENTS // 2)
    lastCheckpoint = max(tmp_path.glob("restart_*.h5"), key=lambda path: int(path.stem.split("_")[-1]))
    resumed = run_sim(restartBaseName=restartBaseName, resumeFrom=str(lastCheckpoint))

    assert np.array_equal(uninterrupted, resumed), "The resumed run did not continue the interrupted one."


if __name__ == "__main__":
    test_sim()
    print("neighbour list skin leaves the result unchanged")
