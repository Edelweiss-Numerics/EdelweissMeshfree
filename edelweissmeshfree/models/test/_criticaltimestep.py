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
"""The critical time step of the particles' explicit scheme for the tests: the estimate the discrete system gives
(:meth:`~edelweissmeshfree.models.mpmdiscretesystem.MPMDiscreteSystem.criticalTimeStepOfEntitiesWithOwnInertia`)
and the exact one, from the eigenvalues of the assembled tangent."""

import numpy as np

from edelweissmeshfree.models.mpmdiscretesystem import (
    CriticalTimeStepBoundOfEachFieldBlock,
)


def serially(entities, work):
    """The loop over the chunks of independent entities, in one chunk."""

    work(entities)


def estimateOfTheSystem(system, weightsOfTheBound: np.ndarray | None = None) -> CriticalTimeStepBoundOfEachFieldBlock:
    """The estimate of the particles' critical time step of a discrete system, from the lumped mass and damping it
    assembles, with the given carried weights (None: without)."""

    return system.criticalTimeStepOfEntitiesWithOwnInertia(
        system.assembleLumpedMassOfEntitiesWithOwnInertia(serially),
        system.assembleLumpedDampingOfEntitiesWithOwnInertia(serially),
        serially,
        weightsOfTheBound,
    )


def exactCriticalTimeStep(model) -> float:
    """The critical time step of the explicit scheme for the particles' stiffness of the model, connected anew
    (:func:`exactCriticalTimeStepOfTheSystem`)."""

    model.updateConnectivity()
    return exactCriticalTimeStepOfTheSystem(model.buildDiscreteSystem())


def exactCriticalTimeStepOfTheSystem(system) -> float:
    """The critical time step of the explicit scheme for the particles' stiffness of a discrete system as it is, from
    the eigenvalues of the assembled system: :math:`2/\\omega_\\mathrm{max}` of the central differences on the degrees
    of freedom with an inertia, whatever their damping, and :math:`2/\\lambda_\\mathrm{max}` of forward Euler on those
    with a damping alone."""

    M = np.asarray(system.assembleLumpedMassOfEntitiesWithOwnInertia(serially))
    C = np.asarray(system.assembleLumpedDampingOfEntitiesWithOwnInertia(serially))
    K = np.zeros((len(M), len(M)))
    for particle in system.particles:
        nDof = particle.nDof
        stiffness = np.zeros(nDof * nDof)
        particle.assembleStiffness(stiffness)
        indices = np.asarray(system.locations.ofEntity[particle])
        K[np.ix_(indices, indices)] += stiffness.reshape(nDof, nDof, order="F")

    stable = np.inf
    for ofTheScheme, divisor, stableOfTheLargestEigenvalue in (
        (M > 0.0, M, lambda largest: 2.0 / np.sqrt(largest)),
        ((M == 0.0) & (C > 0.0), C, lambda largest: 2.0 / largest),
    ):
        if ofTheScheme.any():
            scaling = 1.0 / np.sqrt(divisor[ofTheScheme])
            S = scaling[:, None] * K[np.ix_(ofTheScheme, ofTheScheme)] * scaling[None, :]
            stable = min(stable, stableOfTheLargestEigenvalue(np.max(np.abs(np.linalg.eigvals(S)))))
    return stable
