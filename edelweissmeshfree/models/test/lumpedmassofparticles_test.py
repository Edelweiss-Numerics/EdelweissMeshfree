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
"""The one assembly of the lumped mass of the particles gives a node without inertia zero: a node whose mass is not
positive, or positive only at round-off, as for a kernel function that the particles reach only where it vanishes. The
explicit solver divides by this mass, and the implicit dynamic solver integrates the nodes it is positive at."""

from types import SimpleNamespace

import numpy as np
from edelweissfe.numerics.dofvector import DofVector
from edelweissfe.sets.nodeset import NodeSet

from edelweissmeshfree.models.mpmdiscretesystem import (
    MASS_AT_ROUND_OFF,
    MPMDiscreteSystem,
)


class _ParticleOfMass:
    """A stand-in for a particle: it reports the given lumped mass at the one degree of freedom of its node."""

    def __init__(self, mass: float):
        self.mass = mass

    def computeLumpedInertia(self, M: np.ndarray):
        M += self.mass


class _LocationsOfOneDofPerParticle:
    """A stand-in for the locations of the particles: one degree of freedom per particle, in their order."""

    def __init__(self, particles: list):
        self.particles = particles
        self.cells = []
        self.cellElements = []
        self.ofEntity = {particle: np.array([index]) for index, particle in enumerate(particles)}

    def constructDofVector(self) -> DofVector:
        return DofVector(len(self.ofEntity), self)


def test_a_node_without_inertia_gets_no_mass():
    masses = [1.0, 0.1 * MASS_AT_ROUND_OFF, -0.1, 0.25, 10.0 * MASS_AT_ROUND_OFF]
    particles = {number: _ParticleOfMass(mass) for number, mass in enumerate(masses)}
    model = SimpleNamespace(materialPoints={}, particles=particles, cellElements={})
    system = MPMDiscreteSystem(
        model,
        _LocationsOfOneDofPerParticle(list(particles.values())),
        NodeSet("nodes with persistent field values", []),
        NodeSet("nodes with volatile field values", []),
        None,
    )

    M = system.assembleLumpedMassOfEntitiesWithOwnInertia(lambda entities, work: work(entities))

    np.testing.assert_array_equal(M, [1.0, 0.0, 0.0, 0.25, 10.0 * MASS_AT_ROUND_OFF])
