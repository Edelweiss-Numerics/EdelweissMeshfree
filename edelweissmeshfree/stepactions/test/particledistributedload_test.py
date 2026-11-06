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
"""The particles a distributed load acts on are those of every face of its surface, each of them once: a load on the
top and the right faces of a grid of particles acts on the particles of both rows, and on their common corner particle
once."""

import numpy as np
from edelweissfe.journal.journal import Journal
from edelweissfe.surfaces.entitybasedsurface import EntityBasedSurface

from edelweissmeshfree.generators.rectangularquadparticlegridgenerator import (
    generateRectangularQuadParticleGrid,
)
from edelweissmeshfree.models.mpmmodel import MPMModel
from edelweissmeshfree.stepactions.particledistributedload import (
    ParticleDistributedLoad,
)


def _particleGrid(journal: Journal) -> MPMModel:
    """A model with a grid of three by two particles, ``grid``."""

    from edelweissmeshfree.meshfree.approximations.marmot.marmotmeshfreeapproximation import (
        MarmotMeshfreeApproximationWrapper,
    )
    from edelweissmeshfree.particles.marmot.marmotparticlewrapper import (
        MarmotParticleWrapper,
    )

    approximation = MarmotMeshfreeApproximationWrapper("ReproducingKernelImplicitGradient", 2, completenessOrder=1)
    # K, G, density
    material = {"material": "CompressibleNeoHooke", "properties": np.array([1000.0, 500.0, 1.0])}

    def particleFactory(number, vertexCoordinates, volume):
        # the particle computes its volume from its vertices
        return MarmotParticleWrapper(
            "DisplacementSQCNIxNSNI/PlaneStrain/Quad", number, vertexCoordinates, 0.0, approximation, material
        )

    model = MPMModel(2)
    generateRectangularQuadParticleGrid(model, journal, particleFactory, name="grid", l=3.0, h=2.0, nX=3, nY=2)

    return model


def test_the_particles_of_a_distributed_load_are_those_of_every_face_of_its_surface():
    journal = Journal(verbose=False)
    model = _particleGrid(journal)

    topParticles = model.surfaces["grid_top"][3]
    rightParticles = model.surfaces["grid_right"][2]
    surface = EntityBasedSurface("topAndRight", {3: topParticles, 2: rightParticles})

    load = ParticleDistributedLoad("pressure", model, journal, surface, "pressure", np.array([1.0]))

    assert list(load.particles) == list(dict.fromkeys([*topParticles, *rightParticles]))
    assert len(load.particles) == len(topParticles) + len(rightParticles) - 1
