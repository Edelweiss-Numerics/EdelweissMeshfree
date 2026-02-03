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

import typing

import numpy as np
from edelweissfe.journal.journal import Journal
from edelweissfe.surfaces.entitybasedsurface import EntityBasedSurface

from edelweissmeshfree.models.mpmmodel import MPMModel
from edelweissmeshfree.particles.base.baseparticle import BaseParticle
from edelweissmeshfree.sets.particleset import ParticleSet


def generateBoxHexaParticleGrid(
    model: MPMModel,
    journal: Journal,
    particleFactoryCallback: typing.Callable[[np.ndarray, float], BaseParticle],
    name: str = "box_grid",
    x0: float = 0.0,
    y0: float = 0.0,
    z0: float = 0.0,
    h: float = 1.0,
    l: float = 1.0,
    t: float = 1.0,
    nX: int = 10,
    nY: int = 10,
    nZ: int = 10,
    firstParticleNumber: int = 1,
):
    """Generate a structured box hexahedral particle grid and add it to the model.

    Returns
    -------
    MPMModel
        The updated model.
    """

    nVerticesX = nX + 1
    nVerticesY = nY + 1
    nVerticesZ = nZ + 1

    grid = np.mgrid[
        x0 : x0 + l : nVerticesX * 1j,
        y0 : y0 + h : nVerticesY * 1j,
        z0 : z0 + t : nVerticesZ * 1j,
    ]

    vertices = []
    # currentNodeLabel = 1

    for x in range(nVerticesX):
        for y in range(nVerticesY):
            for z in range(nVerticesZ):
                vertex = grid[:, x, y, z]
                vertices.append(vertex)

    nG = np.asarray(vertices).reshape(nVerticesX, nVerticesY, nVerticesZ, -1)

    currentParticleNumber = firstParticleNumber
    particles = []

    pVolume = (l / nX) * (h / nY) * (t / nZ)

    # check if particle numbers are already used
    for pN in range(firstParticleNumber, firstParticleNumber + nX * nY * nZ):
        if pN in model.particles:
            raise ValueError("Particle number {:} already in use.".format(pN))

    for x in range(nX):
        for y in range(nY):
            for z in range(nZ):
                # particleVertices = np.asarray([nG[x, y], nG[x + 1, y], nG[x + 1, y + 1], nG[x, y + 1]])
                particleVertices = np.asarray(
                    [
                        nG[x, y, z],
                        nG[x + 1, y, z],
                        nG[x + 1, y + 1, z],
                        nG[x, y + 1, z],
                        nG[x, y, z + 1],
                        nG[x + 1, y, z + 1],
                        nG[x + 1, y + 1, z + 1],
                        nG[x, y + 1, z + 1],
                    ]
                )

                particle = particleFactoryCallback(currentParticleNumber, particleVertices, pVolume)
                model.particles[currentParticleNumber] = particle
                particles.append(particle)
                currentParticleNumber += 1

    # ... grid generation and particle creation remain the same ...
    # (The particleVertices block is correct for a Z-up Hex)

    particleGrid = np.asarray(particles).reshape(nX, nY, nZ)

    model.particleSets["{:}_all".format(name)] = ParticleSet("{:}_all".format(name), particleGrid.flatten())

    # --- GEOMETRIC CORRECTION: AXIS ALIGNMENT ---
    # Since connectivity is Z-extruded:
    # Left/Right is X-axis (index 0)
    # Front/Back is Y-axis (index 1)
    # Bottom/Top is Z-axis (index 2)

    model.particleSets["{:}_left".format(name)] = ParticleSet("{:}_left".format(name), particleGrid[0, :, :].flatten())
    model.particleSets["{:}_right".format(name)] = ParticleSet(
        "{:}_right".format(name), particleGrid[-1, :, :].flatten()
    )
    model.particleSets["{:}_front".format(name)] = ParticleSet(
        "{:}_front".format(name), particleGrid[:, 0, :].flatten()
    )
    model.particleSets["{:}_back".format(name)] = ParticleSet("{:}_back".format(name), particleGrid[:, -1, :].flatten())
    model.particleSets["{:}_bottom".format(name)] = ParticleSet(
        "{:}_bottom".format(name), particleGrid[:, :, 0].flatten()
    )
    model.particleSets["{:}_top".format(name)] = ParticleSet("{:}_top".format(name), particleGrid[:, :, -1].flatten())

    # --- ID CORRECTION: ABAQUS STANDARD ---

    # Bottom (-Z) -> S1 (ID 1)
    model.surfaces["{:}_bottom".format(name)] = EntityBasedSurface(
        "{:}_bottom".format(name), {1: list(np.ravel(particleGrid[:, :, 0]))}
    )

    # Top (+Z) -> S2 (ID 2)
    model.surfaces["{:}_top".format(name)] = EntityBasedSurface(
        "{:}_top".format(name), {2: list(np.ravel(particleGrid[:, :, -1]))}
    )

    # Right (+X) -> S4 (ID 4)
    model.surfaces["{:}_right".format(name)] = EntityBasedSurface(
        "{:}_right".format(name), {4: list(np.ravel(particleGrid[-1, :, :]))}
    )

    # Left (-X) -> S6 (ID 6)
    model.surfaces["{:}_left".format(name)] = EntityBasedSurface(
        "{:}_left".format(name), {6: list(np.ravel(particleGrid[0, :, :]))}
    )

    # Front (-Y) -> S3 (ID 3)
    model.surfaces["{:}_front".format(name)] = EntityBasedSurface(
        "{:}_front".format(name), {3: list(np.ravel(particleGrid[:, 0, :]))}
    )

    # Back (+Y) -> S5 (ID 5)
    model.surfaces["{:}_back".format(name)] = EntityBasedSurface(
        "{:}_back".format(name), {5: list(np.ravel(particleGrid[:, -1, :]))}
    )

    return model
