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
"""The search of the KD-bin particle manager: every particle gets the kernel functions that cover one of its evaluation
points, in the order of their labels, whatever the number of threads; the particles whose kernel functions changed
are reported, and every particle is assigned its kernel functions in every search (its shape functions are evaluated
anew); a kernel function whose support is smaller than its box covers only what its support holds."""

from types import SimpleNamespace

import numpy as np
import pytest
from edelweissfe.journal.journal import Journal

from edelweissmeshfree.meshfree.particlekerneldomain import ParticleKernelDomain
from edelweissmeshfree.particlemanagers.kdbinorganizedparticlemanager import (
    KDBinOrganizedParticleManager,
)


class _Kernel:
    """A kernel function as the manager sees it: a box support of half-width ``radius`` about its centre, or, without
    box support, a ball of that radius."""

    def __init__(self, label: int, centre: np.ndarray, radius: float, hasBoxSupport: bool = True):
        self.node = SimpleNamespace(label=label)
        self.center = np.array(centre, dtype=float)
        self.radius = radius
        self.hasBoxSupport = hasBoxSupport

    def getBoundingBox(self):
        return self.center - self.radius, self.center + self.radius

    def isAnyCoordinateInSupport(self, coordinates: np.ndarray) -> bool:
        return bool(np.any(np.linalg.norm(coordinates - self.center, axis=1) < self.radius))

    def covers(self, coordinates: np.ndarray) -> bool:
        if self.hasBoxSupport:
            return bool(np.any(np.all(np.abs(coordinates - self.center) < self.radius, axis=1)))
        return self.isAnyCoordinateInSupport(coordinates)


class _Particle:
    """A particle as the manager sees it: its evaluation points, and the kernel functions assigned to it, counted."""

    def __init__(self, points: np.ndarray):
        self.points = np.array(points, dtype=float)
        self.kernelFunctions = []
        self.assignments = 0
        self.shapeFunctionUpdates = 0

    def getEvaluationCoordinates(self) -> np.ndarray:
        return self.points

    def getCenterCoordinates(self) -> np.ndarray:
        return self.points.mean(axis=0)

    def assignKernelFunctions(self, kernelFunctions: list):
        self.kernelFunctions = kernelFunctions
        self.assignments += 1

    def updateShapeFunctions(self):
        self.shapeFunctionUpdates += 1


def _model(rng: np.random.Generator, hasBoxSupport: bool = True):
    """Kernel functions on a jittered grid, labelled in shuffled order, and particles of four evaluation points each."""

    grid = np.stack(np.meshgrid(np.arange(8.0), np.arange(6.0), indexing="ij"), axis=-1).reshape(-1, 2)
    labels = rng.permutation(len(grid)) + 100
    kernels = [
        _Kernel(label, centre + rng.uniform(-0.2, 0.2, 2), rng.uniform(1.2, 1.8), hasBoxSupport)
        for label, centre in zip(labels, grid)
    ]
    particles = [_Particle(rng.uniform(0.5, 6.5, 2) + rng.uniform(-0.3, 0.3, (4, 2))) for _ in range(60)]
    return kernels, particles


def _expected(kernels: list, particle: _Particle) -> list:
    return sorted([k for k in kernels if k.covers(particle.points)], key=lambda k: k.node.label)


@pytest.mark.parametrize("threads", ["1", "4"])
@pytest.mark.parametrize("hasBoxSupport", [True, False])
def test_every_particle_gets_the_kernels_that_cover_it_and_the_changed_ones_are_reported(
    monkeypatch, threads, hasBoxSupport
):
    monkeypatch.setenv("OMP_NUM_THREADS", threads)
    rng = np.random.default_rng(1)
    kernels, particles = _model(rng, hasBoxSupport)
    manager = KDBinOrganizedParticleManager(ParticleKernelDomain(particles, kernels), 2, Journal(verbose=False))

    assert manager.updateConnectivity()
    assert manager.particlesWithChangedKernelFunctions == particles, "the first search changes every particle"
    for particle in particles:
        assert particle.kernelFunctions == _expected(kernels, particle)

    assert not manager.updateConnectivity(), "nothing moved"
    assert manager.particlesWithChangedKernelFunctions == []
    assert all(
        particle.assignments == 1 and particle.shapeFunctionUpdates == 1 for particle in particles
    ), "the kernel functions are the same, the shape functions are evaluated anew"

    moved = particles[3::7]
    for particle in moved:
        particle.points += rng.uniform(-0.8, 0.8, 2)
    before = [list(particle.kernelFunctions) for particle in particles]

    manager.updateConnectivity()
    for particle, kernelsBefore in zip(particles, before):
        assert particle.kernelFunctions == _expected(kernels, particle)
    changed = [p for p, kernelsBefore in zip(particles, before) if p.kernelFunctions != kernelsBefore]
    assert changed and manager.particlesWithChangedKernelFunctions == changed


def test_a_particle_beyond_every_support_is_reported():
    kernels, particles = _model(np.random.default_rng(2))
    particles[5].points += 100.0
    manager = KDBinOrganizedParticleManager(ParticleKernelDomain(particles, kernels), 2, Journal(verbose=False))

    with pytest.raises(ValueError, match="no associated kernel functions"):
        manager.updateConnectivity()
