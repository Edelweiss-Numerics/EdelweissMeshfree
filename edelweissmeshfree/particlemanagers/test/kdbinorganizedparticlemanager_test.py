# -*- coding: utf-8 -*-
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
"""The search of the KD-bin particle manager: every particle gets the kernel functions that cover one of its evaluation
points, in the order of their labels, whatever the number of threads. A particle whose kernel functions changed is
assigned them and reported; every other particle only updates its shape functions. A kernel function whose support is
smaller than its box covers only what its support holds."""

from types import SimpleNamespace

import numpy as np
import pytest
from edelweissfe.journal.journal import Journal
from edelweissfe.numerics.parallelizationutilities import isFreeThreadingSupported

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
    """A particle as the manager sees it: its evaluation points, and the kernel functions assigned to it; the
    assignments and the updates of its shape functions are counted."""

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


# with more than one thread only where the threads run in parallel: without free threading, the search runs serially
_threads = [
    "1",
    pytest.param("4", marks=pytest.mark.skipif(not isFreeThreadingSupported(), reason="no free threading")),
]


def _searchedOnce(monkeypatch, threads: str, hasBoxSupport: bool, **options):
    monkeypatch.setenv("OMP_NUM_THREADS", threads)
    rng = np.random.default_rng(1)
    kernels, particles = _model(rng, hasBoxSupport)
    manager = KDBinOrganizedParticleManager(
        ParticleKernelDomain(particles, kernels), 2, Journal(verbose=False), **options
    )
    hasChanged = manager.updateConnectivity()
    return rng, kernels, particles, manager, hasChanged


@pytest.mark.parametrize("threads", _threads)
@pytest.mark.parametrize("hasBoxSupport", [True, False])
def test_the_first_search_assigns_every_particle(monkeypatch, threads, hasBoxSupport):
    _, kernels, particles, manager, hasChanged = _searchedOnce(monkeypatch, threads, hasBoxSupport)

    assert hasChanged
    assert manager.particlesWithChangedKernelFunctions == particles
    for particle in particles:
        assert particle.kernelFunctions == _expected(kernels, particle)
        assert particle.assignments == 1 and particle.shapeFunctionUpdates == 0


@pytest.mark.parametrize("threads", _threads)
@pytest.mark.parametrize("hasBoxSupport", [True, False])
def test_without_motion_only_the_shape_functions_are_updated(monkeypatch, threads, hasBoxSupport):
    _, _, particles, manager, _ = _searchedOnce(monkeypatch, threads, hasBoxSupport)

    assert not manager.updateConnectivity()
    assert manager.particlesWithChangedKernelFunctions == []
    assert all(particle.assignments == 1 and particle.shapeFunctionUpdates == 1 for particle in particles)


@pytest.mark.parametrize("threads", _threads)
@pytest.mark.parametrize("hasBoxSupport", [True, False])
def test_after_motion_the_changed_particles_are_reported(monkeypatch, threads, hasBoxSupport):
    rng, kernels, particles, manager, _ = _searchedOnce(monkeypatch, threads, hasBoxSupport)

    for particle in particles[3::7]:
        particle.points += rng.uniform(-0.8, 0.8, 2)
    before = [list(particle.kernelFunctions) for particle in particles]

    assert manager.updateConnectivity()
    for particle in particles:
        assert particle.kernelFunctions == _expected(kernels, particle)
    changed = [
        particle for particle, kernelsBefore in zip(particles, before) if particle.kernelFunctions != kernelsBefore
    ]
    assert changed and manager.particlesWithChangedKernelFunctions == changed
    for particle in particles:
        assert particle.assignments + particle.shapeFunctionUpdates == 2
        assert particle.assignments == (2 if particle in changed else 1)


def test_with_a_skin_and_without_motion_only_the_shape_functions_are_updated(monkeypatch):
    _, _, particles, manager, _ = _searchedOnce(monkeypatch, "1", True, neighbourListSkinFraction=0.1)

    assert not manager.updateConnectivity()
    assert manager.particlesWithChangedKernelFunctions == []
    assert all(particle.assignments == 1 and particle.shapeFunctionUpdates == 1 for particle in particles)


def test_a_particle_beyond_every_support_is_reported():
    kernels, particles = _model(np.random.default_rng(2))
    particles[5].points += 100.0
    manager = KDBinOrganizedParticleManager(ParticleKernelDomain(particles, kernels), 2, Journal(verbose=False))

    with pytest.raises(ValueError, match="no associated kernel functions"):
        manager.updateConnectivity()
