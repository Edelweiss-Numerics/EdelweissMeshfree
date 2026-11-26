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
"""Axisymmetric particles in an ``MPMModel``: a particle says whether it is axisymmetric (Marmot's section of the
particle), an axisymmetric model holds axisymmetric particles only and any other model none, a particle reaching
across the axis is refused, and the size of an axisymmetric particle is that of its section r-z, not of its ring.
Skipped where Marmot is not built."""

import numpy as np
import pytest
from edelweissfe.journal.journal import Journal

from edelweissmeshfree.materialpoints.pythonmaterialpoint.mp import PythonMaterialPoint
from edelweissmeshfree.meshfree.approximations.marmot.marmotmeshfreeapproximation import (
    MarmotMeshfreeApproximationWrapper,
)
from edelweissmeshfree.models.mpmmodel import MPMModel
from edelweissmeshfree.models.test._particleplate import H, particlePlate
from edelweissmeshfree.particles.base.baseparticle import characteristicLength
from edelweissmeshfree.particles.marmot.marmotparticlewrapper import (
    MarmotParticleWrapper,
)

#: A compressible neo-Hookean solid: bulk modulus, shear modulus, density.
NEO_HOOKE = {"material": "CompressibleNeoHooke", "properties": np.array([3500.0, 1500.0, 2.0])}
#: The edge of the square particles.
SIZE = 0.5


def _particle(particleType: str, r0: float, z0: float = 0.0, number: int = 1) -> MarmotParticleWrapper:
    """A square particle of the edge SIZE with its lower left vertex at (r0, z0)."""

    vertices = np.array([[r0, z0], [r0 + SIZE, z0], [r0 + SIZE, z0 + SIZE], [r0, z0 + SIZE]])
    approximation = MarmotMeshfreeApproximationWrapper("ReproducingKernel", 2, completenessOrder=1)
    return MarmotParticleWrapper(particleType, number, vertices, 0.0, approximation, NEO_HOOKE)


def test_a_particle_says_whether_it_is_axisymmetric():
    assert _particle("DisplacementSQCNI/Axisymmetric/Quad", 1.0).axisymmetric
    assert not _particle("DisplacementSQCNI/PlaneStrain/Quad", 1.0).axisymmetric


@pytest.mark.parametrize(
    "modelIsAxisymmetric, particleType",
    [(True, "DisplacementSQCNI/PlaneStrain/Quad"), (False, "DisplacementSQCNI/Axisymmetric/Quad")],
    ids=["plane particle in an axisymmetric model", "axisymmetric particle in a plane model"],
)
def test_a_particle_that_does_not_integrate_as_the_model_is_refused(modelIsAxisymmetric, particleType):
    model = MPMModel(2, axisymmetric=modelIsAxisymmetric)
    model.particles[1] = _particle(particleType, 1.0)

    with pytest.raises(ValueError, match="particle 1"):
        model.prepareYourself(Journal(verbose=False))


def test_an_axisymmetric_model_refuses_material_points():
    """The material points (and the cells and cell elements) of EdelweissMeshfree integrate over the section, not over
    the rings about the axis: an axisymmetric model refuses them."""

    model = MPMModel(2, axisymmetric=True)
    model.materialPoints[1] = PythonMaterialPoint("PlaneStrain", 1, np.array([1.0, 0.0]), 1.0, None)

    with pytest.raises(ValueError, match="1 material points"):
        model.prepareYourself(Journal(verbose=False))


def test_an_axisymmetric_particle_reaching_across_the_axis_is_refused_and_one_touching_it_is_not():
    """The section r-z of a body of revolution lies in r >= 0: a particle with a vertex at r < 0 is refused when it is
    made, as a Python error, not an end of the process; one with an edge on the axis is a particle like any other."""

    with pytest.raises(ValueError, match="r >= 0"):
        _particle("DisplacementSQCNI/Axisymmetric/Quad", -0.1)
    assert _particle("DisplacementSQCNI/Axisymmetric/Quad", 0.0).getVolumeUndeformed() > 0.0


@pytest.mark.parametrize("r0", [0.0, 1.0, 10.0])
def test_the_size_of_an_axisymmetric_particle_is_that_of_its_section(r0):
    """The volume of an axisymmetric particle is the volume of its ring, :math:`2 \\pi r_c A`; its size is the edge of
    its square in the section r-z, as for its plane strain twin, wherever it lies -- the support of the kernel functions
    matched to it (:mod:`~edelweissmeshfree.generators.kernelmatchingtoparticlegenerator`) does not grow with the
    radius."""

    axisymmetric = _particle("DisplacementSQCNI/Axisymmetric/Quad", r0)
    plane = _particle("DisplacementSQCNI/PlaneStrain/Quad", r0)

    assert axisymmetric.getVolumeUndeformed() == pytest.approx(2 * np.pi * (r0 + SIZE / 2) * SIZE**2, rel=1e-12)
    assert characteristicLength(axisymmetric) == pytest.approx(SIZE, rel=1e-12)
    assert characteristicLength(plane) == pytest.approx(SIZE, rel=1e-12)


def test_the_random_shift_of_bonded_kernel_functions_scales_with_the_size_of_the_particles_in_the_section():
    """With a random shift, a bonded kernel function lies off the center of its particle by at most half the factor
    times the size of the particle as it was made, in each direction, at every radius of the ring: the shift scales with
    the size of the particle in the section, not with its ring."""

    factor = 0.1
    model = particlePlate("DisplacementSQCNI/Axisymmetric/Quad", 2.2, 3, randomShift=factor)
    domain = model.particleKernelDomains["b"]
    offsets = np.array(
        [
            kernelFunction.center - particle.getCenterCoordinates()
            for particle, kernelFunction in zip(domain.particles, domain.meshfreeKernelFunctions)
        ]
    )
    assert characteristicLength(domain.particles[0]) == pytest.approx(H, rel=1e-12)
    assert np.abs(offsets).max() <= 0.5 * factor * H
    assert np.abs(offsets).max() > 0.25 * factor * H, "the kernel functions are not shifted"
