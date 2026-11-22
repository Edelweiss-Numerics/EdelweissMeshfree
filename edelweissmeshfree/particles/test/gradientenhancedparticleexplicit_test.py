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
"""The explicit methods of a gradient-enhanced finite-strain particle with six nonlocal variables (GM7 at finite
strains) reach Python: its fields, its lumped damping, its stiffness and the bound of its absolute stiffness.

Skipped where Marmot is not built.
"""

import itertools

import numpy as np
import pytest
from edelweissfe.points.node import Node

# GM7/COROTATIONAL: E, nu, k1, k2, k3, k4, k5, c, m, c3, fc, transitionWidth, transitionExponent, stressUnit, density,
# nonlocal viscosity eta, micro-inertia eta^2/4
ETA = 3e-4
PROPERTIES = np.array(
    [2.5e4, 0.18, 1.5e-4, 110, 30, 100, 1e-4, 25.0, 1.0, 4.0, 36.0, 4.0, 0.4, 1.0, 2.4e-9, ETA, ETA**2 / 4]
)
H = 10.0  # the side of the cubic particle


def _particle():
    from edelweissmeshfree.meshfree.approximations.marmot.marmotmeshfreeapproximation import (
        MarmotMeshfreeApproximationWrapper,
    )
    from edelweissmeshfree.meshfree.kernelfunctions.marmot.marmotmeshfreekernelfunction import (
        MarmotMeshfreeKernelFunctionWrapper,
    )
    from edelweissmeshfree.particles.marmot import marmotparticlewrapper

    if marmotparticlewrapper.__file__.endswith(".py"):
        pytest.skip("Marmot is not built")

    vertices = H * np.array([[i, j, k] for k in (0, 1) for j, i in ((0, 0), (0, 1), (1, 1), (1, 0))], dtype=float)
    particle = marmotparticlewrapper.MarmotParticleWrapper(
        "GradientEnhanced6FiniteStrainSQCNIxNSNI/3D/Hexa",
        1,
        vertices,
        0.0,
        MarmotMeshfreeApproximationWrapper("ReproducingKernel", 3, completenessOrder=1),
        {"material": "GM7/COROTATIONAL", "properties": PROPERTIES},
    )
    particle.assignKernelFunctions(
        [
            MarmotMeshfreeKernelFunctionWrapper(
                Node(label, H * np.array(center)), "BSplineBoxed", supportRadius=1.5 * H, continuityOrder=3
            )
            for label, center in enumerate(itertools.product((-0.5, 0.5, 1.5), repeat=3))
        ]
    )
    particle.initializeYourself()

    return particle


def test_six_nonlocal_variables_are_the_field_strain_symmetric():
    particle = _particle()

    assert particle.baseFields == ["displacement", "strain symmetric"]
    assert particle.nDof == 27 * (3 + 6)


def test_the_lumped_damping_is_the_nonlocal_viscosity_on_the_nonlocal_rows():
    particle = _particle()
    C = np.zeros(particle.nDof)

    particle.computeLumpedDamping(C)

    C = C.reshape(-1, 9)
    assert np.all(C[:, :3] == 0.0)
    # the partition of unity distributes eta V0 over the nodes, for each of the six components
    np.testing.assert_allclose(C[:, 3:].sum(axis=0), ETA * H**3, rtol=1e-12)


def test_the_stiffness_leaves_the_state_unchanged():
    particle = _particle()
    state = np.array(particle._stateVars)
    K = np.zeros(particle.nDof**2)

    particle.assembleStiffness(K)

    np.testing.assert_array_equal(particle._stateVars, state)
    np.testing.assert_array_equal(particle._stateVarsTemp, state)
    K = K.reshape(particle.nDof, particle.nDof, order="F")
    # the displacement block of the elastic stiffness is symmetric and positive semi-definite, a translation is free of
    # stress
    displacement = np.array([i % 9 < 3 for i in range(particle.nDof)])
    Kuu = K[np.ix_(displacement, displacement)]
    translation = np.tile([1.0, 0.0, 0.0], particle.nDof // 9)
    np.testing.assert_allclose(Kuu @ translation, 0.0, atol=1e-8 * np.abs(Kuu).max())


def test_the_bound_of_the_absolute_stiffness_leaves_the_particle_as_it_was():
    """The bound is Marmot's visitor of the stiffness the particle describes, which leaves the particle as it was
    found: in the middle of an increment (advanced, not accepted), its working state and the internal force that
    follows the bound are those before it, with no reset by the host."""
    particle = _particle()
    nDof = particle.nDof
    particle.updatePhysicsExplicit(1e-4 * np.sin(np.arange(nDof)), 1.0, 1.0)
    state = np.array(particle._stateVarsTemp)
    before = np.zeros(nDof)
    particle.computePhysicsKernelsExplicit(before)

    particle.addBoundOfAbsoluteStiffnessTimes(np.ones((nDof, 1), order="F"), np.zeros((nDof, 1), order="F"))

    after = np.zeros(nDof)
    particle.computePhysicsKernelsExplicit(after)
    np.testing.assert_array_equal(particle._stateVarsTemp, state)
    np.testing.assert_array_equal(after, before)
