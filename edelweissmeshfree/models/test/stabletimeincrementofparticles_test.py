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
"""The bound of the stable time increment of the particles is safe and tight: the explicit particle scheme, integrated
by hand with the central difference method on fixed kernel functions, is stable at the bound and unstable at twice
it. The particles bound the row sums of their absolute stiffness by the Cauchy-Schwarz inequality over their nodes,
which costs a little of the increment (the scheme's limit lies between 1.7 and 1.9 times the bound on these blocks).
Skipped where Marmot is not built."""

import numpy as np
import pytest
from edelweissfe.journal.journal import Journal

E, NU, RHO = 2.5e4, 0.18, 2.4e-9
K, G = E / (3 * (1 - 2 * NU)), E / (2 * (1 + NU))
H = 10.0

MATERIALS = {
    "DisplacementSQCNIxNSNI/PlaneStrain/Quad": {
        "material": "FiniteStrainJ2Plasticity",
        "properties": np.array([K, G, 1e9, 1e9, 1, 0, 1, RHO]),
    },
    "GradientEnhancedFiniteStrainSQCNIxNSNI/PlaneStrain/Quad": {
        "material": "GRADIENTENHANCEDCOMPRESSIBLENEOHOOKEDAMAGE",
        "properties": np.array([K, G, 1e-3, 1e-2, 5.0, RHO]),
    },
}


def _model(particleType: str, supportFactor: float, kernelOrder: int):
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
    from edelweissmeshfree.particles.marmot import marmotparticlewrapper

    if marmotparticlewrapper.__file__.endswith(".py"):
        pytest.skip("Marmot is not built")

    journal = Journal()
    model = MPMModel(2)
    nX, nY = 8, 4

    def kernelFunction(node):
        return MarmotMeshfreeKernelFunctionWrapper(
            node, "BSplineBoxed", supportRadius=supportFactor * H, continuityOrder=kernelOrder
        )

    def particle(number, vertexCoordinates, volume):
        return marmotparticlewrapper.MarmotParticleWrapper(
            particleType, number, vertexCoordinates, 0.0, approximation, MATERIALS[particleType]
        )

    approximation = MarmotMeshfreeApproximationWrapper("ReproducingKernel", 2, completenessOrder=1)
    model = generateRectangularKernelFunctionGrid(
        model, journal, kernelFunction, x0=0.5 * H, y0=0.5 * H, l=(nX - 1) * H, h=(nY - 1) * H, nX=nX, nY=nY
    )
    model = generateRectangularQuadParticleGrid(model, journal, particle, l=nX * H, h=nY * H, nX=nX, nY=nY, name="b")
    domain = ParticleKernelDomain(list(model.particles.values()), list(model.meshfreeKernelFunctions.values()))
    manager = KDBinOrganizedParticleManager(domain, 2, journal, bondParticlesToKernelFunctions=True)
    model.particleKernelDomains["b"] = domain
    model.prepareYourself(journal)
    model.particleManagers.append(manager)
    model.updateConnectivity()
    return model


def _isStable(model, dt: float, nIncrements: int = 400) -> bool:
    """The central difference method by hand on the fixed kernel functions, from a velocity projected from the
    particles' momentum; unstable if the nodal velocity grows by 1e3."""
    system = model.buildDiscreteSystem()
    particles = system.particles
    M = np.asarray(system.assembleLumpedMassOfEntitiesWithOwnInertia(lambda entities, work: work(entities)))
    Minv = np.where(M > 0.0, 1.0 / np.where(M > 0.0, M, 1.0), 0.0)
    ofEntity = system.locations.ofEntity

    rng = np.random.default_rng(0)
    momentum = np.zeros_like(M)
    for p in particles:
        p.getResultArray("velocity")[:2] = rng.uniform(-1.0, 1.0, 2)
        pp = np.zeros(p.nDof)
        p.computeLumpedMomentum(pp)
        momentum[ofEntity[p]] += pp
    V = Minv * momentum
    V0 = np.abs(V).max()

    force = np.zeros_like(M)
    for increment in range(nIncrements):
        V -= dt * Minv * force
        dU = dt * V
        force[:] = 0.0
        for p in particles:
            p.updatePhysicsExplicit(np.ascontiguousarray(dU[ofEntity[p]]), increment * dt, dt)
            p.acceptStateAndPosition()
        for p in particles:
            fp = np.zeros(p.nDof)
            p.computePhysicsKernelsExplicit(fp)
            force[ofEntity[p]] += fp
        if not np.abs(V).max() < 1e3 * V0:
            return False
    return True


@pytest.mark.parametrize(
    "particleType, supportFactor, kernelOrder",
    [
        ("DisplacementSQCNIxNSNI/PlaneStrain/Quad", 2.2, 3),
        ("DisplacementSQCNIxNSNI/PlaneStrain/Quad", 1.6, 2),
        ("GradientEnhancedFiniteStrainSQCNIxNSNI/PlaneStrain/Quad", 2.8, 3),
    ],
)
def test_the_bound_is_stable_and_within_a_factor_of_the_stable_increment(particleType, supportFactor, kernelOrder):
    model = _model(particleType, supportFactor, kernelOrder)
    bound = model.buildDiscreteSystem().stableTimeIncrementOfEntitiesWithOwnInertia(
        lambda entities, work: work(entities)
    )

    assert _isStable(_model(particleType, supportFactor, kernelOrder), bound)
    assert not _isStable(_model(particleType, supportFactor, kernelOrder), 2.0 * bound)
