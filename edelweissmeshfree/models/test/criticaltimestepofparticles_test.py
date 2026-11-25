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
"""The estimate of the critical time step of the particles is safe and tight: the explicit particle scheme, integrated
by hand with the central difference method on fixed kernel functions, is stable at the estimate and unstable at twice
it. The particles bound the row sums of their absolute stiffness by the Cauchy-Schwarz inequality over their nodes,
which costs a little of the increment (the scheme's limit lies between 1.7 and 1.9 times the estimate on these blocks).
Skipped where Marmot is not built."""

import numpy as np
import pytest

from edelweissmeshfree.models.test._criticaltimestep import (
    estimateOfTheSystem,
    serially,
)
from edelweissmeshfree.models.test._particleplate import particlePlate


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
        # The nonlocal field of this particle has a damping and no mass: _isStable integrates only the degrees of
        # freedom with a mass, so this case checks the displacement block; the first-order field is tested in
        # solvers/test/ned_gradientenhancedparticles_test.py.
        ("GradientEnhancedFiniteStrainSQCNIxNSNI/PlaneStrain/Quad", 2.8, 3),
    ],
)
def test_the_estimate_is_stable_and_within_a_factor_of_the_critical_time_step(particleType, supportFactor, kernelOrder):
    estimate = estimateOfTheSystem(particlePlate(particleType, supportFactor, kernelOrder).buildDiscreteSystem()).value

    assert _isStable(particlePlate(particleType, supportFactor, kernelOrder), estimate)
    assert not _isStable(particlePlate(particleType, supportFactor, kernelOrder), 2.0 * estimate)


def _plateWithAPenaltySupport(penalty: float):
    """The plate with a support held by a penalty on its left edge
    (:class:`~edelweissmeshfree.constraints.particlepenaltyweakdirichtlet.ParticlePenaltyWeakDirichlet`): its discrete
    system, the lumped mass and damping of the particles, the row sums of the support's mass-scaled stiffness, and the
    exact critical time step of the particles' tangent with the support's springs and without them."""

    from edelweissmeshfree.constraints.particlepenaltyweakdirichtlet import (
        ParticlePenaltyWeakDirichlet,
    )

    model = particlePlate("DisplacementSQCNIxNSNI/PlaneStrain/Quad", 2.2, 3)
    leftColumn = [p for p in model.particles.values() if p.getCenterCoordinates()[0] < 10.0]
    support = ParticlePenaltyWeakDirichlet(
        "support", model, leftColumn, "displacement", {0: 0.0}, penalty, [0, 3], interpolationLinearizedAboutCenter=True
    )
    model.constraints["support"] = support
    model.updateConnectivity()
    system = model.buildDiscreteSystem()

    M = np.asarray(system.assembleLumpedMassOfEntitiesWithOwnInertia(serially))
    C = np.asarray(system.assembleLumpedDampingOfEntitiesWithOwnInertia(serially))
    withMass = M > 0.0
    s = np.zeros_like(M)
    s[withMass] = 1.0 / np.sqrt(M[withMass])

    # the exact stiffness: the particles' tangents and the support's springs k G G^T on the held component
    K = np.zeros((len(M), len(M)))
    for particle in system.particles:
        stiffness = np.zeros(particle.nDof**2)
        particle.assembleStiffness(stiffness)
        indices = np.asarray(system.locations.ofEntity[particle])
        K[np.ix_(indices, indices)] += stiffness.reshape(particle.nDof, particle.nDof, order="F")
    ofTheParticles = K.copy()
    indices = np.asarray(system.locations[support])
    for particle, localNodes in zip(leftColumn, support._localNodesOfParticle):
        for x in particle.getVertexCoordinates()[[0, 3]]:
            G = particle.getInterpolationVectorLinearizedAboutCenter(x).ravel()
            dofs = indices[2 * localNodes]
            K[np.ix_(dofs, dofs)] += penalty * np.outer(G, G)

    def exact(stiffness):
        S = s[withMass, None] * stiffness[np.ix_(withMass, withMass)] * s[None, withMass]
        return 2.0 / np.sqrt(np.max(np.abs(np.linalg.eigvals(S))))

    bound = np.zeros(support.nDof)
    support.addBoundOfAbsoluteStiffnessTimes(None, None, s[indices], bound, None)
    rowSums = np.zeros_like(M)
    np.add.at(rowSums, indices, bound)
    rowSums *= s

    return system, M, C, rowSums, exact(K), exact(ofTheParticles)


@pytest.mark.parametrize("penalty", [1e3, 1e5, 1e7])
def test_the_rows_of_a_penalty_support_added_to_the_particles_bound_the_exact_eigenvalue(penalty):
    """The row sums of the support's mass-scaled stiffness, added to the particles' rows inside their field matrix
    (:class:`~edelweissmeshfree.models.mpmdiscretesystem.CriticalTimeStepEstimateFromTheFieldMatrix`), bound the largest
    eigenvalue of the particles' tangent with the support's springs, from a support softer than the particles, which
    leaves their estimate as it is, to one far stiffer, which sets it."""

    from edelweissmeshfree.models.mpmdiscretesystem import (
        STABLE_FRACTION_OF_THE_TANGENT_LIMIT,
    )

    system, M, C, rowSums, exactWithTheSupport, exactAlone = _plateWithAPenaltySupport(penalty)

    estimate = system.criticalTimeStepOfEntitiesWithOwnInertia(M, C, serially)
    withTheSupport = estimate.criticalTimeStep(rowSums) / STABLE_FRACTION_OF_THE_TANGENT_LIMIT
    alone = estimate.value / STABLE_FRACTION_OF_THE_TANGENT_LIMIT

    assert alone <= exactAlone
    assert withTheSupport <= exactWithTheSupport
    assert withTheSupport > 0.25 * exactWithTheSupport
    if exactWithTheSupport < 0.5 * exactAlone:
        assert withTheSupport < alone, "a support that sets the critical time step lowers the estimate"


@pytest.mark.parametrize("penalty", [1e3, 1e5, 1e7])
@pytest.mark.parametrize("supportSinceTheEstimate", [False, True], ids=["support at the estimate", "support since"])
def test_with_carried_weights_the_rows_of_a_penalty_support_bound_the_exact_eigenvalue(
    penalty, supportSinceTheEstimate, monkeypatch
):
    """With weights carried over a few estimates: where the support takes part in the discrete system at the estimate,
    the weights of the field it acts on stay constant per field, and the estimate is the one without carried weights,
    to round-off, alone and with the support's rows. A support that acts on a field whose weights are carried -- one that
    acts since the estimate, as a contact paired anew -- adds its rows times the largest weight of the block over the
    weight of the row, which still bounds the largest eigenvalue with its springs."""

    from edelweissmeshfree.models.mpmdiscretesystem import (
        STABLE_FRACTION_OF_THE_TANGENT_LIMIT,
        MPMDiscreteSystem,
    )

    if supportSinceTheEstimate:
        monkeypatch.setattr(
            MPMDiscreteSystem,
            "_dofsOfTheFieldsOfTheConstraints",
            lambda system: np.zeros(system.numbering.nDof, dtype=bool),
        )
    system, M, C, rowSums, exactWithTheSupport, exactAlone = _plateWithAPenaltySupport(penalty)
    withoutWeights = system.criticalTimeStepOfEntitiesWithOwnInertia(M, C, serially)
    weights = np.zeros(system.numbering.nDof)
    for _ in range(8):
        estimate = system.criticalTimeStepOfEntitiesWithOwnInertia(M, C, serially, weights)

    assert estimate.value / STABLE_FRACTION_OF_THE_TANGENT_LIMIT <= exactAlone
    assert estimate.criticalTimeStep(rowSums) / STABLE_FRACTION_OF_THE_TANGENT_LIMIT <= exactWithTheSupport
    if supportSinceTheEstimate:
        assert estimate.value > 1.05 * withoutWeights.value, "the weights are carried in the field of the support"
    else:
        assert estimate.value == pytest.approx(withoutWeights.value, rel=1e-12)
        assert estimate.criticalTimeStep(rowSums) == pytest.approx(withoutWeights.criticalTimeStep(rowSums), rel=1e-12)
