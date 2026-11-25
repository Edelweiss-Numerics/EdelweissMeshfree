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
"""EdelweissFE's explicit dynamic solver ``NED`` integrates the nonlocal field of gradient-enhanced particles by the
particles' own operators: a free 3D bar of ``GradientEnhanced6FiniteStrainSQCNIxNSNI/3D/Hexa`` particles with
``GM7/COROTATIONAL`` is stretched by its initial velocity, elastically, and its nonlocal field ``strain symmetric``
follows the strain.

* With a micro-inertia, the nonlocal field is integrated by central differences, damped by its viscosity; with a
  viscosity alone, by forward Euler. ``NED`` derives this from the particles' lumped inertia and damping, in a model of
  particles alone.
* The time increment is the critical time step of the particles, scaled by the courant number. The material
  parameters make the nonlocal field's limit the smaller one by far, so that this is what is tested: the bound lies
  below the exact critical time step of the particles' tangent, and the explicit scheme is stable up to that exact
  increment, whatever the damping.
"""

import numpy as np
import pytest
from edelweissfe.journal.journal import Journal
from edelweissfe.solvers.nonlinearexplicitdynamic import NED
from edelweissfe.steps.adaptivestepforexplicitsimulations import (
    AdaptiveStepForExplicitSimulations,
)
from edelweissfe.steps.stepmanager import StepActionCollection

from edelweissmeshfree.fieldoutput.fieldoutput import MPMFieldOutputController
from edelweissmeshfree.generators.boxhexaparticlegridgenerator import (
    generateBoxHexaParticleGrid,
)
from edelweissmeshfree.generators.kernelmatchingtoparticlegenerator import (
    generateKernelMatchingToParticle,
)
from edelweissmeshfree.meshfree.approximations.marmot.marmotmeshfreeapproximation import (
    MarmotMeshfreeApproximationWrapper,
)
from edelweissmeshfree.meshfree.kernelfunctions.marmot.marmotmeshfreekernelfunction import (
    MarmotMeshfreeKernelFunctionWrapper,
)
from edelweissmeshfree.meshfree.particlekerneldomain import ParticleKernelDomain
from edelweissmeshfree.models.mpmdiscretesystem import (
    STABLE_FRACTION_OF_THE_TANGENT_LIMIT,
)
from edelweissmeshfree.models.mpmmodel import MPMModel
from edelweissmeshfree.models.test._criticaltimestep import (
    estimateOfTheSystem,
    exactCriticalTimeStep,
    serially,
)
from edelweissmeshfree.particlemanagers.kdbinorganizedparticlemanager import (
    KDBinOrganizedParticleManager,
)
from edelweissmeshfree.particles.marmot.marmotparticlewrapper import (
    MarmotParticleWrapper,
)

#: The side of a particle, and the number of particles across and along the bar.
PARTICLE_SIZE = 10.0
N_ACROSS, N_ALONG = 3, 6
#: The stretching rate of the initial velocity, small enough that the bar stays elastic.
STRETCHING_RATE = 4.0

#: Micro-inertia and viscosity of the two nonlocal fields; both make the nonlocal limit of the critical time step about
#: a hundredth of the mechanical one. The micro-inertia is the largest GM7 admits, eta^2/4.
WITH_MICRO_INERTIA = {"microInertia": 1e-7**2 / 4, "viscosity": 1e-7}
VISCOSITY_ONLY = {"microInertia": 0.0, "viscosity": 6e-8}

N_INCREMENTS = 200


def _properties(microInertia: float, viscosity: float) -> np.ndarray:
    """GM7/COROTATIONAL: E, nu, k1, k2, k3, k4, k5, c, m, c3, fc, transitionWidth, transitionExponent, stressUnit,
    density, nonlocal viscosity, micro-inertia."""

    return np.array(
        [2.5e4, 0.18, 1.5e-4, 110, 30, 100, 1e-4, 25.0, 1.0, 4.0, 36.0, 4.0, 0.4, 1.0, 2.4e-9, viscosity, microInertia]
    )


def _bar(journal: Journal, microInertia: float, viscosity: float) -> MPMModel:
    """The bar along z, centered at z = 0, with the initial velocity of a uniform stretching along z."""

    dimension = 3
    model = MPMModel(dimension)
    approximation = MarmotMeshfreeApproximationWrapper("ReproducingKernel", dimension, completenessOrder=1)
    material = {"material": "GM7/COROTATIONAL", "properties": _properties(microInertia, viscosity)}

    def particleFactory(number, vertexCoordinates, volume):
        return MarmotParticleWrapper(
            "GradientEnhanced6FiniteStrainSQCNIxNSNI/3D/Hexa", number, vertexCoordinates, 0.0, approximation, material
        )

    width, length = N_ACROSS * PARTICLE_SIZE, N_ALONG * PARTICLE_SIZE
    generateBoxHexaParticleGrid(
        model,
        journal,
        particleFactory,
        name="bar",
        x0=-width / 2,
        y0=-width / 2,
        z0=-length / 2,
        l=width,
        h=width,
        t=length,
        nX=N_ACROSS,
        nY=N_ACROSS,
        nZ=N_ALONG,
    )

    def kernelFunctionFactory(node, supportSize):
        return MarmotMeshfreeKernelFunctionWrapper(node, "BSplineBoxed", supportRadius=supportSize, continuityOrder=3)

    generateKernelMatchingToParticle(
        model, journal, kernelFunctionFactory, model.particleSets["bar_all"], supportScalingFactor=2.4
    )

    particleKernelDomain = ParticleKernelDomain(
        list(model.particles.values()), list(model.meshfreeKernelFunctions.values())
    )
    model.particleKernelDomains["bar"] = particleKernelDomain
    model.particleManagers.append(
        KDBinOrganizedParticleManager(particleKernelDomain, dimension, journal, bondParticlesToKernelFunctions=True)
    )
    model.prepareYourself(journal)

    for particle in model.particles.values():
        particle.getResultArray("velocity")[2] = STRETCHING_RATE * particle.getCenterCoordinates()[2]
        particle.acceptStateAndPosition()

    return model


def runBar(
    microInertia: float,
    viscosity: float,
    courantNumber: float,
    nIncrements: int = N_INCREMENTS,
    newmarkParametersBefore: tuple[float, float] = (0.0, 0.0),
):
    """Run the bar for ``nIncrements`` increments of the critical time step, scaled by ``courantNumber``. Before the
    step, the particles hold the Newmark parameters ``newmarkParametersBefore``, as the step of a solver before would
    have left them.

    Returns
    -------
    tuple[MPMModel, NED]
        The model and the solver after the step.
    """

    journal = Journal(verbose=False)
    model = _bar(journal, microInertia, viscosity)
    model.setNewmarkParametersOfEntities(*newmarkParametersBefore)
    solver = NED({}, journal, **{"output-frequency": nIncrements, "courant-number": courantNumber})
    fieldOutputController = MPMFieldOutputController(model, journal)
    fieldOutputController.initializeJob()
    try:
        AdaptiveStepForExplicitSimulations(
            0,
            model,
            fieldOutputController,
            journal,
            {},
            solver,
            [],
            StepActionCollection(),
            stepLength=1.0,
            startInc=1.0,
            maxInc=1.0,
            minInc=1.0,
            maxNumInc=nIncrements,
        ).solve()
    finally:
        fieldOutputController.finalizeJob()

    return model, solver


def nonlocalRateAndIncrement(model: MPMModel) -> tuple[float, float]:
    """The largest rate and the largest last increment of the nonlocal field at the nodes; not finite counts as
    infinite."""

    nonlocalField = model.nodeFields["strain symmetric"]
    return tuple(float(np.nan_to_num(np.max(np.abs(nonlocalField[entry])), nan=np.inf)) for entry in ("V", "dU"))


#: The largest rate of the nonlocal field, at the nodes, of a stable run: the strain rate of the stretching is
#: STRETCHING_RATE, and the nonlocal field follows the strain. An unstable run exceeds it by orders of magnitude.
STABLE_NONLOCAL_RATE = 10 * STRETCHING_RATE


@pytest.mark.parametrize(
    "nonlocalField, scheme",
    [(WITH_MICRO_INERTIA, "second order"), (VISCOSITY_ONLY, "first order")],
    ids=["micro-inertia", "viscosity only"],
)
def test_the_nonlocal_field_is_integrated_by_the_particles_operators_on_their_critical_time_step(nonlocalField, scheme):
    model, solver = runBar(**nonlocalField, courantNumber=0.5)

    fields = solver.secondOrderFields if scheme == "second order" else solver.firstOrderFields
    assert "strain symmetric" in fields
    assert "displacement" in solver.secondOrderFields

    # the time increment is the critical time step of the particles, scaled by the courant number
    assert solver._criticalTimeStep == 0.5 * solver._criticalTimeStepOfTheEntities.value
    assert solver.prevTimeStep.timeIncrement == pytest.approx(solver._criticalTimeStep, rel=1e-12)

    # the nonlocal field follows the strain of the stretching bar
    rate, increment = nonlocalRateAndIncrement(model)
    assert increment > 0.0
    assert rate < STABLE_NONLOCAL_RATE


@pytest.mark.parametrize("nonlocalField", [WITH_MICRO_INERTIA, VISCOSITY_ONLY], ids=["micro-inertia", "viscosity only"])
def test_carried_weights_tighten_the_estimate_and_keep_it_below_the_exact_critical_time_step(nonlocalField):
    """Carried from estimate to estimate on the same state, the weights take one step of the power iteration per
    estimate: every estimate, the first from the weights constant per field, is at least the one without weights and
    stays below the exact critical time step of the tangent, scaled by STABLE_FRACTION_OF_THE_TANGENT_LIMIT, and a few
    estimates tighten it. The weights left are positive on the degrees of freedom of the particles' schemes, at most
    one, and zero elsewhere."""

    model = _bar(Journal(verbose=False), **nonlocalField)
    exact = exactCriticalTimeStep(model)
    system = model.buildDiscreteSystem()
    withoutWeights = estimateOfTheSystem(system).value
    weights = np.zeros(system.numbering.nDof)

    estimates = [estimateOfTheSystem(system, weights).value for _ in range(8)]

    for estimate in estimates:
        assert withoutWeights <= estimate <= STABLE_FRACTION_OF_THE_TANGENT_LIMIT * exact
    assert estimates[-1] > 1.05 * withoutWeights
    M = np.asarray(system.assembleLumpedMassOfEntitiesWithOwnInertia(serially))
    C = np.asarray(system.assembleLumpedDampingOfEntitiesWithOwnInertia(serially))
    ofTheSchemes = (M > 0.0) | (C > 0.0)
    assert np.all(weights[ofTheSchemes] > 0.0) and np.all(weights[~ofTheSchemes] == 0.0)
    assert np.max(weights) == 1.0


def isStable(model: MPMModel) -> bool:
    """A run is stable if the rate of the nonlocal field stays that of the stretching; an unstable one exceeds it by
    orders of magnitude."""

    return nonlocalRateAndIncrement(model)[0] < STABLE_NONLOCAL_RATE


@pytest.mark.parametrize(
    "nonlocalField, tightness",
    [(WITH_MICRO_INERTIA, 0.5), (VISCOSITY_ONLY, 1.0 / 3.0)],
    ids=["micro-inertia", "viscosity only"],
)
def test_the_bound_is_below_the_exact_stable_increment_and_the_scheme_is_stable_up_to_it(nonlocalField, tightness):
    """The critical time step of the particles lies below the exact one of their tangent, scaled by
    STABLE_FRACTION_OF_THE_TANGENT_LIMIT, and within a factor of it: two for the central differences, three for forward
    Euler, whose limit is inverse in the largest eigenvalue, so that the overestimate of a Gershgorin bound enters it
    fully, not by its square root. The explicit scheme is stable just below the exact critical time step and unstable
    above it: with a micro-inertia, the viscosity does not lower the limit of the central differences, since NED takes
    the damping with the mean of the velocities before and after the update."""

    model = _bar(Journal(verbose=False), **nonlocalField)
    exact = exactCriticalTimeStep(model)
    bound = estimateOfTheSystem(model.buildDiscreteSystem()).value

    assert tightness * exact < bound <= STABLE_FRACTION_OF_THE_TANGENT_LIMIT * exact

    below, _ = runBar(**nonlocalField, courantNumber=0.95 * exact / bound)
    above, _ = runBar(**nonlocalField, courantNumber=1.2 * exact / bound)
    assert isStable(below)
    assert not isStable(above)


def test_the_particles_carry_no_newmark_inertia_in_a_step_after_an_implicit_dynamic_one():
    """An implicit dynamic step with ``inertia=entities`` leaves the particles its Newmark parameters. ``NED``
    integrates the inertia at the nodes and gives the particles zero parameters at the start of its step: the run, and
    the tangent of a particle after it (of a zero increment, ``computeKernels``, which adds the Newmark inertia of a
    particle that carries parameters), are those of particles that never had any."""

    tangents = []
    for newmarkParametersBefore in [(0.0, 0.0), (0.25, 0.5)]:
        model, _ = runBar(
            **WITH_MICRO_INERTIA, courantNumber=0.5, nIncrements=5, newmarkParametersBefore=newmarkParametersBefore
        )
        particle = next(iter(model.particles.values()))
        zero = np.zeros(particle.nDof)
        tangent = np.zeros(particle.nDof**2)
        particle.computeKernels(tangent, np.zeros(particle.nDof), zero, zero, 0.0, 1.0)
        tangents.append(tangent)

    assert np.any(tangents[0])
    np.testing.assert_array_equal(tangents[1], tangents[0])
