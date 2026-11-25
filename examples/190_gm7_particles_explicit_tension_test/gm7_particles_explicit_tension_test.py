# -*- coding: utf-8 -*-
#  ---------------------------------------------------------------------
#
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
"""Explicit dynamic tension of a concrete bar of gradient-enhanced particles with six nonlocal variables.

A free 3D bar (3 x 3 x 11 ``GradientEnhanced6FiniteStrainSQCNIxNSNI/3D/Hexa`` particles, 10 mm) of GM7 at finite
strains (``GM7/COROTATIONAL``, with the micro-inertia eta^2/4 of its nonlocal strain) is pulled apart by an initial
velocity, a uniform stretching along its axis: no boundary condition, the inertia of the two halves loads the middle.
The middle layer of particles is slightly weaker (k1 reduced by 5 %), so that the crack has a place to start.
EdelweissFE's explicit dynamic solver ``NED`` integrates the displacement and the nonlocal strain (``strain
symmetric``) by the particles' own lumped inertia and damping, on the critical time step of the particles.

Checked: the strain localizes in the middle layer, the nonlocal strain follows, the kinetic energy at the end does not
exceed the energy the bar started with, the time increment is the solver's estimate of the particles' critical time
step, and the particle displacements and the nonlocal strain match the gold file.
"""

import argparse

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
from edelweissmeshfree.models.mpmmodel import MPMModel
from edelweissmeshfree.particlemanagers.kdbinorganizedparticlemanager import (
    KDBinOrganizedParticleManager,
)
from edelweissmeshfree.particles.marmot.marmotparticlewrapper import (
    MarmotParticleWrapper,
)

PARTICLE_SIZE = 10.0  # mm
N_ACROSS, N_ALONG = 3, 11
STRETCHING_RATE = 60.0  # 1/s: the axial velocity is STRETCHING_RATE * z
VISCOSITY = 1e-5  # s, of the nonlocal strain
WEAKENING = 0.95  # of k1 in the middle layer
N_INCREMENTS = 300


def properties(k1Factor: float = 1.0) -> np.ndarray:
    """GM7/COROTATIONAL in N, mm, s: E, nu, k1, k2, k3, k4, k5, c, m, c3, fc, transitionWidth, transitionExponent,
    stressUnit, density, nonlocal viscosity, micro-inertia (eta^2/4, the largest GM7 admits)."""

    return np.array(
        [2.5e4, 0.18, 1.5e-4 * k1Factor, 110, 30, 100, 1e-4, 25.0, 1.0, 4.0, 36.0, 4.0, 0.4, 1.0, 2.4e-9]
        + [VISCOSITY, VISCOSITY**2 / 4]
    )


def run_sim(nIncrements: int = N_INCREMENTS, momentMatrixConditionLimit: float = np.inf):
    journal = Journal()
    dimension = 3
    model = MPMModel(dimension)
    approximation = MarmotMeshfreeApproximationWrapper(
        "ReproducingKernel", dimension, completenessOrder=1, momentMatrixConditionLimit=momentMatrixConditionLimit
    )

    def particleFactory(number, vertexCoordinates, volume):
        inTheMiddleLayer = abs(np.mean(vertexCoordinates[:, 2])) < 0.5 * PARTICLE_SIZE
        material = {"material": "GM7/COROTATIONAL", "properties": properties(WEAKENING if inTheMiddleLayer else 1.0)}
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

    solver = NED({}, journal, **{"output-frequency": 50})
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


def results(model: MPMModel) -> dict:
    """Per particle, ordered along the bar: the axial position, the displacement and the nonlocal strain."""

    particles = sorted(model.particles.values(), key=lambda p: p.number)
    return {
        "z": np.array([p.getCenterCoordinates()[2] - p.getResultArray("displacement")[2] for p in particles]),
        "displacement": np.array([np.array(p.getResultArray("displacement")[:3]) for p in particles]),
        "nonlocal strain": np.array([np.array(p.getResultArray("nonlocal strain")) for p in particles]),
        "local strain": np.array([np.array(p.getResultArray("local strain")) for p in particles]),
    }


def axialStrainByLayer(r: dict) -> np.ndarray:
    """The mean axial (zz) local strain of each layer of particles along the bar."""

    layers = np.round(r["z"] / PARTICLE_SIZE).astype(int)
    return np.array([r["local strain"][layers == k, 2].mean() for k in np.unique(layers)])


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    """No matter where pytest is ran, we set the working dir to this testscript's parent directory."""

    monkeypatch.chdir(request.fspath.dirname)


@pytest.fixture(scope="module")
def unlimitedRun():
    """The run without a limit on the condition of the moment matrix, shared by the tests of this module."""

    return run_sim()


def test_sim(unlimitedRun, assert_gold):
    model, solver = unlimitedRun
    r = results(model)

    assert np.isfinite(r["displacement"]).all() and np.isfinite(r["nonlocal strain"]).all()

    # stable at the solver's own time increment: the particles alone set the critical time step of the model, estimated
    # anew in every increment, and the time increment is close to the courant fraction of it
    assert solver._criticalTimeStepOfTheModel() == solver._criticalTimeStepOfTheEntities.value
    assert solver.prevTimeStep.timeIncrement == pytest.approx(solver._largestTimeIncrement, rel=0.2)

    # the strain localizes in the middle layer, and the nonlocal strain follows it
    strain = axialStrainByLayer(r)
    assert np.argmax(strain) == len(strain) // 2
    assert strain[len(strain) // 2] > 5 * np.median(strain)
    nonlocalZZ = r["nonlocal strain"][:, 2]
    assert nonlocalZZ.max() > 0.0
    assert np.argmax(nonlocalZZ) in np.flatnonzero(np.abs(r["z"]) < 0.5 * PARTICLE_SIZE)

    # no external work: the kinetic energy at the end does not exceed the energy at the start of the step
    assert solver.kineticEnergy(solver._V) <= solver._energyAtStepStart

    gold = np.loadtxt("gold.csv")
    assert_gold(np.c_[r["displacement"], r["nonlocal strain"]], gold)


def test_a_limit_on_the_condition_of_the_moment_matrix_leaves_the_bar_bit_for_bit(unlimitedRun):
    """Every point of the bar, its free faces and corners and the crack in the middle layer included, has a moment
    matrix well below a limit of 1e4: with the limit, the bar moves bit for bit as without one."""

    unlimited, _ = unlimitedRun
    limited, _ = run_sim(momentMatrixConditionLimit=1e4)

    for name, values in results(unlimited).items():
        np.testing.assert_array_equal(results(limited)[name], values)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--create-gold", dest="create_gold", action="store_true", help="create the gold file.")
    args = parser.parse_args()

    model, solver = run_sim()
    r = results(model)
    print("axial strain by layer", axialStrainByLayer(r))
    print(
        "time increment",
        solver.prevTimeStep.timeIncrement,
        "kinetic energy end/start",
        solver.kineticEnergy(solver._V) / solver._energyAtStepStart,
    )
    if args.create_gold:
        np.savetxt("gold.csv", np.c_[r["displacement"], r["nonlocal strain"]])
