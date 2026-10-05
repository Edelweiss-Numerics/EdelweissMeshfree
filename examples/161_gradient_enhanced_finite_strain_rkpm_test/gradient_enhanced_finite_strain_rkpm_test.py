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
"""RKPM with the gradient-enhanced finite-strain particles and damage material.

A plane strain block (4 x 8) of stabilized-conforming-nodal-integration quad particles
(GradientEnhancedFiniteStrainSQCNI and ...SQCNIxNSNI) with a reproducing kernel approximation is compressed by
2 %, with two of Marmot's gradient-enhanced finite-strain materials: GradientEnhancedCompressibleNeoHookeDamage
(loaded to twenty times its damage threshold) and GradientEnhancedFiniteStrainDruckerPrager (Drucker-Prager
plasticity whose dilatant flow drives the implicit-gradient damage). Boundary conditions are imposed weakly with
Lagrange multipliers. Checks: the material-specific mechanism is active (damage / plastic flow), and the particle
displacements match the gold file of each particle type and material.
"""

import argparse

import numpy as np
import pytest
from edelweissfe.journal.journal import Journal
from edelweissfe.linsolve.pardiso.pardiso import pardisoSolve
from edelweissfe.timesteppers.adaptivetimestepper import AdaptiveTimeStepper

from edelweissmeshfree.constraints.particlelagrangianweakdirichlet import (
    ParticleLagrangianWeakDirichletOnParticleSetFactory,
)
from edelweissmeshfree.fieldoutput.fieldoutput import MPMFieldOutputController
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
from edelweissmeshfree.particles.marmot.marmotparticlewrapper import (
    MarmotParticleWrapper,
)
from edelweissmeshfree.solvers.nqs import NonlinearQuasistaticSolver

KAPPA0 = 1e-3
COMPRESSION = -0.16  # 2 % of the height

MATERIALS = {
    # K, G, kappa0, kappaF, l, rho (zero density: no inertia in this quasi-static test)
    "neohooke": {
        "material": "GradientEnhancedCompressibleNeoHookeDamage",
        "properties": np.array([3500.0, 1500.0, KAPPA0, 1e-2, 2.0, 0.0]),
    },
    # K, G, c0, phi, psi, H, epsF, omegaMax, l, m, rho
    "druckerprager": {
        "material": "GradientEnhancedFiniteStrainDruckerPrager",
        "properties": np.array([3500.0, 1500.0, 5.0, 30.0, 10.0, 100.0, 0.02, 0.99, 2.0, 1.0, 0.0]),
    },
}

PARTICLE_TYPES = [
    "GradientEnhancedFiniteStrainSQCNI/PlaneStrain/Quad",
    "GradientEnhancedFiniteStrainSQCNIxNSNI/PlaneStrain/Quad",
]

# the C++ particle keeps a bare reference to the approximation wrapper -> keep it alive
_approximations = []


def run_sim(particleType, materialName):
    dimension = 2
    journal = Journal()
    model = MPMModel(dimension)

    l, h = 4.0, 8.0
    nX, nY = 4, 8
    particleSize = l / nX
    supportRadius = 2.5 * particleSize

    def kernelFunctionFactory(node):
        return MarmotMeshfreeKernelFunctionWrapper(node, "BSplineBoxed", supportRadius=supportRadius, continuityOrder=3)

    model = generateRectangularKernelFunctionGrid(
        model,
        journal,
        kernelFunctionFactory,
        x0=particleSize / 2.0,
        y0=particleSize / 2.0,
        l=l - particleSize,
        h=h - particleSize,
        nX=nX,
        nY=nY,
        name="kernel_grid",
    )

    approximation = MarmotMeshfreeApproximationWrapper("ReproducingKernel", dimension, completenessOrder=1)
    _approximations.append(approximation)

    material = MATERIALS[materialName]

    def particleFactory(number, vertexCoordinates, volume):
        return MarmotParticleWrapper(particleType, number, vertexCoordinates, volume, approximation, material)

    model = generateRectangularQuadParticleGrid(
        model, journal, particleFactory, x0=0.0, y0=0.0, l=l, h=h, nX=nX, nY=nY, name="specimen"
    )

    domain = ParticleKernelDomain(list(model.particles.values()), list(model.meshfreeKernelFunctions.values()))
    model.particleKernelDomains["domain"] = domain

    particleManager = KDBinOrganizedParticleManager(domain, dimension, journal, bondParticlesToKernelFunctions=False)

    sets = model.particleSets
    for name, particleSet, values in (
        ("bottom", sets["specimen_bottom"], {1: 0.0}),
        # the corner lies on the bottom edge, whose y displacement is constrained already: fixing it again
        # would add a linearly dependent Lagrange multiplier row
        ("corner", sets["specimen_leftBottom"], {0: 0.0}),
        ("top", sets["specimen_top"], {1: COMPRESSION}),
    ):
        model.constraints.update(
            ParticleLagrangianWeakDirichletOnParticleSetFactory(
                name, particleSet, "displacement", values, model, location="center"
            )
        )

    model.prepareYourself(journal)

    fieldOutputController = MPMFieldOutputController(model, journal)
    outputs = ["displacement", "nonlocal damage"] + (["alphaP"] if materialName == "druckerprager" else [])
    for name in outputs:
        fieldOutputController.addPerParticleFieldOutput(name, sets["all"], name)
    fieldOutputController.initializeJob()

    adaptiveTimeStepper = AdaptiveTimeStepper(0.0, 1.0, 0.25, 0.25, 1e-3, 100, journal)
    iterationOptions = {"max. iterations": 15, "critical iterations": 8, "allowed residual growths": 4}

    try:
        NonlinearQuasistaticSolver(journal).solveStep(
            adaptiveTimeStepper,
            pardisoSolve,
            model,
            fieldOutputController,
            outputManagers=[],
            particleManagers=[particleManager],
            constraints=model.constraints.values(),
            userIterationOptions=iterationOptions,
        )
    finally:
        fieldOutputController.finalizeJob()

    return fieldOutputController


def result(fieldOutputController, name):
    """The last result, one row per particle (vector fields have three components also in 2D)."""
    return np.asarray(fieldOutputController.fieldOutputs[name].getLastResult())


def goldFile(particleType, materialName):
    suffix = "" if materialName == "neohooke" else "_" + materialName
    return "gold_" + particleType.split("/")[0] + suffix + ".csv"


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    """No matter where pytest is ran, we set the working dir
    to this testscript's parent directory"""

    monkeypatch.chdir(request.fspath.dirname)


@pytest.mark.parametrize("materialName", MATERIALS.keys())
@pytest.mark.parametrize("particleType", PARTICLE_TYPES)
def test_sim(assert_gold, particleType, materialName):
    fieldOutputController = run_sim(particleType, materialName)
    u = result(fieldOutputController, "displacement")[:, :2]
    n = result(fieldOutputController, "nonlocal damage").ravel()

    assert np.isfinite(u).all() and np.isfinite(n).all()
    assert u.shape == (32, 2), "one (u_x, u_y) row per particle"
    assert abs(u[:, 1].min() - COMPRESSION) < 1e-8, "the top edge must follow the imposed compression"
    if materialName == "neohooke":
        assert n.max() > 5 * KAPPA0, "the nonlocal field must drive the material well into damage"
    else:
        assert result(fieldOutputController, "alphaP").max() > 0.0, "the material must yield"
        assert n.max() > 0.0, "the plastic flow must drive the nonlocal field"

    assert_gold(u, np.loadtxt(goldFile(particleType, materialName)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--create-gold", dest="create_gold", action="store_true", help="create the gold files.")
    args = parser.parse_args()

    for materialName in MATERIALS:
        for particleType in PARTICLE_TYPES:
            fieldOutputController = run_sim(particleType, materialName)
            u = result(fieldOutputController, "displacement")[:, :2]
            n = result(fieldOutputController, "nonlocal damage").ravel()
            print(materialName, particleType, "max nonlocal field", n.max(), " min u_y", u[:, 1].min())
            if args.create_gold:
                np.savetxt(goldFile(particleType, materialName), u)
