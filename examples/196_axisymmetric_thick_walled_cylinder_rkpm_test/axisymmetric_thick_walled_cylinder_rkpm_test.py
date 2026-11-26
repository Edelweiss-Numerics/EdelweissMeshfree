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
"""A thick-walled cylinder under internal pressure, of axisymmetric particles, against Lamé's solution.

.. code-block:: text

          z
          |      inner radius a        outer radius b
          |      |                     |
          |      +---------------------+   u_z = 0 everywhere (plane strain): a slice of a long tube
    axis  | p -> |  particles in r-z   |
          |      +---------------------+
          +----------------------------------- r

A long tube, inner radius ``a``, outer radius ``b``, under the pressure ``p`` on its inner surface, in plane strain
(:math:`u_z = 0`): Lamé's solution is

.. math::
    u_r(r) = \\frac{(1 + \\nu)\\, p\\, a^2}{E\\, (b^2 - a^2)} \\left( (1 - 2 \\nu)\\, r + \\frac{b^2}{r} \\right) .

The model is the section r-z of a slice of the tube: an axisymmetric model (``MPMModel(2, axisymmetric=True)``) of
axisymmetric particles (``.../Axisymmetric/Quad``), every particle standing for the ring it sweeps about the axis. The
nodes of the kernel functions sit at the centers of the particles
(:func:`~edelweissmeshfree.generators.kernelmatchingtoparticlegenerator.generateKernelMatchingToParticle`), the
pressure acts on the inner faces of the particles (:class:`~edelweissmeshfree.stepactions.particledistributedload.
ParticleDistributedLoad`, which Marmot integrates over the ring of each face), and :math:`u_z = 0` is held at every
node: the reproducing kernels interpolate a field whose nodal values all vanish as zero. Nothing holds the tube
radially: a radial translation of the section is a change of the radius of every ring, and the hoop strain
:math:`u_r / r` makes it cost energy.

The solid is a compressible neo-Hookean one, loaded so little that it responds linearly: its largest strain is of the
order of :math:`u_r(a) / a`, a few times :math:`10^{-5}`, far below the damage threshold :math:`10^{-3}` of the
gradient-enhanced particle's material.

**The error.** Lamé's displacement is :math:`u_r = c_1 r + c_2 / r`. The particles reproduce its linear part
:math:`c_1 r` exactly: their reproducing kernels are linearly complete, and they pass the axisymmetric patch test
(Marmot's documentation). What is left is the error of the approximation of :math:`c_2 / r`, of the order of
:math:`h^2 |u_r''|` for the particle size :math:`h`, largest at the inner face, where :math:`u_r'' = 2 c_2 / a^3`. So
the error of the particles' centers against Lamé falls with the square of the particle size, and stays below
:math:`h^2 |u_r''(a)|`, for the displacement particle (SQCNI) and for the gradient-enhanced one (SQCNI with NSNI)
(``test_the_particles_of_a_pressed_tube_follow_lame``).

The step is quasi-static, solved by EdelweissFE's ``NISTParallel``.
"""

import numpy as np
import pytest
from edelweissfe.config.configurator import loadConfiguration
from edelweissfe.journal.journal import Journal
from edelweissfe.solvers.nonlinearimplicitstaticparallel import NISTParallel
from edelweissfe.stepactions.dirichlet import StepAction as Dirichlet
from edelweissfe.steps.adaptivestep import AdaptiveStep
from edelweissfe.steps.stepmanager import StepActionCollection

from edelweissmeshfree.fieldoutput.fieldoutput import MPMFieldOutputController
from edelweissmeshfree.generators.kernelmatchingtoparticlegenerator import (
    generateKernelMatchingToParticle,
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
from edelweissmeshfree.stepactions.particledistributedload import (
    ParticleDistributedLoad,
)

#: The tube: inner and outer radius [mm], and the pressure [MPa], small against the moduli.
A, B, PRESSURE = 10.0, 20.0, 1.0
#: The moduli [MPa]: bulk and shear modulus; Young's modulus and Poisson's ratio follow.
K, G = 25000.0, 12500.0
E, NU = 9 * K * G / (3 * K + G), (3 * K - 2 * G) / (2 * (3 * K + G))
#: The materials of the particles: the compressible neo-Hookean solid, and its gradient-enhanced damage twin (bulk and
#: shear modulus, damage threshold and final strain, nonlocal length, density).
MATERIALS = {
    "DisplacementSQCNI/Axisymmetric/Quad": {"material": "CompressibleNeoHooke", "properties": np.array([K, G, 2.4e-9])},
    "GradientEnhancedFiniteStrainSQCNIxNSNI/Axisymmetric/Quad": {
        "material": "GRADIENTENHANCEDCOMPRESSIBLENEOHOOKEDAMAGE",
        "properties": np.array([K, G, 1e-3, 1e-2, 2.0, 2.4e-9]),
    },
}


def lame(r: np.ndarray) -> np.ndarray:
    """Lamé's radial displacement of the tube in plane strain, :math:`c_1 r + c_2 / r`."""

    return (1 + NU) * PRESSURE * A**2 / (E * (B**2 - A**2)) * ((1 - 2 * NU) * r + B**2 / r)


def curvatureOfLameAtTheInnerFace() -> float:
    """:math:`|u_r''(a)| = 2 c_2 / a^3`, the second derivative of the :math:`c_2 / r` part of Lamé's displacement at
    the inner face, where it is largest."""

    c2 = (1 + NU) * PRESSURE * A**2 * B**2 / (E * (B**2 - A**2))
    return 2 * c2 / A**3


def run(particleType: str, nAcross: int) -> tuple[np.ndarray, np.ndarray]:
    """Press the tube from inside.

    Parameters
    ----------
    particleType
        The axisymmetric Marmot particle, a key of ``MATERIALS``.
    nAcross
        The number of particles across the wall; the slice is four particles high.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        The radii of the particles' centers, and their radial displacements.
    """

    journal = Journal(verbose=False)
    model = MPMModel(2, axisymmetric=True)
    size = (B - A) / nAcross

    approximation = MarmotMeshfreeApproximationWrapper("ReproducingKernel", 2, completenessOrder=1)
    generateRectangularQuadParticleGrid(
        model,
        journal,
        lambda number, vertices, volume: MarmotParticleWrapper(
            particleType, number, vertices, 0.0, approximation, MATERIALS[particleType]
        ),
        name="tube",
        x0=A,
        y0=0.0,
        l=B - A,
        h=4 * size,
        nX=nAcross,
        nY=4,
    )
    generateKernelMatchingToParticle(
        model,
        journal,
        lambda node, supportSize: MarmotMeshfreeKernelFunctionWrapper(
            node, "BSplineBoxed", supportRadius=supportSize, continuityOrder=3
        ),
        model.particleSets["tube_all"],
        supportScalingFactor=2.4,
        name="kernels",
    )
    domain = ParticleKernelDomain(list(model.particles.values()), list(model.meshfreeKernelFunctions.values()))
    model.particleKernelDomains["tube"] = domain
    model.particleManagers.append(
        KDBinOrganizedParticleManager(domain, 2, journal, bondParticlesToKernelFunctions=True)
    )

    model.prepareYourself(journal)

    radii = np.array([particle.getCenterCoordinates()[0] for particle in model.particleSets["tube_all"]])
    stepActions = StepActionCollection()
    stepActions["dirichlet"]["plane strain"] = Dirichlet(
        "plane strain", model.nodeSets["kernels_all"], "displacement", {1: 0.0}, model, journal
    )
    # the left faces of the grid's particles are the inner surface of the tube
    pressure = ParticleDistributedLoad(
        "pressure", model, journal, model.surfaces["tube_left"], "pressure", np.array([PRESSURE])
    )
    stepActions["distributedload"][pressure.name] = pressure

    fieldOutputController = MPMFieldOutputController(model, journal)
    jobInfo = loadConfiguration({})
    step = AdaptiveStep(
        0,
        model,
        fieldOutputController,
        journal,
        jobInfo,
        NISTParallel(jobInfo, journal, linsolver="pardiso"),
        [],
        stepActions,
        startInc=1.0,
        maxInc=1.0,
        minInc=1e-3,
        maxNumInc=10,
    )
    fieldOutputController.initializeJob()
    try:
        step.solve()
    finally:
        fieldOutputController.finalizeJob()

    displacements = np.array(
        [particle.getResultArray("displacement")[0] for particle in model.particleSets["tube_all"]]
    )
    return radii, displacements


@pytest.mark.parametrize("particleType", list(MATERIALS))
def test_the_particles_of_a_pressed_tube_follow_lame(particleType):
    """The radial displacement of every particle's center follows Lamé's solution, relative to the inner one, to
    below :math:`h^2 |u_r''(a)| / u_r(a)`, the second-order term of its :math:`c_2 / r` part at the inner face, and the
    error falls with the square of the particle size :math:`h`: by more than :math:`2^{1.5}` from 8 to 16 particles
    across, with room for the range before the asymptotic one (see the module)."""

    errors = []
    for nAcross in (8, 16):
        size = (B - A) / nAcross
        radii, displacements = run(particleType, nAcross)
        errors.append(np.abs(displacements - lame(radii)).max() / lame(A))
        assert errors[-1] < size**2 * curvatureOfLameAtTheInnerFace() / lame(A)
    assert errors[0] / errors[1] > 2**1.5


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    """No matter where pytest is ran, we set the working dir to this testscript's parent directory."""

    monkeypatch.chdir(request.fspath.dirname)


if __name__ == "__main__":
    for particleType in MATERIALS:
        for nAcross in (8, 16, 32):
            radii, displacements = run(particleType, nAcross)
            error = np.abs(displacements - lame(radii)) / lame(A)
            print(f"{particleType}, {nAcross} across: largest deviation from Lame {error.max():.3e} of u_r(a)")
