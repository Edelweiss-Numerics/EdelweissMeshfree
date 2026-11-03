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
"""The free vibration of an RKPM cantilever in implicit dynamics, with both places of the particles' inertia.

.. code-block:: text

      clamped          a pressure on the top, applied suddenly at t = 0
        |#|vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv|
        |#|                       RKPM particles                          |
        |#|---------------------------------------------------------------|
        x = 0                                                         x = 10

The cantilever starts at rest; the load makes it oscillate about its static deflection, mostly in its first bending
mode. It is solved by EdelweissFE's ``NID`` with the average acceleration rule, once with the particles integrating
their own inertia (``inertia=entities``) and once with ``NID`` integrating it at the nodes, with the
particles' lumped mass (``inertia=nodes``).

The independent check is the period of the tip deflection: against the Euler-Bernoulli beam,
:math:`\\omega_1 = 1.8751^2 \\sqrt{E' I / (\\rho A L^4)}` with the plane strain modulus
:math:`E' = E / (1 - \\nu^2)`, and against the same cantilever made of finite elements (``CPE8UL``) under ``NID``. The
continuum is a little softer than the beam (shear deformation and rotary inertia, about one percent at
:math:`L / h = 10`), so the reference of the refinement is the converged continuum; the beam bounds it. The load is
small, so the response is linear.

``python rkpm_cantilever_vibration_test.py --study`` prints the periods of three particle sizes.
"""

import argparse

import numpy as np
import pytest
from edelweissfe.generators.planerectquad import Generator as PlaneRectQuadGenerator
from edelweissfe.generators.planerectquad import PlaneRectQuadSchema
from edelweissfe.journal.journal import Journal
from edelweissfe.models.femodel import FEModel
from edelweissfe.sections.plane import PlaneSectionSchema
from edelweissfe.sections.plane import Section as PlaneSection
from edelweissfe.sets.nodeset import NodeSet
from edelweissfe.solvers.nonlinearimplicitdynamic import NonlinearImplicitDynamic
from edelweissfe.stepactions.dirichlet import StepAction as Dirichlet
from edelweissfe.stepactions.distributedload import StepAction as DistributedLoad
from edelweissfe.steps.adaptivestep import AdaptiveStep
from edelweissfe.steps.stepmanager import StepActionCollection
from edelweissfe.utils.fieldoutput import FieldOutputController

from edelweissmeshfree.constraints.particlepenaltyweakdirichtlet import (
    ParticlePenaltyWeakDirichlet,
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
from edelweissmeshfree.solvers.meshfreeconvergencecriterion import (
    jobInfoWithMeshfreeTolerances,
    meshfreeCriterionOptions,
)
from edelweissmeshfree.stepactions.particledistributedload import (
    ParticleDistributedLoad,
)

length = 10.0
height = 1.0

youngsModulus = 1000.0
poissonsRatio = 0.3
density = 1.0
bulkModulus = youngsModulus / (3 * (1 - 2 * poissonsRatio))
shearModulus = youngsModulus / (2 * (1 + poissonsRatio))
#: A compressible neo-Hookean solid, linear for the small strains of this test.
materialProperties = np.array([bulkModulus, shearModulus, density])

#: The pressure on the top: a static tip deflection of about a thousandth of the length.
pressure = 7.0e-4

#: The penalty parameter of the clamp.
penaltyParameter = 1e6


def eulerBernoulliPeriod() -> float:
    """The period of the first bending mode of the Euler-Bernoulli cantilever of the plane strain modulus.

    Returns
    -------
    float
        The period.
    """

    planeStrainModulus = youngsModulus / (1 - poissonsRatio**2)
    secondMomentOfArea = height**3 / 12
    omega = 1.875104068711961**2 * np.sqrt(planeStrainModulus * secondMomentOfArea / (density * height * length**4))
    return 2 * np.pi / omega


#: The increments per period of the first mode, and the number of periods simulated.
incrementsPerPeriod = 40
periods = 2


def timeIncrementAndLength() -> tuple[float, float]:
    """The time increment and the length of the step.

    Returns
    -------
    tuple[float, float]
        The time increment, and the length of the step.
    """

    period = eulerBernoulliPeriod()
    return period / incrementsPerPeriod, periods * period


def periodOfTheResponse(time: np.ndarray, deflection: np.ndarray) -> float:
    """The period of the tip deflection, fitted as :math:`u(t) = a + b \\cos \\omega t + c \\sin \\omega t` in the
    least-squares sense, by a search over :math:`\\omega` around the beam's.

    Parameters
    ----------
    time
        The times of the samples.
    deflection
        The tip deflection at these times.

    Returns
    -------
    float
        The period :math:`2 \\pi / \\omega` of the best fit.
    """

    def residualOfTheFit(omega):
        basis = np.column_stack((np.ones_like(time), np.cos(omega * time), np.sin(omega * time)))
        _, residual, _, _ = np.linalg.lstsq(basis, deflection, rcond=None)
        return residual[0] if residual.size else 0.0

    omegaBeam = 2 * np.pi / eulerBernoulliPeriod()
    omegas = np.linspace(0.8 * omegaBeam, 1.2 * omegaBeam, 4001)
    best = omegas[np.argmin([residualOfTheFit(omega) for omega in omegas])]
    return 2 * np.pi / best


def solveDynamically(model, stepActions: dict, fieldOutputController, journal: Journal, solverOptions: dict):
    """Solve the step from rest under the sudden load with ``NID``, the average acceleration rule.

    Parameters
    ----------
    model
        The model.
    stepActions
        The step actions.
    fieldOutputController
        The field output controller.
    journal
        The journal.
    solverOptions
        The options of the solver besides the Newmark parameters.
    """

    timeIncrement, stepLength = timeIncrementAndLength()
    jobInfo = jobInfoWithMeshfreeTolerances(fluxResidualTolerance=1e-8, fieldCorrectionTolerance=1e-12)
    solver = NonlinearImplicitDynamic(jobInfo, journal, newmarkBeta=0.25, newmarkGamma=0.5, **solverOptions)
    step = AdaptiveStep(
        0,
        model,
        fieldOutputController,
        journal,
        jobInfo,
        solver,
        [],
        stepActions,
        stepLength=stepLength,
        startInc=timeIncrement / stepLength,
        maxInc=timeIncrement / stepLength,
        minInc=timeIncrement / stepLength,
        maxNumInc=10 * incrementsPerPeriod * periods,
        maxIter=15,
    )

    fieldOutputController.initializeJob()
    try:
        step.solve()
    finally:
        fieldOutputController.finalizeJob()


def run_rkpm(particleSize: float, inertia: str) -> tuple[np.ndarray, np.ndarray]:
    """The RKPM cantilever: its tip deflection over time.

    Parameters
    ----------
    particleSize
        The size of the particles.
    inertia
        Where the particles' inertia is integrated, ``entities`` or ``nodes``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The times, and the mean deflection of the particles at the tip.
    """

    journal = Journal()
    dimension = 2
    model = MPMModel(dimension)

    nX = round(length / particleSize)
    nY = round(height / particleSize)

    def kernelFunctionFactory(node):
        return MarmotMeshfreeKernelFunctionWrapper(
            node, "BSplineBoxed", supportRadius=2 * particleSize, continuityOrder=3
        )

    generateRectangularKernelFunctionGrid(
        model, journal, kernelFunctionFactory, name="beam", x0=0.0, l=length, y0=0.0, h=height, nX=nX, nY=nY
    )

    approximation = MarmotMeshfreeApproximationWrapper(
        "ReproducingKernelImplicitGradient", dimension, completenessOrder=1
    )
    particleMaterial = {"material": "CompressibleNeoHooke", "properties": materialProperties}

    def particleFactory(number, vertexCoordinates, volume):
        return MarmotParticleWrapper(
            "DisplacementSQCNIxNSNI/PlaneStrain/Quad", number, vertexCoordinates, 0.0, approximation, particleMaterial
        )

    generateRectangularQuadParticleGrid(
        model, journal, particleFactory, name="beam", x0=0.0, l=length, y0=0.0, h=height, nX=nX, nY=nY
    )

    particleKernelDomain = ParticleKernelDomain(
        list(model.particles.values()), list(model.meshfreeKernelFunctions.values())
    )
    model.particleKernelDomains["beam"] = particleKernelDomain
    model.particleManagers.append(
        KDBinOrganizedParticleManager(particleKernelDomain, dimension, journal, bondParticlesToKernelFunctions=True)
    )

    # the clamp holds the left face of the particles at the left end: their vertices 0 and 3
    clamp = ParticlePenaltyWeakDirichlet(
        "clamp", model, model.particleSets["beam_left"], "displacement", {0: 0, 1: 0}, penaltyParameter, [0, 3]
    )
    model.constraints[clamp.name] = clamp

    model.prepareYourself(journal)

    load = ParticleDistributedLoad(
        "pressureTop",
        model,
        journal,
        model.surfaces["beam_top"],
        "pressure",
        np.array([pressure]),
        f_t=lambda t: 1.0,
    )
    stepActions = StepActionCollection()
    stepActions["distributedload"][load.name] = load

    fieldOutputController = MPMFieldOutputController(model, journal)
    fieldOutputController.addPerParticleFieldOutput(
        "tip", model.particleSets["beam_right"], "displacement", saveHistory=True, f_x=lambda x: np.mean(x[:, 1])
    )

    solveDynamically(
        model,
        stepActions,
        fieldOutputController,
        journal,
        meshfreeCriterionOptions | {"linsolver": "pardiso", "inertia": inertia},
    )

    tip = fieldOutputController.fieldOutputs["tip"]
    return np.asarray(tip.getTimeHistory()), np.asarray(tip.getResultHistory()).ravel()


def run_finite_elements(elementSize: float) -> tuple[np.ndarray, np.ndarray]:
    """The same cantilever, of quadratic plane strain finite elements (``CPE8UL``): its tip deflection over time.

    Parameters
    ----------
    elementSize
        The size of the elements.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The times, and the mean deflection of the nodes at the tip.
    """

    journal = Journal()
    model = FEModel(2)
    with model.topology.changes():
        PlaneRectQuadGenerator(
            "beam",
            model,
            journal,
            configuration=PlaneRectQuadSchema(
                x0=0.0,
                length=length,
                h=height,
                nX=round(length / elementSize),
                nY=round(height / elementSize),
                elType="CPE8UL",
            ),
        )
    model.sections["beam"] = PlaneSection(
        "beam",
        model,
        {"name": "CompressibleNeoHooke", "properties": materialProperties},
        [model.elementSets["beam_all"]],
        configuration=PlaneSectionSchema(thickness=1.0),
    )
    model.nodeSets["all"] = NodeSet("all", model.nodes.values())
    model.prepareYourself(journal)

    stepActions = StepActionCollection()
    stepActions["dirichlet"]["clamp"] = Dirichlet(
        "clamp", model.nodeSets["beam_left"], "displacement", {0: 0.0, 1: 0.0}, model, journal
    )
    stepActions["distributedload"]["pressureTop"] = DistributedLoad(
        "pressureTop", model.surfaces["beam_top"], np.array([pressure]), "pressure", model, journal, f_t=lambda t: 1.0
    )

    fieldOutputController = FieldOutputController(model, journal)
    fieldOutputController.addPerNodeFieldOutput(
        "tip",
        model.nodeFields["displacement"].subset(model.nodeSets["beam_right"]),
        "U",
        saveHistory=True,
        f_x=lambda x: np.mean(x[:, 1]),
    )

    solveDynamically(model, stepActions, fieldOutputController, journal, {"linsolver": "pardiso"})

    tip = fieldOutputController.fieldOutputs["tip"]
    return np.asarray(tip.getTimeHistory()), np.asarray(tip.getResultHistory()).ravel()


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    """No matter where pytest is ran, we set the working dir
    to this testscript's parent directory"""

    monkeypatch.chdir(request.fspath.dirname)


@pytest.mark.parametrize("inertia", ["entities", "nodes"])
def test_the_period_of_the_rkpm_cantilever_is_that_of_the_continuum(inertia):
    """Coarse: four particles over the height. The period is that of the finite element cantilever to a few percent,
    and longer than the Euler-Bernoulli beam's by the shear deformation and the rotary inertia."""

    referencePeriod = periodOfTheResponse(*run_finite_elements(0.25))
    beamPeriod = eulerBernoulliPeriod()
    assert referencePeriod == pytest.approx(beamPeriod, rel=3e-2)

    time, deflection = run_rkpm(0.25, inertia)
    # it oscillates about the static deflection, downwards
    assert np.all(deflection <= 1e-12)
    assert periodOfTheResponse(time, deflection) == pytest.approx(referencePeriod, rel=5e-2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", action="store_true", help="print the periods under refinement.")
    args = parser.parse_args()

    beamPeriod = eulerBernoulliPeriod()
    print("Euler-Bernoulli period: {:.6f}".format(beamPeriod))
    for size in (0.25, 0.125):
        period = periodOfTheResponse(*run_finite_elements(size))
        print(
            "CPE8UL, element size {:}: period {:.6f} ({:+.3%} vs Euler-Bernoulli)".format(
                size, period, period / beamPeriod - 1
            )
        )
    sizes = (0.25, 0.125, 1.0 / 12) if args.study else (0.25,)
    for inertia in ("entities", "nodes"):
        for size in sizes:
            period = periodOfTheResponse(*run_rkpm(size, inertia))
            print(
                "RKPM, inertia={:}, particle size {:.4f}: period {:.6f} ({:+.3%} vs Euler-Bernoulli)".format(
                    inertia, size, period, period / beamPeriod - 1
                )
            )
