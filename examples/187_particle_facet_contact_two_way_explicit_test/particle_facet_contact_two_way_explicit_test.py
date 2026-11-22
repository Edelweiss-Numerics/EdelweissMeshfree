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
"""Two-way contact between particles and finite elements in one explicit analysis
(:class:`~edelweissmeshfree.constraints.explicit.particlepenaltyfacetcontactexplicit.ParticlePenaltyFacetContactExplicit`
with ``twoWayCoupling``), integrated by EdelweissFE's ``NED``:

- a particle block falls onto a free-floating plate of finite elements, whose only load is the contact: the impulses
  the contact gives to the two bodies are equal and opposite in every increment;
- an elastic block falls onto a supported elastic block: with an impactor of particles, the contact force history and
  the rebound agree with those of an impactor of finite elements, which meets the target through EdelweissFE's
  node-to-surface contact (:mod:`edelweissfe.constraints.nodetodeformablesurfacepenalty`);
- halving the time increment changes the penetration history by less than the stated tolerance.
"""

import argparse

import numpy as np
import pytest
from edelweissfe.config.configurator import loadConfiguration
from edelweissfe.constraints.nodetodeformablesurfacepenalty import (
    Constraint as NodeToDeformableSurfacePenalty,
)
from edelweissfe.constraints.nodetodeformablesurfacepenalty import (
    NodeToDeformableSurfacePenaltySchema,
)
from edelweissfe.generators.planerectquad import Generator as PlaneRectQuadGenerator
from edelweissfe.generators.planerectquad import PlaneRectQuadSchema
from edelweissfe.generators.surfaceelementgenerator import buildContactFacets
from edelweissfe.journal.journal import Journal
from edelweissfe.outputmanagers.base.outputmanagerbase import OutputManagerBase
from edelweissfe.sections.plane import PlaneSectionSchema
from edelweissfe.sections.plane import Section as PlaneSection
from edelweissfe.solvers.nonlinearexplicitdynamic import NED
from edelweissfe.stepactions.dirichlet import StepAction as Dirichlet
from edelweissfe.steps.adaptivestepforexplicitsimulations import (
    AdaptiveStepForExplicitSimulations,
)
from edelweissfe.steps.stepmanager import StepActionCollection

from edelweissmeshfree.constraints.explicit.particlepenaltyfacetcontactexplicit import (
    ParticlePenaltyFacetContactExplicit,
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

#: A compressible neo-Hookean solid, steel-like, in N, mm, t, s: bulk modulus, shear modulus, density.
YOUNGS_MODULUS, POISSONS_RATIO, DENSITY = 210000.0, 0.3, 7.85e-9
MATERIAL = {
    "name": "CompressibleNeoHooke",
    "properties": np.array(
        [
            YOUNGS_MODULUS / (3 * (1 - 2 * POISSONS_RATIO)),
            YOUNGS_MODULUS / (2 * (1 + POISSONS_RATIO)),
            DENSITY,
        ]
    ),
}

#: The impactor: a block of 20 x 20 mm, its bottom 0.05 mm above the target, falling at 10 m/s: it touches after 5 µs.
IMPACTOR_SIZE, STANDOFF, IMPACT_VELOCITY = 20.0, 0.05, -10e3
#: The Courant number of NED, its default.
COURANT_NUMBER = NED.SolverSpecificOptions["courant-number"]
#: The size of the particles and of the finite elements of the impactor and the target.
ENTITY_SIZE = 2.5
#: The interface stiffness per unit area of the finite element contact; the particle contact takes it per contact point,
#: times the length a point stands for: two vertices meet at every point of the bottom of the particle block, half a
#: particle apart, so a point stands for half a particle.
PENALTY_PER_AREA = 2e5
PENALTY_PER_POINT = PENALTY_PER_AREA * ENTITY_SIZE / 2
SEARCH_DISTANCE = 1.0


class ContactHistory(OutputManagerBase):
    """Records, in every increment, the time and what a contact constraint reports: the forces on its two sides, the
    deepest penetration, and the velocity of the impactor."""

    checkpointedState = {}

    def __init__(self, model, contactForces, deepestPenetration, impactorVelocity):
        self.name = "contactHistory"
        self._model = model
        self._contactForces = contactForces
        self._deepestPenetration = deepestPenetration
        self._impactorVelocity = impactorVelocity
        self.times, self.forces, self.penetrations, self.velocities = [], [], [], []

    def initializeJob(self):
        pass

    def initializeStep(self, step):
        pass

    def finalizeIncrement(self, **kwargs):
        self.times.append(self._model.time)
        self.forces.append(self._contactForces())
        self.penetrations.append(self._deepestPenetration())
        self.velocities.append(self._impactorVelocity())

    def finalizeFailedIncrement(self, **kwargs):
        pass

    def finalizeStep(self):
        pass

    def finalizeJob(self):
        pass


def addFiniteElementBlock(
    model: MPMModel, journal: Journal, name: str, x0: float, y0: float, width: float, height: float
):
    """A block of quadratic plane strain elements, ``CPE8UL``, with its section and its node and element sets."""

    with model.topology.changes():
        PlaneRectQuadGenerator(
            name,
            model,
            journal,
            configuration=PlaneRectQuadSchema(
                x0=x0,
                y0=y0,
                length=width,
                h=height,
                nX=round(width / ENTITY_SIZE),
                nY=round(height / ENTITY_SIZE),
                elType="CPE8UL",
            ),
        )
    model.sections[name] = PlaneSection(
        name, model, MATERIAL, [model.elementSets[f"{name}_all"]], configuration=PlaneSectionSchema(thickness=1.0)
    )


def addContactFacets(model: MPMModel, journal: Journal, surface: str, name: str):
    """The Line2 contact facets of a surface of a block of finite elements, ``<name>_facets``: two per element edge,
    through its midside node."""

    with model.topology.changes():
        buildContactFacets(model, surface, name, "midside", "facetConsistent", journal)


def addParticleBlock(model: MPMModel, journal: Journal, name: str, x0: float, y0: float, width: float, height: float):
    """A block of particles, ``Displacement/SQCNIxNSNI/PlaneStrain/Quad``, with its kernel functions and its particle
    manager."""

    grid = dict(
        x0=x0, y0=y0, h=height, l=width, nX=round(width / ENTITY_SIZE), nY=round(height / ENTITY_SIZE), name=name
    )
    generateRectangularKernelFunctionGrid(
        model,
        journal,
        lambda node: MarmotMeshfreeKernelFunctionWrapper(
            node, "BSplineBoxed", supportRadius=3.0 * ENTITY_SIZE, continuityOrder=3
        ),
        firstKernelFunctionNumber=max(model.nodes, default=0) + 1,
        **grid,
    )
    approximation = MarmotMeshfreeApproximationWrapper("ReproducingKernel", 2, completenessOrder=1)
    material = {"material": MATERIAL["name"], "properties": MATERIAL["properties"]}
    generateRectangularQuadParticleGrid(
        model,
        journal,
        lambda number, vertices, volume: MarmotParticleWrapper(
            "Displacement/SQCNIxNSNI/PlaneStrain/Quad", number, vertices, 0.0, approximation, material
        ),
        **grid,
    )
    domain = ParticleKernelDomain(list(model.particles.values()), list(model.meshfreeKernelFunctions.values()))
    model.particleKernelDomains[name] = domain
    model.particleManagers.append(
        KDBinOrganizedParticleManager(domain, 2, journal, bondParticlesToKernelFunctions=True)
    )


def run_sim(
    impactor: str, target: str, courantNumber: float = COURANT_NUMBER, duration: float = 40e-6
) -> ContactHistory:
    """Let an impactor fall onto a target.

    Parameters
    ----------
    impactor
        "particles" or "finite elements".
    target
        "free plate": a free-floating plate, 60 x 5 mm; "supported block": a block of 60 x 20 mm on a support.
    courantNumber
        The Courant number of NED, which sets the time increment.
    duration
        The duration of the step.

    Returns
    -------
    ContactHistory
        The history of the contact.
    """

    journal = Journal(verbose=False)
    model = MPMModel(2)

    targetHeight = 5.0 if target == "free plate" else 20.0
    addFiniteElementBlock(model, journal, "target", -30.0, -targetHeight, 60.0, targetHeight)
    addContactFacets(model, journal, "target_top", "targetTop")

    x0Impactor = -0.5 * IMPACTOR_SIZE
    if impactor == "particles":
        addParticleBlock(model, journal, "impactor", x0Impactor, STANDOFF, IMPACTOR_SIZE, IMPACTOR_SIZE)
        contact = ParticlePenaltyFacetContactExplicit(
            "contact",
            model,
            list(model.particleSets["impactor_bottom"]),
            list(model.elementSets["targetTop_facets"]),
            PENALTY_PER_POINT,
            SEARCH_DISTANCE,
            journal,
        )

        def contactForces():
            return np.array([contact.contactForceOnParticles, contact.contactForceOnMaster])

        def deepestPenetration():
            return contact.deepestPenetration

        def impactorVelocity():
            # the particles have the same mass
            return np.mean([particle.getResultArray("velocity")[1] for particle in model.particleSets["impactor_all"]])

    else:
        addFiniteElementBlock(model, journal, "impactor", x0Impactor, STANDOFF, IMPACTOR_SIZE, IMPACTOR_SIZE)
        addContactFacets(model, journal, "impactor_bottom", "impactorBottom")
        contact = NodeToDeformableSurfacePenalty(
            "contact",
            model,
            model.elementSets["impactorBottom_facets"],
            model.elementSets["targetTop_facets"],
            journal,
            configuration=NodeToDeformableSurfacePenaltySchema(
                slaveSurface="impactorBottom_facets",
                masterSurface="targetTop_facets",
                penalty=PENALTY_PER_AREA,
                searchDistance=SEARCH_DISTANCE,
                sliding="small",
            ),
        )

        def contactForces():
            return np.array([[0.0, contact.totalNormalForce], [0.0, -contact.totalNormalForce]])

        def deepestPenetration():
            gaps = contact.getGaps()
            return float(max(0.0, -gaps.min())) if len(gaps) else 0.0

        def impactorVelocity():
            V = model.nodeFields["displacement"]["V"]
            nodes = model.nodeSets["impactor_all"]
            return np.mean([V[model.nodeFields["displacement"].indexOfNode(node)][1] for node in nodes])

    model.constraints["contact"] = contact
    model.prepareYourself(journal)

    # the impactor falls; the finite element nodes carry the velocity in the model's node field, the particles in their
    # state
    if impactor == "particles":
        for particle in model.particleSets["impactor_all"]:
            particle.getResultArray("velocity")[1] = IMPACT_VELOCITY
            particle.acceptStateAndPosition()
    else:
        displacement = model.nodeFields["displacement"]
        for node in model.nodeSets["impactor_all"]:
            displacement["V"][displacement.indexOfNode(node), 1] = IMPACT_VELOCITY

    stepActions = StepActionCollection()
    if target == "supported block":
        stepActions["dirichlet"]["support"] = Dirichlet(
            "support", model.nodeSets["target_bottom"], "displacement", {0: 0.0, 1: 0.0}, model, journal
        )

    history = ContactHistory(model, contactForces, deepestPenetration, impactorVelocity)
    fieldOutputController = MPMFieldOutputController(model, journal)
    fieldOutputController.initializeJob()
    jobInfo = loadConfiguration({})
    step = AdaptiveStepForExplicitSimulations(
        0,
        model,
        fieldOutputController,
        journal,
        jobInfo,
        # the contact history is recorded in every increment
        NED(
            jobInfo, journal, **{"courant-number": courantNumber, "contact-update-frequency": 1, "output-frequency": 1}
        ),
        [history],
        stepActions,
        stepLength=duration,
        startInc=1.0,
        maxInc=1.0,
        minInc=1e-12,
        maxNumInc=10**6,
    )
    try:
        step.solve()
    finally:
        fieldOutputController.finalizeJob()

    return history


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    """No matter where pytest is ran, we set the working dir to this testscript's parent directory."""

    monkeypatch.chdir(request.fspath.dirname)


def test_the_contact_gives_equal_and_opposite_impulses_to_a_free_floating_plate():
    history = run_sim("particles", "free plate")

    forces = np.array(history.forces)  # increment x (particles, master) x component
    largestForce = np.abs(forces).max()
    assert largestForce > 0.0
    np.testing.assert_allclose(forces[:, 0] + forces[:, 1], 0.0, atol=1e-12 * largestForce)

    impulses = np.cumsum(forces * np.diff(history.times, prepend=0.0)[:, None, None], axis=0)
    np.testing.assert_allclose(impulses[:, 0] + impulses[:, 1], 0.0, atol=1e-12 * np.abs(impulses).max())

    # the impulse the particle block received is the change of its momentum (it has no other load)
    massOfTheImpactor = DENSITY * IMPACTOR_SIZE**2
    changeOfMomentum = massOfTheImpactor * (history.velocities[-1] - IMPACT_VELOCITY)
    assert impulses[-1, 0, 1] == pytest.approx(changeOfMomentum, rel=1e-3)


def test_an_impactor_of_particles_rebounds_as_one_of_finite_elements(assert_gold):
    ofParticles = run_sim("particles", "supported block")
    assert_gold(np.array(ofParticles.forces)[:, 0, 1], np.loadtxt("gold.csv"), atol=1e-6)

    ofFiniteElements = run_sim("finite elements", "supported block")

    peakForceOfParticles = np.abs(np.array(ofParticles.forces)[:, 0, 1]).max()
    peakForceOfFiniteElements = np.abs(np.array(ofFiniteElements.forces)[:, 0, 1]).max()
    assert peakForceOfParticles == pytest.approx(peakForceOfFiniteElements, rel=0.1)

    reboundOfParticles, reboundOfFiniteElements = ofParticles.velocities[-1], ofFiniteElements.velocities[-1]
    assert reboundOfFiniteElements > 0.0
    assert reboundOfParticles == pytest.approx(reboundOfFiniteElements, rel=0.1)
    # a penalty spring does no work: the rebound has at most the energy of the impact
    assert reboundOfParticles <= abs(IMPACT_VELOCITY)


def test_halving_the_time_increment_changes_the_impact_little():
    """The response of the bodies converges with the time increment: the velocity of the impactor, the impulse of the
    contact and the deepest penetration. The penetration of the deepest point increment by increment does not, by
    the same measure: single contact points ring at the frequency of their penalty spring, and that history converges
    only at first order in the time increment."""

    coarse = run_sim("particles", "supported block", courantNumber=COURANT_NUMBER)
    fine = run_sim("particles", "supported block", courantNumber=0.5 * COURANT_NUMBER)

    velocityOfFineAtCoarseTimes = np.interp(coarse.times, fine.times, fine.velocities)
    np.testing.assert_allclose(coarse.velocities, velocityOfFineAtCoarseTimes, atol=0.02 * abs(IMPACT_VELOCITY))

    def impulse(history):
        return np.trapezoid(np.array(history.forces)[:, 0, 1], history.times)

    assert impulse(coarse) == pytest.approx(impulse(fine), rel=0.005)
    assert max(coarse.penetrations) == pytest.approx(max(fine.penetrations), rel=0.03)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("impactor", choices=["particles", "finite elements"])
    parser.add_argument("target", choices=["free plate", "supported block"])
    parser.add_argument("--courant-number", type=float, default=COURANT_NUMBER)
    parser.add_argument(
        "--create-gold", action="store_true", help="write the contact force history of this run to gold.csv."
    )
    args = parser.parse_args()

    history = run_sim(args.impactor, args.target, args.courant_number)
    forces = np.array(history.forces)
    print("increments: {:}, peak force on the impactor: {:.4e}".format(len(history.times), np.abs(forces[:, 0]).max()))
    print("largest |action + reaction|: {:.3e}".format(np.abs(forces[:, 0] + forces[:, 1]).max()))
    print(
        "deepest penetration: {:.4e}, final impactor velocity: {:.4e}".format(
            max(history.penetrations), history.velocities[-1]
        )
    )

    if args.create_gold:
        np.savetxt("gold.csv", forces[:, 0, 1])
