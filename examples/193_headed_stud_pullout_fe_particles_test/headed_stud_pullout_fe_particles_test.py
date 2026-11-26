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
"""A toy pull-out: the head of a stud of finite elements, embedded in a block of particles, is pulled out, and bears
on the particles above it through frictionless contact.

.. code-block:: text

                 pulled up
                     ^
        ===##====|   |====##===    support ring on the top of the block (##), the shaft's hole (| |)
           ......|   |.......
           ......|   |.......      particles
           ..+-----------+...
           ..|   head    |...      finite elements
           ..+-----------+...
           ..................
           ..................     the sides of the block held in x

Plane strain, both solids elastic (compressible neo-Hookean). The shaft is not modelled: the nodes of the head under
it are pulled. The contact points are the quadrature points on the faces of the particles around the head
(:class:`~edelweissmeshfree.constraints.base.contactpointsonparticlefaces.ContactPointsOnParticleFaces`), the master
the faces of the head, in EdelweissFE's contact constraint (:mod:`edelweissfe.constraints.surfacetodeformablesurfacepenalty`).
The step is quasi-static, solved by EdelweissFE's ``NISTParallel``, which uses the tangent of the contact.

What it checks: the head bears on the particles above it and separates from those below it; the force of the contact
on the head is the pull force; the response of the elastic solids is nearly linear; halving the particle size changes
the pull force by little.
"""

import numpy as np
import pytest
from edelweissfe.config.configurator import loadConfiguration
from edelweissfe.constraints.surfacetodeformablesurfacepenalty import (
    Constraint as SurfaceContact,
)
from edelweissfe.constraints.surfacetodeformablesurfacepenalty import (
    SurfaceToDeformableSurfacePenaltySchema,
)
from edelweissfe.generators.planerectquad import Generator as PlaneRectQuadGenerator
from edelweissfe.generators.planerectquad import PlaneRectQuadSchema
from edelweissfe.generators.surfaceelementgenerator import buildContactFacets
from edelweissfe.journal.journal import Journal
from edelweissfe.outputmanagers.base.outputmanagerbase import OutputManagerBase
from edelweissfe.sections.plane import PlaneSectionSchema
from edelweissfe.sections.plane import Section as PlaneSection
from edelweissfe.sets.elementset import ElementSet
from edelweissfe.sets.nodeset import NodeSet
from edelweissfe.solvers.nonlinearimplicitstaticparallel import NISTParallel
from edelweissfe.stepactions.dirichlet import StepAction as Dirichlet
from edelweissfe.steps.adaptivestep import AdaptiveStep
from edelweissfe.steps.stepmanager import StepActionCollection

from edelweissmeshfree.constraints.base.contactpointsonparticlefaces import (
    ContactPointsOnParticleFaces,
)
from edelweissmeshfree.constraints.particlepenaltyweakdirichtlet import (
    ParticlePenaltyWeakDirichlet,
)
from edelweissmeshfree.fieldoutput.fieldoutput import MPMFieldOutputController
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
from edelweissmeshfree.sets.particleset import ParticleSet

#: The block of particles, x in [-HALF_WIDTH, HALF_WIDTH], y in [BOTTOM, TOP]; the head, x in [-HEAD, HEAD], y in
#: [-HEAD_HEIGHT, 0]; the shaft's hole, x in [-SHAFT, SHAFT], y in [0, TOP]; the support ring, |x| >= RING. In mm.
HALF_WIDTH, BOTTOM, TOP = 30.0, -30.0, 20.0
HEAD, HEAD_HEIGHT, SHAFT, RING = 8.0, 4.0, 2.0, 14.0
#: The pull, in four increments.
PULL = 0.02


def neoHooke(youngsModulus: float, poissonsRatio: float) -> dict:
    """A compressible neo-Hookean solid: bulk modulus, shear modulus, density."""

    K, G = youngsModulus / (3 * (1 - 2 * poissonsRatio)), youngsModulus / (2 * (1 + poissonsRatio))
    return {"name": "CompressibleNeoHooke", "properties": np.array([K, G, 1.0])}


STEEL, CONCRETE = neoHooke(210000.0, 0.3), neoHooke(30000.0, 0.2)
#: The interface stiffness per unit area of the contact, and the penalty of the supports of the particles.
CONTACT_PENALTY, SUPPORT_PENALTY = 1e6, 1e7


class PullForce(OutputManagerBase):
    """Records the pull and the pull force after every increment."""

    checkpointedState = {}

    def __init__(self, model: MPMModel, pulledNodes: NodeSet):
        self.name = "pullForce"
        self._model, self._pulledNodes = model, pulledNodes
        self.pulls, self.forces = [0.0], [0.0]

    def initializeJob(self):
        pass

    def initializeStep(self, step):
        pass

    def finalizeIncrement(self, **kwargs):
        displacement = self._model.nodeFields["displacement"]
        rows = [displacement.indexOfNode(node) for node in self._pulledNodes]
        self.pulls.append(float(displacement["U"][rows, 1].mean()))
        self.forces.append(float(displacement["P"][rows, 1].sum()))

    def finalizeFailedIncrement(self, **kwargs):
        pass

    def finalizeStep(self):
        pass

    def finalizeJob(self):
        pass


def addTheHead(model: MPMModel, journal: Journal, size: float) -> tuple[ElementSet, NodeSet]:
    """The head of quadratic plane strain elements, ``CPE8UL``, with its section; returns the contact facets of its
    boundary and the nodes of its top under the shaft."""

    with model.topology.changes():
        PlaneRectQuadGenerator(
            "head",
            model,
            journal,
            configuration=PlaneRectQuadSchema(
                x0=-HEAD, y0=-HEAD_HEIGHT, length=2 * HEAD, h=HEAD_HEIGHT, nX=round(2 * HEAD / size),
                nY=round(HEAD_HEIGHT / size), elType="CPE8UL",
            ),
        )  # fmt: skip
        facets = []
        for side in ("top", "bottom", "left", "right"):
            facetSet, _ = buildContactFacets(
                model, f"head_{side}", f"head_{side}", "midside", "facetConsistent", journal
            )
            facets += list(model.elementSets[facetSet])
    model.sections["head"] = PlaneSection(
        "head", model, STEEL, [model.elementSets["head_all"]], configuration=PlaneSectionSchema(thickness=1.0)
    )
    # strictly under the shaft: the faces of the head the particles bear on begin at the shaft's edge
    underTheShaft = [node for node in model.nodeSets["head_top"] if abs(node.coordinates[0]) < SHAFT - 1e-9]
    return ElementSet("head_boundary", facets), NodeSet("underTheShaft", underTheShaft)


def addTheBlock(model: MPMModel, journal: Journal, size: float) -> list:
    """The block of displacement particles of the given size, without the head and the shaft's hole, on kernel
    functions matched to them; returns the (particle, face) pairs on the boundary of the head's cavity."""

    approximation = MarmotMeshfreeApproximationWrapper("ReproducingKernelImplicitGradient", 2, completenessOrder=1)
    material = {"material": CONCRETE["name"], "properties": CONCRETE["properties"]}

    def inTheCavity(x, y):
        return (abs(x) < HEAD and -HEAD_HEIGHT < y < 0.0) or (abs(x) < SHAFT and y > 0.0)

    particles, faces = [], []
    for x0 in np.arange(-HALF_WIDTH, HALF_WIDTH - 0.5 * size, size):
        for y0 in np.arange(BOTTOM, TOP - 0.5 * size, size):
            xc, yc = x0 + 0.5 * size, y0 + 0.5 * size
            if inTheCavity(xc, yc):
                continue
            vertices = np.array([[x0, y0], [x0 + size, y0], [x0 + size, y0 + size], [x0, y0 + size]])
            number = len(particles) + 1
            particle = MarmotParticleWrapper(
                "DisplacementSQCNIxNSNI/PlaneStrain/Quad", number, vertices, 0.0, approximation, material
            )
            model.particles[number] = particle
            particles.append(particle)
            # the faces on the head's cavity (Abaqus numbering: 1 bottom, 2 right, 3 top, 4 left)
            for faceID, (dx, dy) in {1: (0.0, -size), 2: (size, 0.0), 3: (0.0, size), 4: (-size, 0.0)}.items():
                if inTheCavity(xc + dx, yc + dy) and (abs(xc + dx) < HEAD and -HEAD_HEIGHT < yc + dy < 0.0):
                    faces.append((particle, faceID))

    model.particleSets["block_all"] = ParticleSet("block_all", particles)
    generateKernelMatchingToParticle(
        model,
        journal,
        lambda node, supportSize: MarmotMeshfreeKernelFunctionWrapper(
            node, "BSplineBoxed", supportRadius=supportSize, continuityOrder=3
        ),
        model.particleSets["block_all"],
        supportScalingFactor=2.0,
        firstKernelFunctionNumber=max(model.nodes) + 1,
    )
    domain = ParticleKernelDomain(particles, list(model.meshfreeKernelFunctions.values()))
    model.particleKernelDomains["block"] = domain
    model.particleManagers.append(
        KDBinOrganizedParticleManager(domain, 2, journal, bondParticlesToKernelFunctions=True)
    )

    # the support ring holds the top vertices of the particles at |x| >= RING in y, the sides of the block hold their
    # outer vertices in x
    ring = [
        p for p in particles if p.getCenterCoordinates()[1] > TOP - size and abs(p.getCenterCoordinates()[0]) > RING
    ]
    left = [p for p in particles if p.getCenterCoordinates()[0] < -HALF_WIDTH + size]
    right = [p for p in particles if p.getCenterCoordinates()[0] > HALF_WIDTH - size]
    for name, held, component, vertices in (
        ("ring", ring, 1, [2, 3]),
        ("left", left, 0, [0, 3]),
        ("right", right, 0, [1, 2]),
    ):
        model.constraints[name] = ParticlePenaltyWeakDirichlet(
            name, model, held, "displacement", {component: 0.0}, SUPPORT_PENALTY * size, vertices
        )
    return faces


def run(size: float) -> tuple[PullForce, SurfaceContact]:
    """Pull the head out of a block of particles of the given size.

    Returns
    -------
    tuple[PullForce, SurfaceContact]
        The pull force after every increment, and the contact.
    """

    journal = Journal(verbose=False)
    model = MPMModel(2)
    headBoundary, underTheShaft = addTheHead(model, journal, min(size, 1.0))
    faces = addTheBlock(model, journal, size)
    contact = SurfaceContact(
        "contact",
        model,
        ContactPointsOnParticleFaces(faces, nQuadraturePoints=2),
        headBoundary,
        journal,
        configuration=SurfaceToDeformableSurfacePenaltySchema(penalty=CONTACT_PENALTY, searchDistance=0.5),
    )
    model.constraints["contact"] = contact
    model.prepareYourself(journal)

    stepActions = StepActionCollection()
    stepActions["dirichlet"]["pull"] = Dirichlet(
        "pull", underTheShaft, "displacement", {0: 0.0, 1: PULL}, model, journal
    )
    pullForce = PullForce(model, underTheShaft)
    fieldOutputController = MPMFieldOutputController(model, journal)
    jobInfo = loadConfiguration({})
    step = AdaptiveStep(
        0,
        model,
        fieldOutputController,
        journal,
        jobInfo,
        NISTParallel(jobInfo, journal, linsolver="pardiso"),
        [pullForce],
        stepActions,
        startInc=0.25,
        maxInc=0.25,
        minInc=1e-3,
        maxNumInc=100,
    )
    fieldOutputController.initializeJob()
    try:
        step.solve()
    finally:
        fieldOutputController.finalizeJob()
    return pullForce, contact


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    """No matter where pytest is ran, we set the working dir to this testscript's parent directory."""

    monkeypatch.chdir(request.fspath.dirname)


def test_the_head_bears_on_the_particles_above_it():
    pullForce, contact = run(2.0)
    pulls, forces = np.array(pullForce.pulls), np.array(pullForce.forces)
    pulled = pulls > 0.0

    assert np.count_nonzero(pulled) == 4 and forces[-1] > 0.0
    # the contact force on the head is the pull force
    assert -contact.contactForceOnMaster[1] == pytest.approx(forces[-1], rel=1e-6)
    np.testing.assert_allclose(contact.contactForceOnSlave + contact.contactForceOnMaster, 0.0, atol=1e-9 * forces[-1])
    # the particles above the head carry, those below it are released
    closed = contact.getNormalPressures() > 0.0
    y = contact.slave.currentPointCoordinates(None)[:, 1]
    assert (closed & (y > -1e-3)).any() and not (closed & (y < -HEAD_HEIGHT + 1e-3)).any()
    # nearly linear: the secant stiffness changes by a few percent only, as the area in contact grows
    secant = forces[pulled] / pulls[pulled]
    np.testing.assert_allclose(secant, secant[0], rtol=0.05)


def test_halving_the_particle_size_changes_the_pull_force_little():
    coarse, _ = run(2.0)
    fine, _ = run(1.0)
    print(
        f"pull force at {PULL} mm: particles of 2 mm {coarse.forces[-1]:.4e} N/mm, of 1 mm {fine.forces[-1]:.4e} N/mm"
    )
    assert coarse.forces[-1] == pytest.approx(fine.forces[-1], rel=0.1)


if __name__ == "__main__":
    for size in (2.0, 1.0):
        pullForce, contact = run(size)
        print(size, pullForce.pulls, pullForce.forces)
