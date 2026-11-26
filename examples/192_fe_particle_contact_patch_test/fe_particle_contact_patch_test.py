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
"""The contact patch test, with particles and with finite elements: a block between two platens of finite elements is
compressed, and the contact must transmit a uniform pressure across both interfaces.

.. code-block:: text

        |  |  |  |  |  |  |  |    the top of the upper platen moves down
        +----------------------+
        |    upper platen      |
        +======================+  contact
        |                      |
        |   block: particles   |
        |   or finite elements |
        |                      |
        +======================+  contact
        |    lower platen      |
        +----------------------+
        ///////////////////////   the bottom of the lower platen is held

The platens and the block are elastic, the platens finite elements (``CPE8UL``), the block a compressible neo-Hookean
solid, all without lateral contraction (Poisson's ratio zero), the block held only on its left side, in x. Its state
is homogeneous: the pressure is the same at every contact point of both interfaces, and their resultant is the
reaction of the platens. (With lateral contraction the block slides along the platens, its contact points no longer
meet the faces of the platens as they did, and the integrated contact transmits the pressure to within a few parts in
a thousand only: pointwise contact on a non-matching interface, see EdelweissFE's contact theory.) The same contact constraint
(:mod:`edelweissfe.constraints.surfacetodeformablesurfacepenalty`) serves both blocks; only its contact points differ:
the quadrature points of the faces of the elements, or of the faces of the particles
(:class:`~edelweissmeshfree.constraints.base.contactpointsonparticlefaces.ContactPointsOnParticleFaces`). The block
starts with a tiny overlap of the platens, so that the contact holds it from the first iteration on.

The step is quasi-static, solved by EdelweissFE's ``NISTParallel``: the implicit path of the contact, with its tangent.

**The block of finite elements passes**: the pressure is uniform at every contact point.

**The block of particles passes away from its free ends, and not at them.** Three loads meet there:

- *The particles' own load.* The particles balance a homogeneous stress with the traction on their faces distributed
  with their shape functions at the center of each face, :math:`N(\\boldsymbol{x}_f)`, one point per face: their
  smoothed gradients integrate :math:`N` over their faces with that rule, and a particle distributes its own pressure
  on a face so. In a homogeneous state, the internal forces of the particles are their own pressure loads on their
  boundary faces, at every node, the outermost columns included
  (``test_the_particles_balance_a_homogeneous_stress_with_their_own_pressure_at_every_node``). Only this load keeps
  the homogeneous state.
- *The contact's load.* The contact distributes its force with :math:`G`, the particle's shape functions linearized
  about its center, with which the particle moves its faces (see the module of the contact points). :math:`G` is
  affine, so a uniform pressure loads the particles alike at one or at two points per face. That load differs from
  the particles' own by a load without resultant and moment, which repeats from one column of kernel nodes to the next
  along the row of particles, without resultant in each column: it changes the pressure alike along the row. Only the
  two outermost columns at each free end, where the kernel functions are no longer translates of each other, differ
  (``test_the_contact_loads_the_particles_as_their_own_pressure_does_except_at_the_free_ends``). The shape functions
  at the points, :math:`N`, would not give the particles' own load either at two points per face: :math:`N` is not
  affine, so two points of it are not :math:`N` at the center.
- *Two points per face.* The platens are quadratic, and the contact loads them with the shape functions of their
  faces, which two points per face of the same size integrate exactly; one point at the center of a face would load
  only the midside nodes of the platen, and leave the force short. With two points, the springs also hold the tilt of
  a face: at the ends the pressures of the two points of a face differ by more than their mean deviates.

So the pressure deviates from uniform by a few percent near each free end: near the right end, which nothing holds,
as near the left end, which the support holds in x, by mirrored profiles. It is the difference of the contact's load
from the particles' own at a free end, not a discretization error that shrinks with the particles: at half the particle
size the deviation has the same profile against the distance from the end in particles, in a band half as wide; beyond
the band, the pressure is uniform within a percent, and the force converges to that of the elements as the band
narrows (``test_the_particles_deviate_from_the_uniform_pressure_only_near_their_free_ends``).
"""

import numpy as np
import pytest
from edelweissfe.config.configurator import loadConfiguration
from edelweissfe.constraints.base.contactpointsonslavesurface import (
    ContactPointsOnSlaveSurface,
)
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
from edelweissfe.sections.plane import PlaneSectionSchema
from edelweissfe.sections.plane import Section as PlaneSection
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

#: The block, 40 x 20 mm, of particles or elements of 5 mm (by default); the platens 40 x 10 mm.
WIDTH, HEIGHT, PLATEN, SIZE = 40.0, 20.0, 10.0, 5.0
#: The block's overlap of each platen at the start, and the compression of the step.
OVERLAP, COMPRESSION = 1e-4, 0.05


def neoHooke(youngsModulus: float, poissonsRatio: float) -> dict:
    """A compressible neo-Hookean solid: bulk modulus, shear modulus, density."""

    return {
        "name": "CompressibleNeoHooke",
        "properties": np.array(
            [
                youngsModulus / (3 * (1 - 2 * poissonsRatio)),
                youngsModulus / (2 * (1 + poissonsRatio)),
                1.0,
            ]
        ),
    }


PLATEN_MATERIAL, BLOCK_MATERIAL = neoHooke(210000.0, 0.0), neoHooke(30000.0, 0.0)
#: The interface stiffness per unit area.
PENALTY = 1e6


def addFiniteElementBlock(
    model: MPMModel, journal: Journal, name: str, y0: float, height: float, material: dict, size: float
):
    """A block of ``CPE8UL`` elements of the given size, ``WIDTH`` wide, from ``y0`` up, with its section."""

    with model.topology.changes():
        PlaneRectQuadGenerator(
            name,
            model,
            journal,
            configuration=PlaneRectQuadSchema(
                x0=0.0, y0=y0, length=WIDTH, h=height, nX=round(WIDTH / size), nY=round(height / size), elType="CPE8UL"
            ),
        )
    model.sections[name] = PlaneSection(
        name, model, material, [model.elementSets[f"{name}_all"]], configuration=PlaneSectionSchema(thickness=1.0)
    )


def facets(model: MPMModel, journal: Journal, surface: str) -> str:
    """The contact facets of a surface of finite elements, ``<surface>_facets``."""

    with model.topology.changes():
        return buildContactFacets(model, surface, surface, "midside", "facetConsistent", journal)[0]


def addParticleBlock(model: MPMModel, journal: Journal, y0: float, size: float):
    """The block of displacement particles of the given size, on kernel functions bonded to them."""

    grid = dict(x0=0.0, y0=y0, l=WIDTH, h=HEIGHT + 2 * OVERLAP, nX=round(WIDTH / size), nY=round(HEIGHT / size))
    generateRectangularKernelFunctionGrid(
        model,
        journal,
        lambda node: MarmotMeshfreeKernelFunctionWrapper(
            node, "BSplineBoxed", supportRadius=2 * size, continuityOrder=3
        ),
        name="block",
        firstKernelFunctionNumber=max(model.nodes, default=0) + 1,
        **grid,
    )
    approximation = MarmotMeshfreeApproximationWrapper("ReproducingKernelImplicitGradient", 2, completenessOrder=1)
    material = {"material": BLOCK_MATERIAL["name"], "properties": BLOCK_MATERIAL["properties"]}
    generateRectangularQuadParticleGrid(
        model,
        journal,
        lambda number, vertices, volume: MarmotParticleWrapper(
            "DisplacementSQCNIxNSNI/PlaneStrain/Quad", number, vertices, 0.0, approximation, material
        ),
        name="block",
        **grid,
    )
    domain = ParticleKernelDomain(list(model.particles.values()), list(model.meshfreeKernelFunctions.values()))
    model.particleKernelDomains["block"] = domain
    model.particleManagers.append(
        KDBinOrganizedParticleManager(domain, 2, journal, bondParticlesToKernelFunctions=True)
    )


def run(block: str, size: float = SIZE) -> tuple[MPMModel, list]:
    """Compress the block between the platens.

    Parameters
    ----------
    block
        "particles" or "finite elements".
    size
        The size of the particles and of the elements.

    Returns
    -------
    tuple[MPMModel, list]
        The solved model, and the two contact constraints (lower, upper).
    """

    journal = Journal(verbose=False)
    model = MPMModel(2)
    addFiniteElementBlock(model, journal, "lower", -PLATEN, PLATEN, PLATEN_MATERIAL, size)
    addFiniteElementBlock(model, journal, "upper", HEIGHT, PLATEN, PLATEN_MATERIAL, size)
    lowerTop, upperBottom = facets(model, journal, "lower_top"), facets(model, journal, "upper_bottom")

    stepActions = StepActionCollection()
    if block == "particles":
        addParticleBlock(model, journal, -OVERLAP, size)
        bottom = ContactPointsOnParticleFaces(model.surfaces["block_bottom"], nQuadraturePoints=2)
        top = ContactPointsOnParticleFaces(model.surfaces["block_top"], nQuadraturePoints=2)
        # the left vertices of the particles on the left side, held in x
        model.constraints["blockLeft"] = ParticlePenaltyWeakDirichlet(
            "blockLeft", model, list(model.particleSets["block_left"]), "displacement", {0: 0.0}, PENALTY * size, [0, 3]
        )
    else:
        addFiniteElementBlock(model, journal, "block", -OVERLAP, HEIGHT + 2 * OVERLAP, BLOCK_MATERIAL, size)
        bottom = ContactPointsOnSlaveSurface(model.elementSets[facets(model, journal, "block_bottom")], 3, "lower")
        top = ContactPointsOnSlaveSurface(model.elementSets[facets(model, journal, "block_top")], 3, "upper")
        stepActions["dirichlet"]["blockLeft"] = Dirichlet(
            "blockLeft", model.nodeSets["block_left"], "displacement", {0: 0.0}, model, journal
        )

    configuration = SurfaceToDeformableSurfacePenaltySchema(penalty=PENALTY, searchDistance=1.0)
    contacts = [
        SurfaceContact(name, model, points, model.elementSets[master], journal, configuration=configuration)
        for name, points, master in (("lower", bottom, lowerTop), ("upper", top, upperBottom))
    ]
    for contact in contacts:
        model.constraints[contact.name] = contact

    model.prepareYourself(journal)

    stepActions["dirichlet"]["held"] = Dirichlet(
        "held", model.nodeSets["lower_bottom"], "displacement", {0: 0.0, 1: 0.0}, model, journal
    )
    stepActions["dirichlet"]["pressed"] = Dirichlet(
        "pressed", model.nodeSets["upper_top"], "displacement", {0: 0.0, 1: -COMPRESSION}, model, journal
    )
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
    return model, contacts


def reactionOfThePress(model: MPMModel) -> float:
    """The vertical force with which the press pushes the top of the upper platen down, positive."""

    displacement = model.nodeFields["displacement"]
    return -displacement["P"][[displacement.indexOfNode(node) for node in model.nodeSets["upper_top"]]][:, 1].sum()


def pressuresAndPositions(model: MPMModel, contacts: list) -> tuple[np.ndarray, np.ndarray]:
    """The pressure and the x coordinate of every contact point of both interfaces."""

    return (
        np.concatenate([contact.getNormalPressures() for contact in contacts]),
        np.concatenate([contact.slave.currentPointCoordinates(model)[:, 0] for contact in contacts]),
    )


def checkEquilibrium(model: MPMModel, contacts: list) -> float:
    """Each interface transmits the force of the press, and the forces on its two sides cancel; returns the force."""

    reaction = reactionOfThePress(model)
    assert reaction > 0.0
    for contact in contacts:
        assert -contact.totalNormalForce == pytest.approx(reaction, rel=1e-8)
        np.testing.assert_allclose(
            contact.contactForceOnSlave + contact.contactForceOnMaster, 0.0, atol=1e-9 * reaction
        )
    return reaction


def test_finite_elements_transmit_a_uniform_pressure():
    """Matched faces of finite elements: the pressure is uniform at every contact point, up to the slight lateral
    stress of the neo-Hookean solid at finite strain, which the block's held left side takes."""

    model, contacts = run("finite elements")
    reaction = checkEquilibrium(model, contacts)
    pressures, _ = pressuresAndPositions(model, contacts)
    np.testing.assert_allclose(pressures, reaction / WIDTH, rtol=1e-5)


def connectedBlockOfParticles(size: float = SIZE) -> MPMModel:
    """A model of the block of particles alone, its particles connected to their kernel functions."""

    journal = Journal(verbose=False)
    model = MPMModel(2)
    addParticleBlock(model, journal, 0.0, size)
    model.prepareYourself(journal)
    model.updateConnectivity()
    return model


def centersOfTheKernelFunctions(model: MPMModel) -> dict:
    """Where the kernel function of each node is: bonded to its particle, at the particle's center."""

    return {
        kernelFunction.node: kernelFunction.center
        for particle in model.particles.values()
        for kernelFunction in particle.kernelFunctions
    }


def ownPressureLoads(model: MPMModel, sides: tuple[str, ...]) -> dict:
    """The forces of a unit pressure on the given sides of the block of particles (``bottom``, ``top``, ``left``,
    ``right``) on the nodes of their kernel functions, as the particles distribute their own pressure on a face."""

    forces = {}
    for side in sides:
        for faceID, particles in model.surfaces[f"block_{side}"].items():
            for particle in particles:
                P = np.zeros(particle.nDof)
                particle.computeDistributedLoad("PRESSURE", faceID, np.ones(1), P, np.zeros(particle.nDof**2), 0.0, 0.0)
                for node, force in zip(particle.nodes, -P.reshape(-1, 2)):  # the residual contribution, negated
                    forces[node] = forces.get(node, 0.0) + force
    return forces


def test_the_particles_balance_a_homogeneous_stress_with_their_own_pressure_at_every_node():
    """The premise of the mechanism (see the module): the nodes of the block of particles displaced by a homogeneous
    stretch, ``stretch`` times where their kernel functions are, the particles' internal forces are, at every node, the
    outermost columns at the free ends included, the loads of their own pressure on all four sides of the block times
    one factor, the stress (an isotropic stretch gives an isotropic stress in this isotropic solid)."""

    stretch = 1e-3
    model = connectedBlockOfParticles()
    centers = centersOfTheKernelFunctions(model)
    internal = {}
    for particle in model.particles.values():
        P, K = np.zeros(particle.nDof), np.zeros(particle.nDof**2)
        dU = np.ascontiguousarray(stretch * np.array([centers[node] for node in particle.nodes]).ravel())
        particle.computeKernels(K, P, np.zeros(particle.nDof), dU, 1.0, 1.0)
        for node, force in zip(particle.nodes, P.reshape(-1, 2)):
            internal[node] = internal.get(node, 0.0) + force
    own = ownPressureLoads(model, ("bottom", "top", "left", "right"))

    nodes = list(internal)
    internalForces = np.array([internal[node] for node in nodes])
    ownLoads = np.array([own.get(node, np.zeros(2)) for node in nodes])
    stress = np.sum(internalForces * ownLoads) / np.sum(ownLoads**2)
    assert stress != 0.0
    np.testing.assert_allclose(internalForces, stress * ownLoads, rtol=0.0, atol=1e-12 * np.abs(internalForces).max())


def loadsOfAUnitPressureOnTheBottom(
    size: float = SIZE, nQuadraturePoints: int = 2
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The forces of a unit pressure on the bottom faces of the block of particles, on the nodes of the kernel functions
    of these particles: as the particles distribute their own pressure on a face, and as the contact points do, with
    ``nQuadraturePoints`` per face.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Where the kernel functions of the nodes are, the forces of the particles' own pressure, and the forces of the
        contact points.
    """

    model = connectedBlockOfParticles(size)
    faces = [
        (particle, faceID) for faceID, particles in model.surfaces["block_bottom"].items() for particle in particles
    ]
    own = ownPressureLoads(model, ("bottom",))

    points = ContactPointsOnParticleFaces(faces, nQuadraturePoints=nQuadraturePoints)
    nodes = points.locatePoints(np.arange(points.nPoints))
    forcesOnPoints = points.areas[:, None] * np.array([0.0, 1.0])
    centers = centersOfTheKernelFunctions(model)
    return (
        np.array([centers[node] for node in nodes]),
        np.array([own[node] for node in nodes]),
        points.interpolationMatrixOfLocatedPoints().T @ forcesOnPoints,
    )


def test_the_contact_loads_the_particles_as_their_own_pressure_does_except_at_the_free_ends():
    """The forces of a unit pressure on the bottom of the block of particles on the nodes of their kernel functions:
    the particles' own (:math:`N` at the face centers), the contact points' (:math:`G` at their points). Both push the
    block up by the width; their difference has no resultant and no moment. It is the same in every column of kernel
    nodes but the two outermost at each free end, and sums to zero in each of these middle columns. The contact points
    load alike at one point per face: :math:`G` is affine."""

    coordinates, own, byContact = loadsOfAUnitPressureOnTheBottom()
    _, _, byOnePointPerFace = loadsOfAUnitPressureOnTheBottom(nQuadraturePoints=1)
    np.testing.assert_allclose(byOnePointPerFace, byContact, rtol=0.0, atol=1e-12 * WIDTH)
    np.testing.assert_allclose(own.sum(axis=0), [0.0, WIDTH], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(byContact.sum(axis=0), [0.0, WIDTH], rtol=1e-12, atol=1e-12)

    difference = byContact - own
    scale = np.abs(difference).max()
    assert scale > 1e-2, "the contact points do not distribute as the particles do"
    np.testing.assert_allclose(difference.sum(axis=0), 0.0, atol=1e-12 * WIDTH)
    moment = np.sum(coordinates[:, 0] * difference[:, 1] - coordinates[:, 1] * difference[:, 0])
    assert abs(moment) < 1e-12 * WIDTH**2

    columns = np.unique(coordinates[:, 0].round(9))
    differenceInColumn = [
        difference[inColumn][np.argsort(coordinates[inColumn, 1])]
        for inColumn in (np.isclose(coordinates[:, 0], x) for x in columns)
    ]
    inTheMiddle, atTheEnds = differenceInColumn[2:-2], differenceInColumn[:2] + differenceInColumn[-2:]
    for column in inTheMiddle:
        np.testing.assert_allclose(column, inTheMiddle[0], atol=1e-12 * scale)
        np.testing.assert_allclose(column.sum(axis=0), 0.0, atol=1e-12 * scale)
    for column in atTheEnds:
        assert np.abs(column - inTheMiddle[0]).max() > 1e-2 * scale


def test_the_particles_deviate_from_the_uniform_pressure_only_near_their_free_ends():
    """A block of particles transmits the force of the block of finite elements, at two particle sizes. Beyond two and a
    half particles from a free end the pressure is uniform within a percent. Within, it deviates by a few percent: near
    the right end, which nothing holds, as near the left one, which the support holds in x, by profiles that mirror each
    other within a percent of the pressure; at the faces at the right end, the pressures of the two points of a face
    differ more than their mean deviates, the springs holding the tilt of the face. At half the size the profile against
    the distance from the right end in particles is the same, within a percent of the pressure: the band narrows, the
    deviation stays. The force converges to that of the elements as the band narrows."""

    profiles, forceErrors = {}, {}
    for size in (SIZE, SIZE / 2):
        reference = reactionOfThePress(run("finite elements", size)[0])
        model, contacts = run("particles", size)
        reaction = checkEquilibrium(model, contacts)
        forceErrors[size] = abs(reaction / reference - 1.0)

        pressures, x = pressuresAndPositions(model, contacts)
        deviation = pressures / (reference / WIDTH) - 1.0
        nearTheLeftEnd, nearTheRightEnd = x / size < 2.5, (WIDTH - x) / size < 2.5
        assert np.abs(deviation[~(nearTheLeftEnd | nearTheRightEnd)]).max() < 1e-2
        assert np.abs(deviation[nearTheRightEnd]).max() > 3e-2

        interface = np.repeat(np.arange(len(contacts)), [contact.slave.nPoints for contact in contacts])
        for i in range(len(contacts)):
            fromTheLeft = deviation[interface == i][np.argsort(x[interface == i])]
            np.testing.assert_allclose(fromTheLeft, fromTheLeft[::-1], atol=1e-2)

        # the two points of each face (the points are numbered face after face)
        ofTheFaces = deviation.reshape(-1, 2)
        facesAtTheRightEnd = ofTheFaces[nearTheRightEnd.reshape(-1, 2).all(axis=1)]
        assert np.ptp(facesAtTheRightEnd, axis=1).max() > np.abs(facesAtTheRightEnd.mean(axis=1)).max()

        order = np.lexsort((WIDTH - x[nearTheRightEnd], interface[nearTheRightEnd]))
        profiles[size] = deviation[nearTheRightEnd][order]

    np.testing.assert_allclose(profiles[SIZE / 2], profiles[SIZE], atol=1e-2)
    assert forceErrors[SIZE] < 1e-3
    assert forceErrors[SIZE / 2] < 0.5 * forceErrors[SIZE]


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    """No matter where pytest is ran, we set the working dir to this testscript's parent directory."""

    monkeypatch.chdir(request.fspath.dirname)


if __name__ == "__main__":
    for block in ("finite elements", "particles"):
        model, contacts = run(block)
        pressures, _ = pressuresAndPositions(model, contacts)
        print(
            f"{block}: force of the press {reactionOfThePress(model):.6e}, pressure {pressures.mean():.6e} "
            f"({pressures.min():.6e} to {pressures.max():.6e})"
        )
