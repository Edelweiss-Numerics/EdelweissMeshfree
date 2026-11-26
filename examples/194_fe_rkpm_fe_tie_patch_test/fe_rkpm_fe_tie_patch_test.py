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
"""The patch test of the tie between particles and finite elements: a column of a block of particles between two
blocks of finite elements, tied to both, is stretched or sheared homogeneously, and the ties must transmit the
uniform traction of that state.

.. code-block:: text

        |  |  |  |  |  |  |  |    the top of the upper block is moved: up (tension) or sideways (shear)
        +----------------------+
        |  upper block (FE)    |
        +======================+  tie
        |                      |
        |  block of particles  |
        |  (or of finite       |
        |   elements)          |
        +======================+  tie
        |  lower block (FE)    |
        +----------------------+
        ///////////////////////   the bottom of the lower block is held

All three blocks are elastic (compressible neo-Hookean, Poisson's ratio zero), in 3D: ``C3D20RUL`` elements of 10 mm,
displacement particles (``DisplacementSQCNIxNSNI/3D/Hexa``) of 5 mm, so the faces of four particles lie on the face of
one element. The ties are EdelweissFE's one contact constraint with the law ``type=tied``
(:mod:`edelweissfe.constraints.surfacetodeformablesurfacepenalty`): the quadrature points of the faces of the particles
(:class:`~edelweissmeshfree.constraints.base.contactpointsonparticlefaces.ContactPointsOnParticleFaces`) are tied to the
faces of the elements, in every direction. The middle block may start with a gap or an overlap to both outer blocks,
which the ties keep as their state at rest.

The sides of the column hold the components of the homogeneous state that are zero on them: in tension, the normal
component (uniaxial strain, which with Poisson's ratio zero is uniaxial stress); in shear, the vertical and the
out-of-plane component, so that the column shears uniformly, ``u_x = gamma z``, the sides carrying the complementary
shear. The exact solution is homogeneous in every block, with the traction ``t = E delta / (H + 2 E / k)`` (tension,
modulus ``E``) or ``G delta / (H + 2 G / k)`` (shear, modulus ``G``) at every tied point: the height of the column
``H``, and the two interfaces, each as compliant as its springs ``k``. The sides of the elements are held by Dirichlet
boundary conditions; those of the particles by springs at the four vertices of each particle on a side, each standing
for a quarter of its face there (:func:`sideStiffness`).

A middle block of finite elements passes the test: every tied point transmits that traction. In tension to round-off:
uniaxial strain is homogeneous in the finite-strain solid as well. In shear to first order in the shear ``gamma``: the
finite-strain solid, sheared between sides held in the vertical, is homogeneous only to that order, so the traction
varies from point to point by a fraction of ``gamma``.

A middle block of particles transmits the force of the column, to a few thousandths, and the error halves with the
size of the particles; but not the same traction at every point. The mechanism is the one of the contact patch test
(example 192, and the module of the contact points): the particles (``SQCNI``) balance a homogeneous stress exactly
only with the load of a traction distributed with their shape functions at the center of each face, :math:`N`, as
their own pressure on a face is. The tie distributes it differently, in two ways:

- with the particles' interpolation linearized about their centers, :math:`G`, as every contact point of a particle
  does. On each face that differs from :math:`N` at the center by a load without resultant and moment, which repeats
  from one column of kernel nodes to the next inside a row of particles, without resultant in each column, but not at
  the sides of the column: the particles deviate slightly from the homogeneous state within two or three particles of
  the sides, which at this size of the particles is the whole width of the column;
- at 2 x 2 points per face, which the quadratic faces of the elements need (one point per face of a particle would
  load them inconsistently, only at their midside nodes, and leave the force short). Four points on a face also hold
  its tilt.

So the stiff springs of the tie hold the faces of the particles against the small tilts of that deviation, and the
traction varies across each face, the more the stiffer the tie, whose springs then hold the tilts rather than follow
them. The traction of a tie at a single point of particles is therefore not a measure
of the stress there; its resultant over a few particles is. The deviation narrows with the size of the particles, so
the traction inside the column becomes more uniform.

In shear the sides load the particles as well, through the springs at their vertices, and add two smaller effects to
the tie's. The springs give way by the complementary shear over their stiffness per unit area ``k_s``, which lowers the
force of the column by about ``G / (k_s W)``, ``W`` the width of the column; ``k_s`` grows as ``1 / h``, so that this
halves with the size of the particles too. And they distribute the load with :math:`N` at the vertices (the default of
the support): equal forces at the four vertices of a face are then the trapezoidal rule of :math:`N` over the face,
which agrees with the particles' own load of the face, :math:`N` at its center, to second order, so the band they
leave at the sides is smaller than the tie's. With :math:`G`, the four springs of a face would hold its tilt, as the
four points of the tie do. Their stiffness, ``k N G^T``, is unsymmetric, which the implicit step takes.

The step is quasi-static, solved by EdelweissFE's ``NISTParallel``: the implicit path of the tie, with its tangent.
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
from edelweissfe.generators.boxgen import BoxgenSchema
from edelweissfe.generators.boxgen import Generator as BoxGenerator
from edelweissfe.generators.surfaceelementgenerator import buildContactFacets
from edelweissfe.journal.journal import Journal
from edelweissfe.sections.solid import Section as SolidSection
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

#: The column: WIDTH x WIDTH in plan, the outer blocks OUTER high, the middle block MIDDLE high; elements of ELEMENT,
#: particles of PARTICLE.
WIDTH, OUTER, MIDDLE, ELEMENT, PARTICLE = 20.0, 10.0, 20.0, 10.0, 5.0
#: Young's modulus, the shear modulus (Poisson's ratio zero), and the stiffness of the ties per unit area.
E = 30000.0
G = E / 2
PENALTY = 1e6
#: The strain, or the shear, of the column: small enough for the neo-Hookean solid to respond linearly.
STRAIN = 1e-5
#: The displacement of the top.
DELTA = STRAIN * (2 * OUTER + MIDDLE)
#: The vertices of a hexahedral particle on each of its sides.
VERTICES_ON_SIDE = {"xMin": [0, 3, 4, 7], "xMax": [1, 2, 5, 6], "yMin": [0, 1, 4, 5], "yMax": [2, 3, 6, 7]}

MATERIAL = {"name": "CompressibleNeoHooke", "properties": np.array([E / 3, G, 1.0])}


def addFiniteElementBlock(model: MPMModel, journal: Journal, name: str, z0: float, height: float):
    """A block of ``C3D20RUL`` elements, ``WIDTH`` x ``WIDTH`` x ``height``, from ``z0`` up, with its section."""

    with model.topology.changes():
        BoxGenerator(
            name,
            model,
            journal,
            configuration=BoxgenSchema(
                x0=0.0,
                y0=0.0,
                z0=z0,
                lX=WIDTH,
                lY=WIDTH,
                lZ=height,
                nX=round(WIDTH / ELEMENT),
                nY=round(WIDTH / ELEMENT),
                nZ=max(1, round(height / ELEMENT)),
                elType="C3D20RUL",
            ),
        )
    model.sections[name] = SolidSection(name, model, MATERIAL, [model.elementSets[f"{name}_all"]])


def facets(model: MPMModel, journal: Journal, surface: str) -> str:
    """The contact facets of a surface of finite elements, ``<surface>_facets``."""

    with model.topology.changes():
        return buildContactFacets(model, surface, surface, "midside", "facetConsistent", journal)[0]


def addParticleBlock(model: MPMModel, journal: Journal, z0: float, particleSize: float):
    """The block of displacement particles of the given size, on kernel functions matched to them."""

    approximation = MarmotMeshfreeApproximationWrapper("ReproducingKernel", 3, completenessOrder=1)
    material = {"material": MATERIAL["name"], "properties": MATERIAL["properties"]}
    nInPlane = round(WIDTH / particleSize)
    generateBoxHexaParticleGrid(
        model,
        journal,
        lambda number, vertices, volume: MarmotParticleWrapper(
            "DisplacementSQCNIxNSNI/3D/Hexa", number, vertices, 0.0, approximation, material
        ),
        name="block",
        x0=0.0,
        y0=0.0,
        z0=z0,
        l=WIDTH,
        h=WIDTH,
        t=MIDDLE,
        nX=nInPlane,
        nY=nInPlane,
        nZ=round(MIDDLE / particleSize),
    )
    generateKernelMatchingToParticle(
        model,
        journal,
        lambda node, supportSize: MarmotMeshfreeKernelFunctionWrapper(
            node, "BSplineBoxed", supportRadius=supportSize, continuityOrder=3
        ),
        model.particleSets["block_all"],
        supportScalingFactor=2.4,
        firstKernelFunctionNumber=max(model.nodes, default=0) + 1,
    )
    domain = ParticleKernelDomain(list(model.particles.values()), list(model.meshfreeKernelFunctions.values()))
    model.particleKernelDomains["block"] = domain
    model.particleManagers.append(
        KDBinOrganizedParticleManager(domain, 3, journal, bondParticlesToKernelFunctions=True)
    )


def sideStiffness(particleSize: float) -> float:
    """The stiffness per unit area of the springs that hold the sides of the block of particles: the penalty of the ties
    at the size of the particles of the example, and stiffer in proportion as the particles get smaller."""

    return PENALTY * PARTICLE / particleSize


def holdTheSides(
    model: MPMModel, journal: Journal, load: str, middle: str, particleSize: float, stepActions: StepActionCollection
):
    """The sides of the column hold what is zero there in the homogeneous state: the normal component in tension, the
    vertical and the out-of-plane component in shear (and the out-of-plane component on the sides normal to y)."""

    held = {"x": [0] if load == "tension" else [1, 2], "y": [1]}
    feSides = {"x": ["left", "right"], "y": ["bottom", "top"]}
    blocks = ["lower", "upper"] + (["block"] if middle == "finite elements" else [])
    for direction, components in held.items():
        for block in blocks:
            for side in feSides[direction]:
                name = f"{block}_{side}"
                stepActions["dirichlet"][name] = Dirichlet(
                    name, model.nodeSets[name], "displacement", {c: 0.0 for c in components}, model, journal
                )
    if middle == "particles":
        particleSides = {"x": [("left", "xMin"), ("right", "xMax")], "y": [("front", "yMin"), ("back", "yMax")]}
        for direction, components in held.items():
            for setName, side in particleSides[direction]:
                name = f"block_{setName}"
                # A spring at each of the four vertices of a particle on the side, standing for a quarter of the
                # face of the particle there: the stiffness per unit area of the sides times that area.
                model.constraints[name] = ParticlePenaltyWeakDirichlet(
                    name,
                    model,
                    list(model.particleSets[name]),
                    "displacement",
                    {c: 0.0 for c in components},
                    sideStiffness(particleSize) * particleSize**2 / 4,
                    VERTICES_ON_SIDE[side],
                )


def run(
    middle: str, load: str, offset: float = 0.0, delta: float = DELTA, particleSize: float = PARTICLE
) -> tuple[MPMModel, list]:
    """Stretch or shear the column.

    Parameters
    ----------
    middle
        The middle block: "particles" or "finite elements".
    load
        "tension" or "shear".
    offset
        The gap (positive) or the overlap (negative) between the middle block and each outer block at the start.
    delta
        The displacement of the top of the column: up in tension, along x in shear.
    particleSize
        The size of the particles.

    Returns
    -------
    tuple[MPMModel, list]
        The solved model, and the two ties (lower, upper).
    """

    journal = Journal(verbose=False)
    model = MPMModel(3)
    addFiniteElementBlock(model, journal, "lower", -OUTER, OUTER)
    addFiniteElementBlock(model, journal, "upper", MIDDLE + 2 * offset, OUTER)
    # boxgen names the faces back (z min) and front (z max)
    lowerTop, upperBottom = facets(model, journal, "lower_front"), facets(model, journal, "upper_back")

    stepActions = StepActionCollection()
    if middle == "particles":
        addParticleBlock(model, journal, offset, particleSize)
        bottom = ContactPointsOnParticleFaces(model.surfaces["block_bottom"], model, nQuadraturePoints=4)
        top = ContactPointsOnParticleFaces(model.surfaces["block_top"], model, nQuadraturePoints=4)
    else:
        addFiniteElementBlock(model, journal, "block", offset, MIDDLE)
        bottom = ContactPointsOnSlaveSurface(model.elementSets[facets(model, journal, "block_back")], 3, model, "lower")
        top = ContactPointsOnSlaveSurface(model.elementSets[facets(model, journal, "block_front")], 3, model, "upper")
    holdTheSides(model, journal, load, middle, particleSize, stepActions)

    configuration = SurfaceToDeformableSurfacePenaltySchema(penalty=PENALTY, contactType="tied", searchDistance=1.0)
    ties = [
        SurfaceContact(name, model, points, model.elementSets[master], journal, configuration=configuration)
        for name, points, master in (("lower", bottom, lowerTop), ("upper", top, upperBottom))
    ]
    for tie in ties:
        model.constraints[tie.name] = tie

    model.prepareYourself(journal)

    stepActions["dirichlet"]["held"] = Dirichlet(
        "held", model.nodeSets["lower_back"], "displacement", {0: 0.0, 1: 0.0, 2: 0.0}, model, journal
    )
    top = {0: 0.0, 1: 0.0, 2: delta} if load == "tension" else {0: delta, 1: 0.0, 2: 0.0}
    stepActions["dirichlet"]["moved"] = Dirichlet(
        "moved", model.nodeSets["upper_front"], "displacement", top, model, journal
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
    return model, ties


def exactTraction(load: str) -> np.ndarray:
    """The traction on the points of the middle block at its bottom, of the homogeneous state: the force with which the
    lower block holds it, per unit area."""

    modulus, component = (E, 2) if load == "tension" else (G, 0)
    traction = np.zeros(3)
    traction[component] = -modulus * DELTA / (2 * OUTER + MIDDLE + 2 * modulus / PENALTY)
    return traction


def reactionOfTheBottom(model: MPMModel) -> np.ndarray:
    """The force with which the held bottom holds the column: the internal force ``P`` at its nodes, the reaction on
    the column there."""

    displacement = model.nodeFields["displacement"]
    return displacement["P"][[displacement.indexOfNode(node) for node in model.nodeSets["lower_back"]]].sum(axis=0)


def tractionsOfTheTies(model: MPMModel, ties: list, load: str) -> tuple[np.ndarray, np.ndarray]:
    """The component of the load of the traction on every point of the middle block (bottom tie, then top tie, the
    latter's sign turned, so that all are alike), and whether a point lies in the inner half of the column's width."""

    component = 2 if load == "tension" else 0
    tractions = np.concatenate([sign * tie.getTractions()[:, component] for tie, sign in zip(ties, (1.0, -1.0))])
    x = np.concatenate([tie.slave.currentPointCoordinates(model)[:, :2] for tie in ties])
    return tractions, np.all(np.abs(x - 0.5 * WIDTH) < 0.25 * WIDTH, axis=1)


def checkEquilibrium(model: MPMModel, ties: list, load: str):
    """The forces on the two sides of each tie cancel, and the bottom holds the column with the resultant of the lower
    tie, which the upper one transmits as well."""

    scale = WIDTH**2 * np.abs(exactTraction(load)).max()
    for tie in ties:
        np.testing.assert_allclose(tie.contactForceOnSlave + tie.contactForceOnMaster, 0.0, atol=1e-9 * scale)
    # along the load: the sides of the column carry none of it
    component = 2 if load == "tension" else 0
    assert reactionOfTheBottom(model)[component] == pytest.approx(
        ties[0].contactForceOnSlave[component], abs=1e-6 * scale
    )
    assert ties[0].contactForceOnSlave[component] == pytest.approx(
        -ties[1].contactForceOnSlave[component], abs=1e-6 * scale
    )


@pytest.mark.parametrize("load", ["tension", "shear"])
def test_finite_elements_tied_to_finite_elements_transmit_the_homogeneous_state(load):
    """The patch test of the tie: every point transmits the same traction, that of the column; in tension to round-off,
    in shear to first order in the shear (see the module)."""

    model, ties = run("finite elements", load)
    checkEquilibrium(model, ties, load)
    tractions, _ = tractionsOfTheTies(model, ties, load)
    exact = exactTraction(load).sum()
    np.testing.assert_allclose(tractions, tractions.mean(), rtol=1e-10 if load == "tension" else STRAIN)
    # the neo-Hookean solid responds linearly to about the strain
    assert tractions.mean() == pytest.approx(exact, rel=1e-4)


@pytest.mark.parametrize("load", ["tension", "shear"])
def test_particles_tied_to_finite_elements_transmit_the_homogeneous_state(load):
    """Particles tied to finite elements transmit the force of the column of finite elements, to a few thousandths, and
    the error halves with the size of the particles: the band of the particles' deviation from the homogeneous state at
    the sides of the column is a few particles wide, so its share of the column falls with their size (and, in shear,
    so does the compliance of the sides, see the module). The traction from point to point is not uniform: the stiff
    springs hold the faces of the particles against the tilts of that deviation. The deviation narrows with the size of
    the particles, and the traction inside the column becomes more uniform."""

    model, ties = run("finite elements", load)
    reference, _ = tractionsOfTheTies(model, ties, load)
    model, ties = run("particles", load)
    checkEquilibrium(model, ties, load)
    tractions, inner = tractionsOfTheTies(model, ties, load)
    errorOfTheForce = abs(tractions.mean() / reference.mean() - 1.0)
    assert errorOfTheForce < 5e-3

    spread = np.ptp(tractions[inner]) / abs(tractions.mean())
    model, ties = run("particles", load, particleSize=PARTICLE / 2)
    tractions, inner = tractionsOfTheTies(model, ties, load)
    assert abs(tractions.mean() / reference.mean() - 1.0) < 0.6 * errorOfTheForce
    assert np.ptp(tractions[inner]) / abs(tractions.mean()) < 0.5 * spread


@pytest.mark.parametrize("offset", [0.1, -0.1])
def test_a_gap_or_an_overlap_at_the_start_is_the_state_at_rest(offset):
    """With a gap or an overlap of 0.1 mm between the particles and both blocks of elements, the column is at rest when
    the top is not moved, and stretched it transmits the tractions of the column without one, to round-off."""

    model, ties = run("particles", "tension", offset, delta=0.0)
    for tie in ties:
        np.testing.assert_allclose(tie.getTractions(), 0.0, atol=1e-9 * np.abs(exactTraction("tension")).max())
    reference, _ = tractionsOfTheTies(*run("particles", "tension"), "tension")
    tractions, _ = tractionsOfTheTies(*run("particles", "tension", offset), "tension")
    np.testing.assert_allclose(tractions, reference, rtol=1e-9)


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    """No matter where pytest is run, the working directory is the parent directory of this test script."""

    monkeypatch.chdir(request.fspath.dirname)


if __name__ == "__main__":
    for middle in ("finite elements", "particles"):
        for load in ("tension", "shear"):
            tractions, inner = tractionsOfTheTies(*run(middle, load), load)
            mean = tractions.mean()
            print(
                f"{middle}, {load}: mean traction {mean:.6e} (linearized exact {exactTraction(load).sum():.6e}), "
                f"spread {np.ptp(tractions) / abs(mean):.3e}, inner half {np.ptp(tractions[inner]) / abs(mean):.3e}"
            )
