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
"""The axisymmetric patch test of the tie between particles and finite elements, the twin of example 194: a ring of
axisymmetric particles between two rings of ``CX8RUL`` elements, tied to both, carries a homogeneous state, and the
ties must transmit its uniform traction on the full rings.

.. code-block:: text

      axial tension (ties at z = const)            radial expansion (ties at r = const)

      z    |  |  |  |  |   top moved up              z
      |  +--------------+                           |  +------+=+---------+=+------+
      |  | upper (FE)   |                           |  |inner |t|particles|t|outer |
      |  +==============+ tie                       |  | (FE) |i|         |i| (FE) |
      |  |  particles   |                           |  |      |e|         |e|      |
      |  +==============+ tie                       |  +------+=+---------+=+------+
      |  | lower (FE)   |                           |  u_r = a r on the inner and the outer face
      |  +--------------+  bottom held in z         |  the bottoms of the element rings held in z
      +---------------------- r                    +----------------------------------- r
         R0            R0 + W                          R0

The model is the section r-z of a body of revolution (``MPMModel(2, axisymmetric=True)``): the elements are
axisymmetric (``CX8RUL``), the particles too (``DisplacementSQCNI/Axisymmetric/Quad``, with the nodes of their kernel
functions at their centers), and every contact point of the ties stands for the ring it sweeps
(:class:`~edelweissmeshfree.constraints.base.contactpointsonparticlefaces.ContactPointsOnParticleFaces` and EdelweissFE's
points of the faces of elements alike measure their areas from the model when they are made). All rings are elastic
(compressible neo-Hookean, Poisson's ratio zero).

- **Axial tension**: the top of the upper ring moves up, the bottom of the lower ring is held in z. With Poisson's
  ratio zero the exact state is uniaxial, :math:`u_r = 0`, :math:`\\sigma_{zz}` uniform, with the traction
  ``t = E delta / (H + 2 E / k)`` on every tied ring (the height ``H = MIDDLE + 2 OUTER`` of the column of the three
  rings, the two interfaces as compliant as their springs ``k``), and the ties carry the force
  :math:`t\\, \\pi (r_o^2 - r_i^2)` of the annulus. The middle ring is held by the two ties alone, so their forces
  cancel.
- **Radial expansion**: the inner face of the inner ring and the outer face of the outer ring move by
  :math:`u_r = a\\, r`, the bottoms of the element rings are held in z. The state is the uniform expansion,
  :math:`\\sigma_{rr} = \\sigma_{\\theta\\theta} = 2 G a`, up to the compliance of the interfaces, and the ties at
  :math:`r = r_1, r_2` carry the traction :math:`\\sigma_{rr}` on cylinders whose areas grow with the radius: their
  forces are in the ratio :math:`r_2 / r_1`, not equal. What balances the difference is the hoop stress, the lesson
  of an axisymmetric body: the radial forces on a ring of the section, :math:`F_2 - F_1 = 2 \\pi \\int
  \\sigma_{\\theta\\theta}\\, \\mathrm{d}r\\, \\mathrm{d}z`. The springs give way by :math:`\\sigma_{rr} / k` at each
  interface, a jump of :math:`u_r` that perturbs the uniform expansion, and with it the ratio, at the relative order
  :math:`E / (k R_0)` per interface.

Without the ring weight on the contact points, neither would balance: the springs would carry the forces of a unit
thickness, the elements and the particles those of the rings. A middle ring of finite elements passes the test exactly
(axial tension: every tied point transmits the traction of the column, to round-off). A middle ring of particles
transmits the force of the middle ring of finite elements to a few thousandths, but not the same traction at every
point. The mechanism is that of example 194, not of the ring weight: the particles balance a homogeneous stress with
their shape functions at the centers of their faces, the tie loads them with their interpolation linearized about their
centers at two points per edge, and the particles deviate slightly from the homogeneous state within two or three
particles of the free ends of a tied edge -- the inner and the outer cylinder in axial tension, the top and the bottom
in radial expansion, which with four particles across is the whole ring. The stiff springs hold the edges against the
tilts of that deviation, so the traction scatters from point to point, the more the stiffer the tie, while the force
stays the same. Its resultant over a few particles is the measure of the stress, not the traction at a point. One point
per edge would hold no tilt, but it loads the quadratic edges of the elements inconsistently, only at their midside
nodes, and does not integrate the ring weight of the nodal forces, which is quadratic along an edge: hence two.

The step is quasi-static, solved by EdelweissFE's ``NISTParallel``. The axisymmetric elements ignore the thickness of
their plane section: the ring weight :math:`2 \\pi r` replaces it.
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
from edelweissfe.numerics.parallelizationutilities import (
    getNumberOfWorkerThreads,
    isFreeThreadingSupported,
)
from edelweissfe.sections.plane import PlaneSectionSchema
from edelweissfe.sections.plane import Section as PlaneSection
from edelweissfe.solvers.nonlinearimplicitstaticparallel import NISTParallel
from edelweissfe.stepactions.dirichlet import StepAction as Dirichlet
from edelweissfe.steps.adaptivestep import AdaptiveStep
from edelweissfe.steps.stepmanager import StepActionCollection

from edelweissmeshfree.constraints.base.contactpointsonparticlefaces import (
    ContactPointsOnParticleFaces,
)
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

#: The inner radius of the rings and the size of a block in the section: the middle block MIDDLE x MIDDLE, the outer
#: blocks OUTER thick; elements of ELEMENT, particles of PARTICLE.
R0, MIDDLE, OUTER, ELEMENT, PARTICLE = 20.0, 20.0, 10.0, 10.0, 5.0
#: Young's modulus, the shear modulus (Poisson's ratio zero), and the stiffness of the ties per unit area.
E = 30000.0
G = E / 2
PENALTY = 1e6
#: The strain of the state, small enough for the neo-Hookean solid to respond linearly.
STRAIN = 1e-5
MATERIAL = {"name": "CompressibleNeoHooke", "properties": np.array([E / 3, G, 1.0])}

#: The blocks of the two states, (name, (r0, z0), (width, height)): the outer blocks first, the middle one last; and
#: the faces of the middle block and of the outer blocks the two ties join, (middle face, outer block face).
LAYOUTS = {
    "axial tension": (
        [
            ("lower", (R0, -OUTER), (MIDDLE, OUTER)),
            ("upper", (R0, MIDDLE), (MIDDLE, OUTER)),
            ("block", (R0, 0.0), (MIDDLE, MIDDLE)),
        ],
        [("bottom", "lower_top"), ("top", "upper_bottom")],
    ),
    "radial expansion": (
        [
            ("inner", (R0 - OUTER, 0.0), (OUTER, MIDDLE)),
            ("outer", (R0 + MIDDLE, 0.0), (OUTER, MIDDLE)),
            ("block", (R0, 0.0), (MIDDLE, MIDDLE)),
        ],
        [("left", "inner_right"), ("right", "outer_left")],
    ),
}


def addFiniteElementBlock(model: MPMModel, journal: Journal, name: str, origin: tuple, size: tuple):
    """A ring of ``CX8RUL`` elements, the rectangle of the given origin (r, z) and size in the section, with its
    section."""

    with model.topology.changes():
        PlaneRectQuadGenerator(
            name,
            model,
            journal,
            configuration=PlaneRectQuadSchema(
                x0=origin[0],
                y0=origin[1],
                length=size[0],
                h=size[1],
                nX=round(size[0] / ELEMENT),
                nY=round(size[1] / ELEMENT),
                elType="CX8RUL",
            ),
        )
    model.sections[name] = PlaneSection(
        name, model, MATERIAL, [model.elementSets[f"{name}_all"]], configuration=PlaneSectionSchema(thickness=1.0)
    )


def facets(model: MPMModel, journal: Journal, surface: str) -> str:
    """The contact facets of a surface of finite elements, ``<surface>_facets``."""

    with model.topology.changes():
        return buildContactFacets(model, surface, surface, "midside", "facetConsistent", journal)[0]


def addParticleBlock(model: MPMModel, journal: Journal, origin: tuple, size: tuple):
    """The ring of axisymmetric displacement particles, on kernel functions at their centers."""

    approximation = MarmotMeshfreeApproximationWrapper("ReproducingKernel", 2, completenessOrder=1)
    material = {"material": MATERIAL["name"], "properties": MATERIAL["properties"]}
    generateRectangularQuadParticleGrid(
        model,
        journal,
        lambda number, vertices, volume: MarmotParticleWrapper(
            "DisplacementSQCNI/Axisymmetric/Quad", number, vertices, 0.0, approximation, material
        ),
        name="block",
        x0=origin[0],
        y0=origin[1],
        l=size[0],
        h=size[1],
        nX=round(size[0] / PARTICLE),
        nY=round(size[1] / PARTICLE),
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
        KDBinOrganizedParticleManager(domain, 2, journal, bondParticlesToKernelFunctions=True)
    )


def prescribedMotions(model: MPMModel, journal: Journal, state: str) -> StepActionCollection:
    """The supports and the prescribed motions of the state (see the module documentation); the middle ring is held by
    the ties alone."""

    stepActions = StepActionCollection()

    def prescribe(name: str, nodeSet: str, components: dict):
        stepActions["dirichlet"][name] = Dirichlet(
            name, model.nodeSets[nodeSet], "displacement", components, model, journal
        )

    if state == "axial tension":
        prescribe("held", "lower_bottom", {1: 0.0})
        prescribe("moved", "upper_top", {1: STRAIN * (MIDDLE + 2 * OUTER)})
    else:
        prescribe("inner face", "inner_left", {0: STRAIN * (R0 - OUTER)})
        prescribe("outer face", "outer_right", {0: STRAIN * (R0 + MIDDLE + OUTER)})
        for block in ("inner", "outer"):
            prescribe(f"{block} held", f"{block}_bottom", {1: 0.0})
    return stepActions


def run(state: str, middle: str) -> tuple[MPMModel, list]:
    """Load the rings into the state.

    Parameters
    ----------
    state
        "axial tension" or "radial expansion".
    middle
        The middle ring: "particles" or "finite elements".

    Returns
    -------
    tuple[MPMModel, list]
        The solved model, and the two ties.
    """

    journal = Journal(verbose=False)
    model = MPMModel(2, axisymmetric=True)
    blocks, tiedFaces = LAYOUTS[state]
    for name, origin, size in blocks[:2]:
        addFiniteElementBlock(model, journal, name, origin, size)
    masters = [facets(model, journal, outerFace) for _, outerFace in tiedFaces]

    _, origin, size = blocks[2]
    if middle == "particles":
        addParticleBlock(model, journal, origin, size)
        slaves = [
            ContactPointsOnParticleFaces(model.surfaces[f"block_{face}"], model, nQuadraturePoints=2)
            for face, _ in tiedFaces
        ]
    else:
        addFiniteElementBlock(model, journal, "block", origin, size)
        slaves = [
            ContactPointsOnSlaveSurface(
                model.elementSets[facets(model, journal, f"block_{face}")], 3, model, f"tie {face}"
            )
            for face, _ in tiedFaces
        ]

    configuration = SurfaceToDeformableSurfacePenaltySchema(penalty=PENALTY, contactType="tied", searchDistance=1.0)
    ties = [
        SurfaceContact(f"tie {face}", model, points, model.elementSets[master], journal, configuration=configuration)
        for (face, _), points, master in zip(tiedFaces, slaves, masters)
    ]
    for tie in ties:
        model.constraints[tie.name] = tie

    model.prepareYourself(journal)
    stepActions = prescribedMotions(model, journal, state)

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


#: The component of the force each tie transmits in a state: z in axial tension, r in radial expansion.
COMPONENT = {"axial tension": 1, "radial expansion": 0}


def forcesOfTheTies(ties: list, state: str) -> np.ndarray:
    """The force on the middle ring at each tie, in the direction of the load: the sum of the forces on the rings of
    its points (in radial expansion the sum of their radial forces, which cancel about the axis in every ring, but add
    up to the traction times the area of the cylinder)."""

    return np.array([tie.contactForceOnSlave[COMPONENT[state]] for tie in ties])


def checkEquilibrium(ties: list, state: str):
    """The forces on the two sides of each tie cancel, and the two ties pull the middle ring apart: in axial tension
    with the same force, to the solver's tolerance, since the middle ring is held by the ties alone; in radial
    expansion with forces in the ratio of the radii of the interfaces, as the uniform :math:`\\sigma_{rr}` acts on
    cylinders whose areas grow with the radius, the hoop stress balancing the difference, up to the compliance of the
    two interfaces, :math:`E / (k R_0)` each (see the module)."""

    forces = forcesOfTheTies(ties, state)
    for tie in ties:
        np.testing.assert_allclose(
            tie.contactForceOnSlave + tie.contactForceOnMaster, 0.0, atol=1e-9 * np.abs(forces).max()
        )
    if state == "axial tension":
        assert -forces[1] / forces[0] == pytest.approx(1.0, rel=1e-9)
    else:
        assert -forces[1] / forces[0] == pytest.approx((R0 + MIDDLE) / R0, rel=2 * E / (PENALTY * R0))


def exactAxialTraction() -> float:
    """The traction of the column in axial tension, on the middle ring at its bottom."""

    return -E * STRAIN * (MIDDLE + 2 * OUTER) / (MIDDLE + 2 * OUTER + 2 * E / PENALTY)


def test_finite_elements_tied_to_finite_elements_transmit_the_axial_state_exactly():
    """Every tied point transmits the same traction, to the solver's tolerance, and the ties carry it on the annulus.
    The traction is the column's of the linear solid to the order of the strain: the neo-Hookean solid deviates from
    linear by a relative amount of that order."""

    model, ties = run("axial tension", "finite elements")
    checkEquilibrium(ties, "axial tension")
    tractions = ties[0].getTractions()[:, 1]
    np.testing.assert_allclose(tractions, tractions.mean(), rtol=1e-10)
    assert tractions.mean() == pytest.approx(exactAxialTraction(), rel=10 * STRAIN)
    annulus = np.pi * ((R0 + MIDDLE) ** 2 - R0**2)
    assert forcesOfTheTies(ties, "axial tension")[0] == pytest.approx(exactAxialTraction() * annulus, rel=10 * STRAIN)


@pytest.mark.parametrize("state", list(LAYOUTS))
def test_particles_tied_to_finite_elements_transmit_the_state_of_the_finite_elements(state):
    """The ties of the ring of particles transmit the forces of the ring of finite elements, to a few thousandths: the
    deviation of the particles from the homogeneous state near the free ends of the tied edges (example 194), not the
    ring weight, which the points of both kinds measure alike."""

    _, reference = run(state, "finite elements")
    checkEquilibrium(reference, state)
    _, ties = run(state, "particles")
    checkEquilibrium(ties, state)
    np.testing.assert_allclose(forcesOfTheTies(ties, state), forcesOfTheTies(reference, state), rtol=5e-3)


@pytest.mark.skipif(not isFreeThreadingSupported(), reason="the thread pool runs serially with the GIL enabled")
def test_the_tie_of_particles_is_the_same_on_1_4_and_8_threads(monkeypatch):
    """The particles, the elements and the ties are evaluated in chunks on the thread pool, whose size the process
    reads from ``OMP_NUM_THREADS`` for every computation; the result does not depend on how many threads share them,
    up to the round-off of the order of the sums. Only the pool changes: the OpenMP parts, such as the assembly, keep
    the thread count the process started with, which OpenMP reads once, when it is loaded. Without free threading the
    pool runs serially whatever its size, and there is nothing to compare."""

    results = []
    for threads in ("1", "4", "8"):
        monkeypatch.setenv("OMP_NUM_THREADS", threads)
        assert getNumberOfWorkerThreads() == int(threads)
        model, ties = run("radial expansion", "particles")
        displacement = model.nodeFields["displacement"]
        results.append((np.array(displacement["U"]), forcesOfTheTies(ties, "radial expansion")))
    for U, forces in results[1:]:
        np.testing.assert_allclose(U, results[0][0], rtol=0, atol=1e-7 * np.abs(results[0][0]).max())
        np.testing.assert_allclose(forces, results[0][1], rtol=1e-7)


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    """No matter where pytest is ran, we set the working dir to this testscript's parent directory."""

    monkeypatch.chdir(request.fspath.dirname)


if __name__ == "__main__":
    for state in LAYOUTS:
        for middle in ("finite elements", "particles"):
            model, ties = run(state, middle)
            tractions = np.concatenate([tie.getTractions()[:, COMPONENT[state]] for tie in ties])
            print(
                f"{state}, {middle}: forces of the ties {forcesOfTheTies(ties, state)}, traction from "
                f"{tractions.min():.6e} to {tractions.max():.6e}"
            )
