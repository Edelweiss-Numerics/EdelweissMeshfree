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
"""A wave through the ties between particles and finite elements, in explicit dynamics: a bar of finite elements with a
segment of particles in its middle, tied to both of its parts, carries a pulse from one end to the other, and the ties
must let it through as the bar of finite elements alone does.

.. code-block:: text

      pulse ->  +--------------------+=============+--------------------+
                |  finite elements   |  particles  |  finite elements   |   free end
                +--------------------+=============+--------------------+
                     ^ probe        tie           tie       ^ probe

The bar, ``WIDTH`` x ``WIDTH`` in section and elastic (compressible neo-Hookean, Poisson's ratio zero, so that the wave
is one-dimensional), is of ``C3D8UL`` elements and of displacement particles (``DisplacementSQCNIxNSNI/3D/Hexa``), all
of the size ``h``; the ties are EdelweissFE's one contact constraint with the law ``type=tied``
(:mod:`edelweissfe.constraints.surfacetodeformablesurfacepenalty`), the quadrature points of the faces of the particles
tied to the faces of the elements. The faces are named by two generators: the elements' (boxgen) are back at z min and
front at z max, the particles' (the particle grid) are bottom at z min and top at z max. The end of the bar is moved by
``U0 (1 - cos(pi t / T)) / 2`` and then held: a pulse of the velocity ``v = U0 pi / (2 T) sin(pi t / T)``, ``c T``
long. EdelweissFE's ``NEDParallel`` integrates it.

**A tie.** A spring layer of the stiffness ``k`` per unit area between two equal bars of the impedance ``Z = rho c``
reflects a harmonic wave of the circular frequency ``omega`` with the amplitude ``R = omega Z / sqrt((omega Z)^2 + 4
k^2)``, about ``epsilon = omega Z / (2 k)``, and transmits it with ``1 / (1 - i epsilon)``: delayed by the compliance of
the springs, by the phase ``epsilon``. Two ties far enough apart that their reflections do not overlap reflect ``2 R^2``
of the energy, transmit the rest, ``1 - 2 R^2``, and change the transmitted pulse, by the delay through both, by
``|1 - 1 / (1 - i epsilon)^2|^2``, about ``4 R^2``: twice what they reflect. For the pulse, ``R^2`` is taken at its
root-mean-square circular frequency, ``pi / T`` (:func:`reflectionOfATie`). Those are the expectations of a continuum.
The discrete interface -- each spring between two nodes that carry half the mass of an element each -- reflects and
delays less, by a deficit of first order in ``omega h / c``, ``pi h / (c T)``, which halves with the size.

The stiffness of a tie is limited by the time step: a spring as stiff as the material over the size of a particle,
``E / h``, is as stiff as a particle itself, and ``NED``'s critical time step counts the springs of the ties with the
elements and particles they join (EdelweissFE's theory page "The critical time step of an explicit analysis", "Penalty
constraints"), so the time increment follows the stiffness of the tie without a setting here. ``STIFFNESS = 8`` (in
units of ``E / h``) reflects a few hundredths of a percent of the energy per tie (:func:`reflectionOfATie`), for a
time increment that falls as
``1 / sqrt(STIFFNESS)`` once the springs are stiffer than the elements. The bar of finite elements alone is run with
the time increment of the bar it is compared with: the central difference carries the pulse with a dispersion that
depends on the time increment, and so only the ties, and the segment, are to differ.

**The probes** (:class:`Geometry`). The pulse passes the first probe, in front of the ties, before the reflection of
the first tie comes back to it; the reflections of the two ties pass it one after the other, ``2 (end - start)`` apart,
more than the length of the pulse, so they do not overlap; and the run ends when the echo of the first tie's reflection
off the held end comes back to the first probe, before the echo of the transmitted pulse off the free end comes back to
the second one. So the first probe sees the incident pulse in one window and the two reflections in the next, and the
second probe the whole transmitted pulse, each with the tail the dispersion of the central difference leaves behind it.
The energy of a pulse is that of its velocity at a probe, ``Z int v^2 dt``.

**What it checks.**

- A segment of elements tied in reflects ``2 R^2``, transmits the rest -- what it reflects is what it does not
  transmit, the springs keep nothing once the pulse has passed --, and changes the transmitted pulse by twice what it
  reflects: the delay through two springs. Each within ``omega h / c`` of the continuum's.
- A segment of particles changes the pulse more than its ties: compared with the segment of elements tied in, run with
  the same time increment, it reflects more and changes the transmitted pulse more. A segment twice as long shows
  where: the reflection beyond the ties stays about the same, so it arises at the ends of the segment, where the
  particles meet the ties; the change of the transmitted pulse beyond the ties grows by more than twice, towards the
  fourfold of a delay in proportion to the length: the particles carry the wave with a phase speed of their own, and
  the transmitted pulse is delayed along the whole segment.
- Both shrink with the size of the particles and elements: at half the size, the reflection beyond the ties to below a
  quarter, faster than the ties' own reflection, which falls to a quarter with their stiffness in units of ``E / h``
  (a layer of a fixed number of particles at the ends would give the fourfold, an impedance of the particles that
  differs from the elements' to second order in their size the sixteenfold; the test does not tell these apart); the
  change of the transmitted pulse beyond the ties to below a third, as a phase speed that differs from the
  elements' to first order in the size makes it.
- The ties hand their state on to a new mesh of the elements without a jump of their forces (EdelweissFE's contact
  theory, *A tie*, *A new mesh*). ``NED`` refuses a model modifier that acts during the analysis in a model with
  particles, so it never changes the mesh in the middle of a pulse. What it hands on is the state its ties accepted:
  the one a checkpoint holds, and the one a refinement in a following step, or in a run resumed from that checkpoint,
  starts from. The test stands in for those: it stops ``NED`` a quarter of the pulse into the first tie, makes the new
  mesh of the elements by hand, and evaluates the tie at the configuration ``NED`` accepted, with the displacement of
  each tied point from rest that ``NED`` accepted last.

With a segment of particles, not all the energy of the pulse reaches a probe within the windows, and no assertion
counts on it. The probes see the mean axial velocity at two cross-sections, within the windows. They do not see what
the segment still holds when the run ends, what the ends of the segment turn into motion other than the axial one, or
what the particles do not conserve: they are updated-Lagrangian, and their shape functions are evaluated anew in every
increment, so they are not the fixed system whose energy the central difference keeps (to a modified energy, for a
linear one). This example does not tell these apart.
"""

from dataclasses import dataclass

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
from edelweissfe.models.modelchange import ModelChange
from edelweissfe.models.modelchangeobserver import ModelChangeType
from edelweissfe.outputmanagers.base.outputmanagerbase import OutputManagerBase
from edelweissfe.sections.solid import Section as SolidSection
from edelweissfe.solvers.nonlinearexplicitdynamicparallel import NEDParallel
from edelweissfe.stepactions.dirichlet import StepAction as Dirichlet
from edelweissfe.steps.adaptivestepforexplicitsimulations import (
    AdaptiveStepForExplicitSimulations,
)
from edelweissfe.steps.stepmanager import StepActionCollection
from edelweissfe.utils.meshtools import currentNodeDisplacements

from edelweissmeshfree.constraints.base.contactpointsonparticlefaces import (
    ContactPointsOnParticleFaces,
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

#: The section of the bar, WIDTH x WIDTH, and the size of the elements and particles [mm].
WIDTH, SIZE = 10.0, 5.0
#: Young's modulus [MPa], density [t/mm^3], and the speed of the wave [mm/s].
E, RHO = 30000.0, 2.4e-9
C = np.sqrt(E / RHO)
#: The pulse: the end moves by U0 [mm] over the time T [s], so the pulse is C T = 50 mm long.
U0, T = 1e-3, 50.0 / C
#: The stiffness of the ties per unit area, in units of E / SIZE.
STIFFNESS = 8.0


@dataclass(frozen=True)
class Geometry:
    """Where the segment and the probes are along the bar, from its driven end, and how long the bar is [mm]; and from
    these, when the pulses pass the probes (see the module documentation, *The probes*). The pulse is ``C T`` long.

    Parameters
    ----------
    start
        Where the segment starts: the first tie.
    end
        Where it ends: the second tie.
    length
        The length of the bar.
    probes
        The two probes, in front of the first tie and behind the second one.
    """

    start: float
    end: float
    length: float
    probes: tuple[float, float]

    def __post_init__(self):
        pulse, (first, second) = C * T, self.probes
        if not first + pulse <= 2 * self.start - first:
            raise ValueError("the first tie's reflection would reach the first probe before the incident pulse passed")
        if not 2 * (self.end - self.start) >= pulse:
            raise ValueError("the reflections of the two ties would overlap")
        if not 2 * self.end - first + pulse < C * self.endTime:
            raise ValueError("the second tie's reflection would not pass the first probe before the run ends")
        if not (second + pulse < C * self.endTime and 2 * self.length - second >= C * self.endTime):
            raise ValueError("the transmitted pulse would not pass the second probe before its echo comes back")

    @property
    def firstReflectionArrives(self) -> float:
        """When the reflection of the first tie reaches the first probe: the incident pulse has passed it."""

        return (2 * self.start - self.probes[0]) / C

    @property
    def endTime(self) -> float:
        """When the run ends: the echo of the first tie's reflection off the held end reaches the first probe."""

        return (2 * self.start + self.probes[0]) / C


#: The bar of the example: elements from 0 to 150, particles from 150 to 200, elements from 200 to 350; the probes at
#: 100 and 250.
BAR = Geometry(start=150.0, end=200.0, length=350.0, probes=(100.0, 250.0))
#: Two bars that differ only in the length of the segment, the one as long as in BAR, the other twice as long: where a
#: segment of particles changes the pulse, at its ends or along it.
SHORT_SEGMENT = Geometry(start=200.0, end=250.0, length=450.0, probes=(150.0, 300.0))
LONG_SEGMENT = Geometry(start=200.0, end=300.0, length=450.0, probes=(150.0, 350.0))

MATERIAL = {"name": "CompressibleNeoHooke", "properties": np.array([E / 3, E / 2, RHO])}


def addFiniteElementBar(model: MPMModel, journal: Journal, name: str, z0: float, length: float, size: float):
    """A bar of ``C3D8UL`` elements of the given size from ``z0``, ``length`` long, with its section."""

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
                lZ=length,
                nX=round(WIDTH / size),
                nY=round(WIDTH / size),
                nZ=round(length / size),
                elType="C3D8UL",
            ),
        )
    model.sections[name] = SolidSection(name, model, MATERIAL, [model.elementSets[f"{name}_all"]])


def addParticleSegment(model: MPMModel, journal: Journal, geometry: Geometry, size: float):
    """The segment of displacement particles of the given size, on kernel functions matched to them."""

    approximation = MarmotMeshfreeApproximationWrapper("ReproducingKernel", 3, completenessOrder=1)
    material = {"material": MATERIAL["name"], "properties": MATERIAL["properties"]}
    generateBoxHexaParticleGrid(
        model,
        journal,
        lambda number, vertices, volume: MarmotParticleWrapper(
            "DisplacementSQCNIxNSNI/3D/Hexa", number, vertices, 0.0, approximation, material
        ),
        name="segment",
        x0=0.0,
        y0=0.0,
        z0=geometry.start,
        l=WIDTH,
        h=WIDTH,
        t=geometry.end - geometry.start,
        nX=round(WIDTH / size),
        nY=round(WIDTH / size),
        nZ=round((geometry.end - geometry.start) / size),
    )
    generateKernelMatchingToParticle(
        model,
        journal,
        lambda node, supportSize: MarmotMeshfreeKernelFunctionWrapper(
            node, "BSplineBoxed", supportRadius=supportSize, continuityOrder=3
        ),
        model.particleSets["segment_all"],
        supportScalingFactor=2.4,
        firstKernelFunctionNumber=max(model.nodes) + 1,
    )
    domain = ParticleKernelDomain(list(model.particles.values()), list(model.meshfreeKernelFunctions.values()))
    model.particleKernelDomains["segment"] = domain
    model.particleManagers.append(
        KDBinOrganizedParticleManager(domain, 3, journal, bondParticlesToKernelFunctions=True)
    )


def facets(model: MPMModel, journal: Journal, surface: str) -> str:
    """The contact facets of a surface of finite elements, ``<surface>_facets``; built anew if they exist."""

    with model.topology.changes():
        return buildContactFacets(model, surface, surface, "midside", "facetConsistent", journal)[0]


#: The ties, and the faces of the elements the ends of the segment are tied to: the front of the left part, the back of
#: the right one.
MASTERS = {"first": "left_front", "second": "right_back"}


def tieTheSegment(model: MPMModel, journal: Journal, segment: str, size: float):
    """Tie the ends of the segment to the faces of the elements they meet: the quadrature points of the faces of its
    particles, 2 x 2 per face, or of its elements, with the stiffness ``STIFFNESS`` in units of E / ``size``."""

    masters = {name: facets(model, journal, face) for name, face in MASTERS.items()}
    if segment == "particles":
        ends = [
            ContactPointsOnParticleFaces(model.surfaces[f"segment_{end}"], model, nQuadraturePoints=4)
            for end in ("bottom", "top")
        ]
    else:
        # 3: the points of a triangle rule on each of the triangles a facet splits the parent face into
        ends = [
            ContactPointsOnSlaveSurface(model.elementSets[facets(model, journal, f"segment_{face}")], 3, model, name)
            for face, name in (("back", "first"), ("front", "second"))
        ]
    configuration = SurfaceToDeformableSurfacePenaltySchema(
        penalty=STIFFNESS * E / size, contactType="tied", searchDistance=0.1 * size
    )
    for (name, master), points in zip(masters.items(), ends):
        model.constraints[name] = SurfaceContact(
            name, model, points, model.elementSets[master], journal, configuration=configuration
        )


class VelocityAtTheProbes(OutputManagerBase):
    """Records, in every increment, the mean axial velocity of the nodes of the elements in the cross-sections of the
    probes."""

    checkpointedState = {}

    def __init__(self, model: MPMModel, probes: tuple[float, float]):
        self.name = "velocityAtTheProbes"
        self._model = model
        displacement = model.nodeFields["displacement"]
        nodes = {node for element in model.elements.values() for node in element.nodes}
        self._probes = [
            [displacement.indexOfNode(node) for node in nodes if abs(node.coordinates[2] - z) < 1e-9] for z in probes
        ]
        self.times, self.velocities = [], []

    def initializeJob(self):
        pass

    def initializeStep(self, step):
        pass

    def finalizeIncrement(self, **kwargs):
        V = self._model.nodeFields["displacement"]["V"]
        self.times.append(self._model.time)
        self.velocities.append([V[probe, 2].mean() for probe in self._probes])

    def finalizeFailedIncrement(self, **kwargs):
        pass

    def finalizeStep(self):
        pass

    def finalizeJob(self):
        pass


def run(
    bar: str,
    timeIncrement: float | None = None,
    size: float = SIZE,
    geometry: Geometry = BAR,
    endTime: float | None = None,
) -> tuple[np.ndarray, np.ndarray, MPMModel]:
    """Send the pulse through the bar.

    Parameters
    ----------
    bar
        "finite elements" (the bar of elements alone), "particles" (a segment of particles tied in its middle), or
        "tied finite elements" (a segment of elements tied in its middle).
    timeIncrement
        The time increment the step asks for; ``NED`` takes it where it is below the critical time step, the latter
        otherwise. None asks for the whole step, so that the critical time step sets the increment.
    size
        The size of the elements and particles.
    geometry
        Where the segment and the probes are, and how long the bar is.
    endTime
        When the run ends; None for when the windows of the probes end, ``geometry.endTime``.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, MPMModel]
        The times, the velocities at the two probes, and the model at the end.
    """

    endTime = geometry.endTime if endTime is None else endTime
    journal = Journal(verbose=False)
    model = MPMModel(3)
    if bar in ("particles", "tied finite elements"):
        addFiniteElementBar(model, journal, "left", 0.0, geometry.start, size)
        addFiniteElementBar(model, journal, "right", geometry.end, geometry.length - geometry.end, size)
        if bar == "particles":
            addParticleSegment(model, journal, geometry, size)
            tieTheSegment(model, journal, "particles", size)
        else:
            addFiniteElementBar(model, journal, "segment", geometry.start, geometry.end - geometry.start, size)
            tieTheSegment(model, journal, "finite elements", size)
        end = model.nodeSets["left_back"]
    else:
        addFiniteElementBar(model, journal, "bar", 0.0, geometry.length, size)
        end = model.nodeSets["bar_back"]
    model.prepareYourself(journal)

    stepActions = StepActionCollection()
    pulse = lambda progress: 0.5 * (1.0 - np.cos(np.pi * min(progress * endTime / T, 1.0)))  # noqa: E731
    stepActions["dirichlet"]["pulse"] = Dirichlet("pulse", end, "displacement", {2: U0}, model, journal, f_t=pulse)

    jobInfo = loadConfiguration({})
    solver = NEDParallel(jobInfo, journal, **{"output-frequency": 1})
    probes = VelocityAtTheProbes(model, geometry.probes)
    fieldOutputController = MPMFieldOutputController(model, journal)
    fieldOutputController.initializeJob()
    step = AdaptiveStepForExplicitSimulations(
        0,
        model,
        fieldOutputController,
        journal,
        jobInfo,
        solver,
        [probes],
        stepActions,
        stepLength=endTime,
        startInc=1.0 if timeIncrement is None else timeIncrement / endTime,
        maxInc=1.0 if timeIncrement is None else timeIncrement / endTime,
        minInc=1e-12,
        maxNumInc=10**5,
    )
    try:
        step.solve()
    finally:
        fieldOutputController.finalizeJob()
    return np.array(probes.times), np.array(probes.velocities), model


def pulsesThrough(bar: str, size: float = SIZE, geometry: Geometry = BAR) -> dict:
    """The pulses of a bar with a tied segment, against those of the bar of elements alone, run with the same time
    increment (see the module documentation, *The probes*): the energy of what the bar reflects, in the window of the
    reflections at the first probe, relative to the incident pulse there; the energy of the transmitted pulse at the
    second probe, and of its change, both relative to the transmitted pulse of the bar alone there. For a segment of
    particles, also what it reflects and changes beyond its ties: the same energies of the difference to the segment of
    elements tied in, also run with the same time increment.

    Parameters
    ----------
    bar
        The bar with a tied segment, see :func:`run`.
    size
        The size of the elements and particles of the bars.
    geometry
        Where the segment and the probes are, and how long the bar is.

    Returns
    -------
    dict
        The energies, and the largest velocity at the probes, of this bar and of the bar of elements alone.
    """

    times, velocities, _ = run(bar, size=size, geometry=geometry)
    # The time increment the ties allow: every increment but the zero increment at the start and the shorter last one.
    timeIncrement = np.diff(times).max()

    def atTheSameTimeIncrement(reference: str) -> np.ndarray:
        referenceTimes, referenceVelocities, _ = run(reference, timeIncrement, size, geometry)
        return np.column_stack([np.interp(times, referenceTimes, v) for v in referenceVelocities.T])

    def energy(v: np.ndarray, window: np.ndarray) -> float:
        return float(np.trapezoid(v[window] ** 2, times[window]))

    alone = atTheSameTimeIncrement("finite elements")
    incidentWindow = times < geometry.firstReflectionArrives
    reflectedWindow = ~incidentWindow
    wholeRun = np.full(len(times), True)
    incident, transmittedAlone = energy(alone[:, 0], incidentWindow), energy(alone[:, 1], wholeRun)
    pulses = {
        "reflected": energy(velocities[:, 0] - alone[:, 0], reflectedWindow) / incident,
        "transmitted": energy(velocities[:, 1], wholeRun) / transmittedAlone,
        "changeOfTheTransmitted": energy(velocities[:, 1] - alone[:, 1], wholeRun) / transmittedAlone,
        "largestVelocity": float(np.abs(velocities).max()),
        "largestVelocityOfTheBarAlone": float(np.abs(alone).max()),
    }
    if bar == "particles":
        tied = atTheSameTimeIncrement("tied finite elements")
        pulses["reflectedBeyondTheTies"] = energy(velocities[:, 0] - tied[:, 0], reflectedWindow) / incident
        pulses["changeOfTheTransmittedBeyondTheTies"] = (
            energy(velocities[:, 1] - tied[:, 1], wholeRun) / transmittedAlone
        )
    return pulses


def reflectionOfATie(size: float = SIZE) -> float:
    """The energy one tie reflects of the pulse, relative to the incident one: ``R^2`` of the harmonic wave of the
    circular frequency ``pi / T``, with the stiffness of the ties at the size of the elements and particles ``size``.
    That frequency is the root-mean-square one of the pulse: its half sine of the velocity has ``int v'^2 dt / int v^2
    dt = (pi / T)^2`` (Parseval). Since ``R^2`` is ``(omega Z / 2 k)^2`` to leading order, quadratic in ``omega``, it is
    the energy a tie reflects of the whole pulse, to that order."""

    omegaZ = np.pi / T * RHO * C
    k = STIFFNESS * E / size
    return omegaZ**2 / (omegaZ**2 + 4 * k**2)


def resolution(size: float = SIZE) -> float:
    """``omega h / c`` at the root-mean-square circular frequency of the pulse, ``pi / T``: how finely the elements of
    the size ``h`` resolve it, and the order of the deficit of the discrete ties against the continuum's."""

    return np.pi / T * size / C


def isStable(pulses: dict) -> bool:
    """The pulse grows nowhere beyond what the bar alone carries: the time increment is below the critical one."""

    return pulses["largestVelocity"] < 1.05 * pulses["largestVelocityOfTheBarAlone"]


def tractionsOfATieAtTheModel(tie: SurfaceContact, model: MPMModel) -> np.ndarray:
    """Evaluate a tie at the state of the model, with no increment, as an update of its connectivity finds it, and
    return the tractions at its points."""

    U = currentNodeDisplacements(tie.nodes, model, 3).ravel()
    tie.computeKernels(U, np.zeros_like(U), np.zeros_like(U), None, None)
    return tie.getTractions()


@pytest.fixture(autouse=True)
def _changeTestDir(request, monkeypatch):
    """No matter where pytest is run, the working directory is the parent directory of this test script."""

    monkeypatch.chdir(request.fspath.dirname)


def test_ties_between_finite_elements_reflect_and_delay_the_pulse_as_springs_do():
    """A segment of elements tied into the bar: the two ties reflect ``2 R^2``; what they reflect is what they do not
    transmit; and they change the transmitted pulse by twice what they reflect, the delay through both springs. The
    first is the continuum's to within ``omega h / c``, the order of the discrete deficit; the other two are balances
    that hold for the discrete ties as well, to the order ``(omega h / c)^2`` at which the energy of a discrete wave
    differs from ``Z int v^2 dt`` (see the module)."""

    pulses = pulsesThrough("tied finite elements")
    assert isStable(pulses)
    assert pulses["reflected"] == pytest.approx(2 * reflectionOfATie(), rel=resolution())
    assert 1.0 - pulses["transmitted"] == pytest.approx(pulses["reflected"], rel=resolution() ** 2)
    assert pulses["changeOfTheTransmitted"] == pytest.approx(2 * pulses["reflected"], rel=resolution() ** 2)


def test_a_segment_of_particles_changes_the_pulse_at_its_ends_and_along_its_length():
    """Beyond what its ties do, a segment of particles twice as long reflects about the same, within a factor of two
    either way -- the reflection arises at its ends --, and changes the transmitted pulse by more than twice as much:
    closer to the fourfold of a delay in proportion to its length than to the same of a change at its ends. The
    particles carry the wave with a phase speed of their own (see the module)."""

    short, long = pulsesThrough("particles", geometry=SHORT_SEGMENT), pulsesThrough("particles", geometry=LONG_SEGMENT)
    assert isStable(short) and isStable(long)
    assert short["reflectedBeyondTheTies"] / 2 < long["reflectedBeyondTheTies"] < 2 * short["reflectedBeyondTheTies"]
    assert long["changeOfTheTransmittedBeyondTheTies"] > 2 * short["changeOfTheTransmittedBeyondTheTies"]


def test_what_a_segment_of_particles_changes_beyond_its_ties_shrinks_with_the_size_of_the_particles():
    """With particles and elements of half the size, and the stiffness of the ties in units of E / size, the ties' own
    reflection falls to a quarter. What the segment of particles reflects beyond its ties falls faster, below a quarter;
    how it changes the transmitted pulse beyond its ties falls about as fast, as a phase speed that differs from the
    elements' to first order in the size makes it (see the module)."""

    coarse, fine = pulsesThrough("particles"), pulsesThrough("particles", size=SIZE / 2)
    assert isStable(coarse) and isStable(fine)
    assert reflectionOfATie(SIZE / 2) == pytest.approx(reflectionOfATie() / 4, rel=1e-3)
    assert fine["reflectedBeyondTheTies"] < coarse["reflectedBeyondTheTies"] / 4
    assert fine["changeOfTheTransmittedBeyondTheTies"] < coarse["changeOfTheTransmittedBeyondTheTies"] / 3


def test_a_new_mesh_of_the_elements_in_the_middle_of_the_pulse_hands_the_tie_on_without_a_jump():
    """``NED`` evaluates the ties of particles only after it reconnected the particles, at the configuration it
    accepted, and the ties accept the state of that evaluation: the state a checkpoint holds and a refinement in a
    following step or a resumed run starts from (``NED`` itself refuses a model modifier that acts during the analysis
    in a model with particles). The test stands in for such a refinement: it stops ``NED`` a quarter of the pulse into
    the first tie, where the stretch of the tie is already large and still changing quickly, and makes a new mesh of
    the elements the tie is tied to by hand. The tie, handed on with that state and evaluated at the same configuration,
    exerts the forces it exerted before (EdelweissFE's contact theory, *A tie*, *A new mesh*). A state accepted before
    the reconnection would be the one of the increment before, and the forces would jump by the change of the stretch
    over an increment."""

    _, _, model = run("particles", endTime=(BAR.start + 0.25 * C * T) / C)
    tie, face = model.constraints["first"], MASTERS["first"]
    # NED's last evaluation, at the configuration it accepted
    before = tie.getTractions().copy()
    assert np.abs(before).max() > 0.0

    facets(model, Journal(verbose=False), face)
    assert tie.refresh(model, ModelChange(kind=ModelChangeType.REFINEMENT, changedSurfaces={face}))
    tie.updateConnectivity(model)
    np.testing.assert_allclose(
        tractionsOfATieAtTheModel(tie, model), before, rtol=1e-9, atol=1e-9 * np.abs(before).max()
    )


if __name__ == "__main__":
    print(f"one tie reflects {reflectionOfATie():.3e}; omega h / c = {resolution():.3f}")
    for bar in ("tied finite elements", "particles"):
        print(bar, pulsesThrough(bar))
