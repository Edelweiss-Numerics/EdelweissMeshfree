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
"""The model of the tutorial "FE projectile on an RKPM slab", small enough to run as a test, and its checks.

A steel projectile of finite elements (``C3D20RUL``, ``VONMISES/HUGHES-WINGET``) hits, at 300 m/s, a concrete slab of
RKPM particles (``GradientEnhanced6FiniteStrainSQCNIxNSNI/3D/Hexa``, ``GM7/COROTATIONAL``) held at its edges; the two
meet through the two-way penalty contact of
:class:`~edelweissmeshfree.constraints.explicit.particlepenaltyfacetcontactexplicit.ParticlePenaltyFacetContactExplicit`,
and one explicit analysis, EdelweissFE's ``NEDParallel``, integrates both. A quarter model, with the symmetry planes
x = 0 and y = 0. The tutorial includes its code from this file; the production run of the tutorial is the same model,
larger and longer.

The tests assert the checks of the tutorial's "What to check", on the first microseconds of the impact: the kinetic
energy never exceeds the projectile's initial one, action and reaction cancel in every increment, no contact point is
found inside, the deepest overlap stays far below the search distance, the projectile slows down, and halving the time
increment changes the nose's motion by little.

Units: N, mm, t, s.
"""

import numpy as np
import pytest
from edelweissfe.config.configurator import loadConfiguration
from edelweissfe.generators.boxgen import BoxgenSchema
from edelweissfe.generators.boxgen import Generator as BoxGenerator
from edelweissfe.generators.surfaceelementgenerator import buildContactFacets
from edelweissfe.journal.journal import Journal
from edelweissfe.outputmanagers.base.outputmanagerbase import OutputManagerBase
from edelweissfe.sections.solid import Section as SolidSection
from edelweissfe.solvers.nonlinearexplicitdynamicparallel import NEDParallel
from edelweissfe.stepactions.dirichlet import StepAction as Dirichlet
from edelweissfe.steps.adaptivestepforexplicitsimulations import (
    AdaptiveStepForExplicitSimulations,
)
from edelweissfe.steps.stepmanager import StepActionCollection

from edelweissmeshfree.constraints.explicit.particlepenaltyfacetcontactexplicit import (
    ParticlePenaltyFacetContactExplicit,
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

#: The projectile: a steel cube (the quarter of it: 25 x 25 x 50 mm), its nose STANDOFF above the slab, at 300 m/s.
PROJECTILE_EDGE, STANDOFF, IMPACT_VELOCITY = 50.0, 0.5, -300e3
#: VONMISES/HUGHES-WINGET: E, nu, yield stress, linear hardening, delta yield stress, delta, density.
STEEL = [210000.0, 0.3, 1200.0, 1000.0, 0.0, 1.0, 7.85e-9]
#: The slab: its thickness, the half of its span (the quarter model), and the particles through its thickness.
SLAB_THICKNESS, HALF_SPAN, LAYERS = 100.0, 100.0, 5
#: GM7 at finite strains: E, nu, k1, k2, k3, k4, k5, c [mm^2], m, c3, fc, transitionWidth, transitionExponent,
#: stressUnit, density; then the nonlocal viscosity eta and the micro-inertia eta^2 / 4 (critically damped).
NONLOCAL_VISCOSITY = 2e-6
GM7 = [30600.0, 0.2, 0.903e-4, 110.0, 18.0, 100.0, 1e-4, 16.0, 1.05, 9.9, 32.3, 4.0, 0.4, 1.0, 2.4e-9]
GM7 += [NONLOCAL_VISCOSITY, NONLOCAL_VISCOSITY**2 / 4]
#: The penalties: of the supports and symmetry planes, and of the contact, per point; and the contact search distance.
SUPPORT_PENALTY, CONTACT_PENALTY, SEARCH_DISTANCE = 1e6, 2e5, 10.0
#: The vertices of a hexahedral particle on its faces x min, x max, y min, y max.
VERTICES_ON_FACE = {"xMin": [0, 3, 4, 7], "xMax": [1, 2, 5, 6], "yMin": [0, 1, 4, 5], "yMax": [2, 3, 6, 7]}


def buildProjectile(model: MPMModel, journal: Journal) -> list:
    """The quarter of the cube, of C3D20RUL elements, with its steel section; returns the contact facets of its faces
    that can touch the slab (the nose and the two outer faces)."""

    with model.topology.changes():
        BoxGenerator(
            "projectile",
            model,
            journal,
            configuration=BoxgenSchema(
                x0=0.0,
                y0=0.0,
                z0=STANDOFF,
                lX=PROJECTILE_EDGE / 2,
                lY=PROJECTILE_EDGE / 2,
                lZ=PROJECTILE_EDGE,
                nX=1,
                nY=1,
                nZ=2,
                elType="C3D20RUL",
            ),
        )
    model.sections["projectile"] = SolidSection(
        "projectile",
        model,
        {"name": "VONMISES/HUGHES-WINGET", "properties": np.array(STEEL)},
        [model.elementSets["projectile_all"]],
    )

    # boxgen names the faces left/right (x), bottom/top (y), back/front (z): the nose is "back"
    facets = []
    with model.topology.changes():
        for face in ["back", "right", "top"]:
            facetSet, _ = buildContactFacets(
                model, f"projectile_{face}", f"projectile_{face}_contact", "midside", "facetConsistent", journal
            )
            facets += list(model.elementSets[facetSet])
    return facets


def buildSlab(model: MPMModel, journal: Journal):
    """The quarter of the slab, of GM7 particles with six nonlocal variables, on kernel functions matched to the
    particles, and the manager that assigns the particles to the kernel functions they reach."""

    particleSize = SLAB_THICKNESS / LAYERS
    nInPlane = int(round(HALF_SPAN / particleSize))
    approximation = MarmotMeshfreeApproximationWrapper("ReproducingKernel", 3, completenessOrder=1)
    material = {"material": "GM7/COROTATIONAL", "properties": np.array(GM7)}

    def particleFactory(number, vertexCoordinates, volume):
        return MarmotParticleWrapper(
            "GradientEnhanced6FiniteStrainSQCNIxNSNI/3D/Hexa", number, vertexCoordinates, 0.0, approximation, material
        )

    generateBoxHexaParticleGrid(
        model,
        journal,
        particleFactory,
        name="slab",
        x0=0.0,
        y0=0.0,
        z0=-SLAB_THICKNESS,
        l=nInPlane * particleSize,
        h=nInPlane * particleSize,
        t=SLAB_THICKNESS,
        nX=nInPlane,
        nY=nInPlane,
        nZ=LAYERS,
    )

    def kernelFunctionFactory(node, supportSize):
        return MarmotMeshfreeKernelFunctionWrapper(node, "BSplineBoxed", supportRadius=supportSize, continuityOrder=3)

    # the nodes of the kernel functions are numbered after the projectile's
    generateKernelMatchingToParticle(
        model,
        journal,
        kernelFunctionFactory,
        model.particleSets["slab_all"],
        supportScalingFactor=2.4,
        firstKernelFunctionNumber=max(model.nodes) + 1,
    )
    domain = ParticleKernelDomain(list(model.particles.values()), list(model.meshfreeKernelFunctions.values()))
    model.particleKernelDomains["slab"] = domain
    model.particleManagers.append(
        KDBinOrganizedParticleManager(domain, 3, journal, bondParticlesToKernelFunctions=True)
    )


def holdTheSlab(model: MPMModel):
    """The supported edges (z and the normal component of the vertices on the edge faces) and the symmetry planes (the
    normal component of the vertices on them). Created before a checkpoint is restored: they hold the positions of the
    model they are created with."""

    for setName, face, normal in [("slab_right", "xMax", 0), ("slab_back", "yMax", 1)]:
        model.constraints[f"support_{setName}"] = ParticlePenaltyWeakDirichlet(
            f"support_{setName}",
            model,
            list(model.particleSets[setName]),
            "displacement",
            {normal: 0.0, 2: 0.0},
            SUPPORT_PENALTY,
            constrain=VERTICES_ON_FACE[face],
            interpolationLinearizedAboutCenter=True,
        )
    for setName, face, normal in [("slab_left", "xMin", 0), ("slab_front", "yMin", 1)]:
        model.constraints[f"symmetry_{setName}"] = ParticlePenaltyWeakDirichlet(
            f"symmetry_{setName}",
            model,
            list(model.particleSets[setName]),
            "displacement",
            {normal: 0.0},
            SUPPORT_PENALTY,
            constrain=VERTICES_ON_FACE[face],
            interpolationLinearizedAboutCenter=True,
        )


def addTheContact(model: MPMModel, facets: list, journal: Journal) -> ParticlePenaltyFacetContactExplicit:
    """The contact of the vertices of the particles the projectile can reach (here, those within one particle of its
    footprint) with the facets of the projectile."""

    reach = PROJECTILE_EDGE / 2 + SLAB_THICKNESS / LAYERS
    particlesNearTheImpact = [
        particle for particle in model.particleSets["slab_all"] if np.all(particle.getCenterCoordinates()[:2] <= reach)
    ]
    contact = ParticlePenaltyFacetContactExplicit(
        "contact",
        model,
        particlesNearTheImpact,
        facets,
        CONTACT_PENALTY,
        SEARCH_DISTANCE,
        journal,
        interpolationLinearizedAboutCenter=True,
    )
    model.constraints["contact"] = contact
    return contact


def nodesOfTheProjectile(model: MPMModel) -> list:
    """The nodes of the projectile's elements, each once."""

    return list(dict.fromkeys(node for element in model.elementSets["projectile_all"] for node in element.nodes))


def startTheProjectile(model: MPMModel):
    """Give the projectile its velocity, in the entry ``V`` of the model's node field, after ``model.prepareYourself``:
    the nodes of finite elements start from the velocity the model holds."""

    displacement = model.nodeFields["displacement"]
    for node in nodesOfTheProjectile(model):
        displacement["V"][displacement.indexOfNode(node), 2] = IMPACT_VELOCITY


class ImpactRecord(OutputManagerBase):
    """Records, in every increment, what the checks need: the contact forces on both sides, the points in contact and
    found inside, the deepest overlap, the projectile's momentum along z and its nose, and the total kinetic energy."""

    checkpointedState = {}

    def __init__(self, model: MPMModel, solver: NEDParallel, contact: ParticlePenaltyFacetContactExplicit):
        self.name = "impactRecord"
        self._model, self._solver, self._contact = model, solver, contact
        displacement = model.nodeFields["displacement"]
        nodes = nodesOfTheProjectile(model)
        self._projectileNodes = [displacement.indexOfNode(node) for node in nodes]
        self._projectileZ = np.array([node.coordinates[2] for node in nodes])
        self.rows = []

    def initializeJob(self):
        pass

    def initializeStep(self, step):
        pass

    def finalizeIncrement(self, **kwargs):
        contact, displacement = self._contact, self._model.nodeFields["displacement"]
        self.rows.append(
            dict(
                time=self._model.time,
                actionPlusReaction=contact.contactForceOnParticles + contact.contactForceOnMaster,
                forceOnMaster=float(np.linalg.norm(contact.contactForceOnMaster)),
                inContact=contact.numberOfPointsInContact,
                foundInside=contact.numberOfPointsFoundInside,
                deepestPenetration=contact.deepestPenetration,
                kineticEnergy=self._solver.kineticEnergy(self._solver._V),
                velocityOfTheProjectile=float(np.mean(displacement["V"][self._projectileNodes, 2])),
                nose=float(np.min(self._projectileZ + displacement["U"][self._projectileNodes, 2])),
            )
        )

    def finalizeFailedIncrement(self, **kwargs):
        pass

    def finalizeStep(self):
        pass

    def finalizeJob(self):
        pass


def run(endTime: float, courantNumber: float = 0.8) -> tuple[ImpactRecord, float]:
    """Build the model, start the projectile, and run the impact up to ``endTime``.

    Returns
    -------
    tuple[ImpactRecord, float]
        What the run recorded, and the projectile's initial kinetic energy.
    """

    journal = Journal(verbose=False)
    model = MPMModel(3)
    facets = buildProjectile(model, journal)
    buildSlab(model, journal)
    holdTheSlab(model)
    contact = addTheContact(model, facets, journal)
    model.prepareYourself(journal)

    startTheProjectile(model)

    # the projectile's nodes on the symmetry planes
    stepActions = StepActionCollection()
    for setName, component in [("projectile_left", 0), ("projectile_bottom", 1)]:
        stepActions["dirichlet"][setName] = Dirichlet(
            setName, model.nodeSets[setName], "displacement", {component: 0.0}, model, journal
        )

    jobInfo = loadConfiguration({})
    solver = NEDParallel(jobInfo, journal, **{"courant-number": courantNumber, "output-frequency": 1})
    record = ImpactRecord(model, solver, contact)
    fieldOutputController = MPMFieldOutputController(model, journal)
    fieldOutputController.initializeJob()
    step = AdaptiveStepForExplicitSimulations(
        0,
        model,
        fieldOutputController,
        journal,
        jobInfo,
        solver,
        [record],
        stepActions,
        stepLength=endTime,
        startInc=1.0,
        maxInc=1.0,
        minInc=1e-12,
        maxNumInc=10**5,
    )
    step.solve()
    fieldOutputController.finalizeJob()

    projectileMass = PROJECTILE_EDGE**3 / 4 * STEEL[-1]
    return record, 0.5 * projectileMass * IMPACT_VELOCITY**2


#: The time the tests run: the nose meets the slab after STANDOFF / 300 m/s, about 2 us, and enters it by about 9 mm.
END_TIME = 40e-6


@pytest.fixture(autouse=True)
def _changeTestDir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


def test_the_checks_of_the_tutorial_hold_on_the_first_microseconds_of_the_impact():
    record, initialKineticEnergy = run(END_TIME)
    rows = record.rows

    # the contact acted, and slowed the projectile down by more than a fifth
    assert max(row["inContact"] for row in rows) > 0
    assert rows[-1]["velocityOfTheProjectile"] > 0.8 * IMPACT_VELOCITY
    # energy: nothing does work on the model, so the kinetic energy never exceeds the projectile's initial one
    assert max(row["kineticEnergy"] for row in rows) <= initialKineticEnergy * (1 + 1e-12)
    # momentum: action and reaction cancel, in every increment, to round-off
    largestForce = max(row["forceOnMaster"] for row in rows)
    for row in rows:
        np.testing.assert_allclose(row["actionPlusReaction"], 0.0, atol=1e-12 * largestForce)
    # search distance: no point found inside, and the deepest overlap far below the search distance
    assert all(row["foundInside"] == 0 for row in rows)
    assert max(row["deepestPenetration"] for row in rows) < 0.5 * SEARCH_DISTANCE


def test_halving_the_time_increment_changes_the_motion_of_the_nose_little():
    record, _ = run(END_TIME)
    halved, _ = run(END_TIME, courantNumber=0.4)

    def noseAt(record, time):
        times, noses = zip(*[(row["time"], row["nose"]) for row in record.rows])
        return np.interp(time, times, noses)

    penetration = -noseAt(record, END_TIME)
    assert penetration > 0.0
    assert abs(noseAt(halved, END_TIME) - noseAt(record, END_TIME)) < 0.05 * penetration
