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
"""A cantilever whose middle part is an RKPM domain, embedded between two finite element parts.

.. code-block:: text

      clamped                                                          tip deflection
        |#|-------------------+=========================+-------------------|  |
        |#|  finite elements  |  RKPM particles          |  finite elements  |  v
        |#|-------------------+=========================+-------------------|
        x = 0               x = 3                      x = 7              x = 10

The finite element parts consist of quadratic, finite strain plane strain elements (Marmot's ``CPE8UL``), the RKPM
part of displacement particles. Each interface is tied by a
:class:`~edelweissmeshfree.constraints.particlefiniteelementmortartie.ParticleFiniteElementMortarTie`. Both boundary
conditions act on finite element nodes: the left end is clamped, the right end is deflected. The step is quasi-static
and solved by EdelweissFE's ``NISTParallel``, which solves the coupled model as it solves a finite element model.

The material is a compressible neo-Hookean solid; the deflection of a tenth of the length makes the response
geometrically nonlinear.

The independent check: the same cantilever, made of finite elements only, is the reference. The coupled model must
reproduce its tip reaction and its deflection line, up to the discretization error of the two models.
"""

import argparse

import edelweissfe.utils.performancetiming as performancetiming
import h5py
import numpy as np
import pytest
from edelweissfe.config.configurator import loadConfiguration
from edelweissfe.generators.planerectquad import Generator as PlaneRectQuadGenerator
from edelweissfe.generators.planerectquad import PlaneRectQuadSchema
from edelweissfe.journal.journal import Journal
from edelweissfe.models.femodel import FEModel
from edelweissfe.sections.plane import PlaneSectionSchema
from edelweissfe.sections.plane import Section as PlaneSection
from edelweissfe.sets.nodeset import NodeSet
from edelweissfe.solvers.nonlinearimplicitstaticparallel import NISTParallel
from edelweissfe.stepactions.dirichlet import StepAction as Dirichlet
from edelweissfe.steps.adaptivestep import AdaptiveStep
from edelweissfe.steps.stepmanager import StepActionCollection
from edelweissfe.utils.fieldoutput import FieldOutputController

from edelweissmeshfree.constraints.particlefiniteelementmortartie import (
    ParticleFiniteElementMortarTie,
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

length = 10.0
height = 1.0
#: Where the RKPM domain begins and ends.
rkpmStart, rkpmEnd = 3.0, 7.0
tipDeflection = -1.0

youngsModulus = 1000.0
poissonsRatio = 0.3
bulkModulus = youngsModulus / (3 * (1 - 2 * poissonsRatio))
shearModulus = youngsModulus / (2 * (1 + poissonsRatio))
#: A compressible neo-Hookean solid; the density, which the particles require, plays no role in a quasi-static step.
materialProperties = np.array([bulkModulus, shearModulus, 1.0])

#: The size of the finite elements, and of the particles.
defaultElementSize = 0.5
defaultParticleSize = 1.0 / 8

increment = 0.2


def addFiniteElementBeam(model: FEModel, journal: Journal, name: str, x0: float, x1: float, size: float):
    """Add a beam part of quadratic plane strain elements, ``CPE8UL``, with its section.

    Parameters
    ----------
    model
        The model.
    journal
        The journal.
    name
        The name of the part, the prefix of its node and element sets.
    x0
        The left end.
    x1
        The right end.
    size
        The element size.
    """

    with model.topology.changes():
        PlaneRectQuadGenerator(
            name,
            model,
            journal,
            configuration=PlaneRectQuadSchema(
                x0=x0,
                length=x1 - x0,
                h=height,
                nX=round((x1 - x0) / size),
                nY=round(height / size),
                elType="CPE8UL",
            ),
        )

    model.sections[name] = PlaneSection(
        name,
        model,
        {"name": "CompressibleNeoHooke", "properties": materialProperties},
        [model.elementSets["{:}_all".format(name)]],
        configuration=PlaneSectionSchema(thickness=1.0),
    )


def buildCoupledModel(journal: Journal, elementSize: float, particleSize: float) -> MPMModel:
    """The cantilever: two finite element parts, the RKPM part between them, and the two ties.

    Parameters
    ----------
    journal
        The journal.
    elementSize
        The size of the finite elements.
    particleSize
        The size of the particles.

    Returns
    -------
    MPMModel
        The model, prepared for the simulation.
    """

    dimension = 2
    model = MPMModel(dimension)

    addFiniteElementBeam(model, journal, "feLeft", 0.0, rkpmStart, elementSize)
    addFiniteElementBeam(model, journal, "feRight", rkpmEnd, length, elementSize)

    nParticlesX = round((rkpmEnd - rkpmStart) / particleSize)
    nParticlesY = round(height / particleSize)

    def kernelFunctionFactory(node):
        return MarmotMeshfreeKernelFunctionWrapper(
            node, "BSplineBoxed", supportRadius=2 * particleSize, continuityOrder=3
        )

    # The kernel functions get node labels after those of the finite elements.
    generateRectangularKernelFunctionGrid(
        model,
        journal,
        kernelFunctionFactory,
        name="rkpm",
        x0=rkpmStart,
        l=rkpmEnd - rkpmStart,
        y0=0.0,
        h=height,
        nX=nParticlesX,
        nY=nParticlesY,
        firstKernelFunctionNumber=max(model.nodes) + 1,
    )

    approximation = MarmotMeshfreeApproximationWrapper(
        "ReproducingKernelImplicitGradient", dimension, completenessOrder=1
    )
    particleMaterial = {"material": "CompressibleNeoHooke", "properties": materialProperties}

    def particleFactory(number, vertexCoordinates, volume):
        # the particle computes its volume from its vertices
        return MarmotParticleWrapper(
            "DisplacementSQCNIxNSNI/PlaneStrain/Quad",
            number,
            vertexCoordinates,
            0.0,
            approximation,
            particleMaterial,
        )

    generateRectangularQuadParticleGrid(
        model,
        journal,
        particleFactory,
        name="rkpm",
        x0=rkpmStart,
        l=rkpmEnd - rkpmStart,
        y0=0.0,
        h=height,
        nX=nParticlesX,
        nY=nParticlesY,
    )

    particleKernelDomain = ParticleKernelDomain(
        list(model.particles.values()), list(model.meshfreeKernelFunctions.values())
    )
    model.particleKernelDomains["rkpm"] = particleKernelDomain
    # semi-Lagrangian RKPM: every kernel function moves with its particle
    model.particleManagers.append(
        KDBinOrganizedParticleManager(particleKernelDomain, dimension, journal, bondParticlesToKernelFunctions=True)
    )

    for name, particles, nodes in (
        ("tieLeft", model.particleSets["rkpm_left"], model.nodeSets["feLeft_right"]),
        ("tieRight", model.particleSets["rkpm_right"], model.nodeSets["feRight_left"]),
    ):
        model.constraints[name] = ParticleFiniteElementMortarTie(name, model, particles, nodes)

    model.prepareYourself(journal)
    # besides the solution U, the model receives the increment dU and the reaction P of every converged increment

    return model


def solve(model: FEModel, clampedNodes: NodeSet, tipNodes: NodeSet, fieldOutputController, journal: Journal):
    """Clamp the left end, deflect the right end, and solve the quasi-static step with ``NISTParallel``.

    Parameters
    ----------
    model
        The model.
    clampedNodes
        The finite element nodes at the left end.
    tipNodes
        The finite element nodes at the right end.
    fieldOutputController
        The field output controller of the model.
    journal
        The journal.
    """

    stepActions = StepActionCollection()
    stepActions["dirichlet"]["clamped"] = Dirichlet(
        "clamped", clampedNodes, "displacement", {0: 0.0, 1: 0.0}, model, journal
    )
    stepActions["dirichlet"]["tip"] = Dirichlet("tip", tipNodes, "displacement", {1: tipDeflection}, model, journal)

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
        startInc=increment,
        maxInc=increment,
        minInc=increment / 8,
        maxNumInc=40,
    )

    fieldOutputController.initializeJob()
    try:
        step.solve()
    finally:
        fieldOutputController.finalizeJob()


def finiteElementNodes(model: FEModel) -> list:
    """The nodes of the finite elements of a model, by label.

    Parameters
    ----------
    model
        The model.

    Returns
    -------
    list
        The nodes.
    """

    return sorted({node for element in model.elements.values() for node in element.nodes}, key=lambda node: node.label)


def tipReactionAndDeflectionLine(model: FEModel, tipNodes: NodeSet) -> tuple[float, np.ndarray]:
    """The vertical reaction at the deflected tip, and the deflection of the finite element nodes on the beam axis.

    Parameters
    ----------
    model
        The solved model.
    tipNodes
        The finite element nodes at the right end.

    Returns
    -------
    tuple[float, np.ndarray]
        The reaction, and the rows ``(x, deflection)`` of the nodes on the axis, sorted by ``x``.
    """

    displacement = model.nodeFields["displacement"]

    def values(entry, nodes):
        return displacement[entry][[displacement.indexOfNode(node) for node in nodes]]

    tipReaction = values("P", tipNodes)[:, 1].sum()

    axisNodes = sorted(
        (node for node in finiteElementNodes(model) if np.isclose(node.coordinates[1], 0.5 * height)),
        key=lambda node: node.coordinates[0],
    )
    deflectionLine = np.column_stack(([node.coordinates[0] for node in axisNodes], values("U", axisNodes)[:, 1]))

    return tipReaction, deflectionLine


def particleDeflectionAlongTheAxis(model: MPMModel) -> np.ndarray:
    """The deflection of the beam axis in the RKPM part: the mean deflection of the two rows of particles next to the
    axis, at the particles' reference positions.

    Parameters
    ----------
    model
        The solved coupled model.

    Returns
    -------
    np.ndarray
        The rows ``(x, deflection)``, sorted by ``x``.
    """

    centers = np.array([particle.getCenterCoordinates() for particle in model.particles.values()])
    # the particles report their displacement with three components
    displacements = np.array([particle.getResultArray("displacement")[:2] for particle in model.particles.values()])
    referenceCenters = centers - displacements

    rows = np.unique(referenceCenters[:, 1].round(12))
    rowsNextToTheAxis = rows[np.argsort(np.abs(rows - 0.5 * height))[:2]]
    nextToTheAxis = np.isin(referenceCenters[:, 1].round(12), rowsNextToTheAxis)
    xs = np.unique(referenceCenters[nextToTheAxis, 0].round(12))
    deflections = [displacements[nextToTheAxis & np.isclose(referenceCenters[:, 0], x), 1].mean() for x in xs]

    return np.column_stack((xs, deflections))


@performancetiming.timeit("simulation")
def run_sim(elementSize: float = defaultElementSize, particleSize: float = defaultParticleSize):
    """Solve the coupled cantilever.

    Parameters
    ----------
    elementSize
        The size of the finite elements.
    particleSize
        The size of the particles.

    Returns
    -------
    tuple
        The model, and the result array compared with the gold: the total displacement of the finite element nodes,
        by label, then the displacement of the particles, by number.
    """

    journal = Journal()

    model = buildCoupledModel(journal, elementSize, particleSize)

    solve(
        model,
        model.nodeSets["feLeft_left"],
        model.nodeSets["feRight_right"],
        MPMFieldOutputController(model, journal),
        journal,
    )

    displacement = model.nodeFields["displacement"]
    result = np.concatenate(
        (
            displacement["U"][[displacement.indexOfNode(node) for node in finiteElementNodes(model)]].flatten(),
            np.concatenate([particle.getResultArray("displacement") for particle in model.particles.values()]),
        )
    )

    return model, result


def run_finite_element_reference(size: float):
    """The same cantilever, made of finite elements only.

    Parameters
    ----------
    size
        The element size.

    Returns
    -------
    FEModel
        The solved model.
    """

    journal = Journal()

    model = FEModel(2)
    addFiniteElementBeam(model, journal, "beam", 0.0, length, size)
    model.nodeSets["all"] = NodeSet("all", model.nodes.values())
    model.prepareYourself(journal)

    solve(
        model, model.nodeSets["beam_left"], model.nodeSets["beam_right"], FieldOutputController(model, journal), journal
    )

    return model


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    """No matter where pytest is ran, we set the working dir
    to this testscript's parent directory"""

    monkeypatch.chdir(request.fspath.dirname)


def test_finite_element_and_kernel_nodes_share_one_equation_system(tmp_path):
    """The equation system of the coupled model numbers the nodes of the finite elements, the nodes of the kernel
    functions and the tractions of the ties once each. The solution passes between the model and the solver's
    vectors as the two methods require: the finite element nodes carry their total displacement into a rebuilt
    equation system, the kernel nodes start every increment at zero, so that the model holds zero as their solution,
    and all results are written to the model. What the model holds does not depend on when the equation system is
    rebuilt, nor on a restart."""

    model = buildCoupledModel(Journal(), defaultElementSize, defaultParticleSize)
    model.updateConnectivity()
    layout = model.buildEquationSystem()
    dofManager = layout.dofManager

    nElements = len(model.elements)
    assert layout.elements[:nElements] == list(model.elements.values())
    assert layout.elements[nElements:] == list(model.particles.values())

    elementNodes = finiteElementNodes(model)
    kernelNodes = [kernelFunction.node for kernelFunction in model.meshfreeKernelFunctions.values()]
    assert not set(elementNodes) & set(kernelNodes)

    def dofsOf(node):
        return dofManager.idcsOfFieldVariablesInDofVector[node.fields["displacement"]]

    tractionDofs = [
        dofManager.idcsOfScalarVariablesInDofVector[traction]
        for constraint in model.constraints.values()
        for traction in constraint.scalarVariables
    ]
    assert len(tractionDofs) == 2 * 2 * len(model.nodeSets["feLeft_right"])
    allDofs = np.concatenate([dofsOf(node) for node in elementNodes + kernelNodes] + [tractionDofs])
    assert sorted(allDofs) == list(range(dofManager.nDof))

    # the converged state of the last increment, as the model holds it
    displacement = model.nodeFields["displacement"]
    for node in elementNodes + kernelNodes:
        displacement["U"][displacement.indexOfNode(node)] = [node.label, -node.label]
    for constraint in model.constraints.values():
        for i, traction in enumerate(constraint.scalarVariables):
            traction.value = 1.0 + i

    U = dofManager.constructDofVector()
    layout.readStateFromModel(U)
    for node in elementNodes:
        np.testing.assert_array_equal(U[dofsOf(node)], [node.label, -node.label])
    for node in kernelNodes:
        np.testing.assert_array_equal(U[dofsOf(node)], 0.0)
    for constraint in model.constraints.values():
        np.testing.assert_array_equal(
            U[[dofManager.idcsOfScalarVariablesInDofVector[t] for t in constraint.scalarVariables]],
            1.0 + np.arange(len(constraint.scalarVariables)),
        )

    # the solution of a converged increment: the coefficients of the kernel nodes are reset for the next increment
    U[:] = np.arange(dofManager.nDof)
    layout.zeroIncrementalCoefficients(U)
    for node in elementNodes:
        np.testing.assert_array_equal(U[dofsOf(node)], dofsOf(node))
    for node in kernelNodes:
        np.testing.assert_array_equal(U[dofsOf(node)], 0.0)
    np.testing.assert_array_equal(U[tractionDofs], tractionDofs)

    # the results of the increment, written to every node; a static solver leaves the velocity as it is
    displacement["V"][:] = -1.0
    increment = dofManager.constructDofVector()
    increment[:] = np.arange(dofManager.nDof)
    layout.writeResultsToModel(U, 2 * increment, 3 * increment)
    for node in elementNodes + kernelNodes:
        row = displacement.indexOfNode(node)
        np.testing.assert_array_equal(displacement["U"][row], dofsOf(node) if node in elementNodes else 0.0)
        np.testing.assert_array_equal(displacement["dU"][row], 2 * dofsOf(node))
        np.testing.assert_array_equal(displacement["P"][row], 3 * dofsOf(node))
    np.testing.assert_array_equal(displacement["V"], -1.0)
    for constraint in model.constraints.values():
        for traction in constraint.scalarVariables:
            assert traction.value == dofManager.idcsOfScalarVariablesInDofVector[traction]

    held = {entry: np.array(displacement[entry]) for entry in displacement.ENTRIES}

    def assertTheModelHoldsTheSame():
        for entry, values in held.items():
            np.testing.assert_array_equal(displacement[entry], values, err_msg=entry)

    # a rebuilt equation system starts from the same state, and writes it back unchanged
    rebuilt = model.buildEquationSystem()
    U = rebuilt.dofManager.constructDofVector()
    rebuilt.readStateFromModel(U)
    rebuilt.zeroIncrementalCoefficients(U)
    rebuilt.writeResultsToModel(U, None, None)
    assertTheModelHoldsTheSame()

    # a restart restores every entry
    checkpoint = tmp_path / "checkpoint.h5"
    with h5py.File(checkpoint, "w") as restartFile:
        model.writeRestart(restartFile)
    for entry in displacement.ENTRIES:
        displacement[entry][:] = 7.0
    with h5py.File(checkpoint, "r") as restartFile:
        model.readRestart(restartFile)
    assertTheModelHoldsTheSame()


def test_sim(assert_gold):
    model, result = run_sim()

    gold = np.loadtxt("gold.csv")
    assert_gold(result, gold, rtol=1e-8, atol=1e-10)

    # The independent check: the cantilever of finite elements only, with the elements of the coupled model. The two
    # differ only where the particles replace the elements. Refined together, both approach the same limit.
    reference = run_finite_element_reference(defaultElementSize)
    referenceReaction, referenceLine = tipReactionAndDeflectionLine(reference, reference.nodeSets["beam_right"])

    def referenceDeflection(x):
        return np.interp(x, referenceLine[:, 0], referenceLine[:, 1])

    reaction, finiteElementLine = tipReactionAndDeflectionLine(model, model.nodeSets["feRight_right"])
    particleLine = particleDeflectionAlongTheAxis(model)

    assert reaction == pytest.approx(referenceReaction, rel=1e-2)
    for line in (finiteElementLine, particleLine):
        np.testing.assert_allclose(line[:, 1], referenceDeflection(line[:, 0]), atol=2e-3 * abs(tipDeflection))


if __name__ == "__main__":
    model, result = run_sim()

    parser = argparse.ArgumentParser()
    parser.add_argument("--create-gold", dest="create_gold", action="store_true", help="create the gold file.")
    args = parser.parse_args()

    if args.create_gold:
        np.savetxt("gold.csv", result)
