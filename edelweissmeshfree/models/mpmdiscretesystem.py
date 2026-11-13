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

"""What the discrete system of a material point or RKPM model is made of: the answer an
:class:`~edelweissmeshfree.models.mpmmodel.MPMModel` gives EdelweissFE's implicit solvers and
:class:`~edelweissmeshfree.solvers.meshfreeexplicitdynamic.MeshfreeExplicitDynamic` whenever they (re)build
their discrete system, see :meth:`~edelweissmeshfree.models.mpmmodel.MPMModel.buildDiscreteSystem`."""

from collections.abc import Callable

import numpy as np
from edelweissfe.models.discretesystem import DiscreteSystem
from edelweissfe.numerics.dofvector import DofVector
from edelweissfe.sets.nodeset import NodeSet
from edelweissfe.timesteppers.timestep import TimeStep

from edelweissmeshfree.numerics.doflocations import MPMDofLocations

#: The lumped mass of a node below this fraction of the largest one is zero up to round-off: the particles reach its
#: kernel function only where it vanishes. A solver that divides by it would give the node an acceleration of round-off
#: over round-off.
MASS_AT_ROUND_OFF = 1e-12


class MPMDiscreteSystem(DiscreteSystem):
    """The discrete system of a meshfree model, on its active domain: the background cells that hold
    material points, the nodes these cells and the particles' kernel functions reach, and the finite elements with
    their nodes.

    Two kinds of nodes take part. The nodes of finite elements (and of cell elements) carry their total solution from
    increment to increment, as in a finite element model. The nodes of the grid and of the kernel functions do not: the
    material point method and RKPM are incremental on the current configuration -- the cells are reset after every
    increment, and the kernel functions move with their particles, while the state lives in the material points and
    particles. Their nodal values are the coefficients of one increment: every increment starts them at zero, so that
    their solution ``U`` is zero, and their increment ``dU`` is their meaningful nodal value.

    The model's node fields are read and written straight, by node: read on the nodes that carry an entry (see
    :meth:`readEntryOfNodeField`), written on the nodes of the active domain from the vectors of the discrete system,
    zero on all other nodes (see :meth:`writeEntryOfNodeField`).

    Parameters
    ----------
    model
        The model.
    locations
        The locations of the entities in the numbering of the degrees of freedom of the active domain; the finite
        elements, the active cells (those that hold material points, sorted by their number), the cell elements and the
        particles, in the order in which the solver computes them, are the elements of the discrete system.
    nodesWithPersistentFieldValues
        The nodes that carry their total solution from increment to increment: those of the finite elements and of
        the cell elements.
    nodesWithVolatileFieldValues
        The nodes of the active cells and of the particles' kernel functions.
    modelForLinearSolver
        The model reduced to its active domain, for the linear solver.
    """

    def __init__(
        self,
        model,
        locations: MPMDofLocations,
        nodesWithPersistentFieldValues: NodeSet,
        nodesWithVolatileFieldValues: NodeSet,
        modelForLinearSolver,
    ):
        super().__init__(model, locations)
        #: The nodes that carry their total solution from increment to increment.
        self.nodesWithPersistentFieldValues = nodesWithPersistentFieldValues
        #: The nodes of the active domain whose degrees of freedom are the coefficients of one increment: those of
        #: the grid and of the kernel functions that no finite element or cell element shares.
        self.incrementalNodes = NodeSet(
            "incremental nodes of the active domain",
            [node for node in nodesWithVolatileFieldValues if node not in nodesWithPersistentFieldValues],
        )
        #: The nodes of the active domain, on which the discrete system writes the model's node fields.
        self.activeNodes = NodeSet(
            "nodes of the active domain",
            sorted(set(nodesWithPersistentFieldValues) | set(nodesWithVolatileFieldValues), key=lambda n: n.label),
        )
        self.modelForLinearSolver = modelForLinearSolver
        #: The material points of the model, a list the solver hands out in chunks.
        self.materialPoints = list(model.materialPoints.values())

    @property
    def activeCells(self) -> list:
        """The cells that hold material points, sorted by their number, i.e. the cells of the locations."""

        return self.locations.cells

    @property
    def particles(self) -> list:
        """The particles of the model, a list the solver hands out in chunks, i.e. the particles of the locations."""

        return self.locations.particles

    @property
    def cellsInterpolatingToMaterialPoints(self) -> list:
        """The active cells and the cell elements, which interpolate the solution to the material points they hold."""

        return [*self.locations.cells, *self.locations.cellElements]

    def _newLocations(self) -> MPMDofLocations:
        """The new locations of :meth:`~edelweissfe.models.discretesystem.DiscreteSystem.withNewLocations` after a
        change of the connectivity that leaves the active nodes as they are (the model reports
        :attr:`~edelweissfe.models.change.Change.LOCATIONS`, see
        :meth:`~edelweissmeshfree.models.mpmmodel.MPMModel.updateConnectivity`): the particles whose kernel functions
        changed, and the active constraints, are located anew. The numbering of the nodes stands, so that is what
        rebuilding would give, at a fraction of its cost.

        Returns
        -------
        MPMDofLocations
            The new locations.
        """

        return self.locations.withNewLocationsOf(
            self.model.activeConstraints(),
            [
                particle
                for particleManager in self.model.particleManagers
                for particle in particleManager.particlesWithChangedKernelFunctions
            ],
        )

    def readEntryOfNodeField(self, dofVector: DofVector, nodeField, entry: str):
        """Read one entry of one node field of the model into a vector of the discrete system, by node; every read of
        :meth:`~edelweissfe.models.discretesystem.DiscreteSystem.readStateFromModel` goes through here.
        The solution ``U`` is read at the nodes that carry their total solution (those of finite elements and cell
        elements): the incremental nodes start every increment at zero. The velocity ``V`` and the acceleration ``A``
        are read at every node of the active domain: they are carried by node from one discrete system to the next, so
        that a node keeps them when the active domain changes around it, and a node that has just become active is at
        rest.

        Parameters
        ----------
        dofVector
            The vector of the discrete system, zero, overwritten on the degrees of freedom of the field.
        nodeField
            The node field of the model.
        entry
            The entry.
        """

        nodes = self.nodesWithPersistentFieldValues if entry == "U" else self.activeNodes
        if nodes:
            self.numbering.writeNodeFieldToDofVector(dofVector, nodeField, entry, nodes)

    def zeroIncrementalCoefficients(self, U: DofVector):
        """Set the degrees of freedom of the incremental nodes (:attr:`incrementalNodes`) to zero in the solution
        vector of a converged increment: their solution is the coefficients of one increment, and the next increment
        starts them at zero.

        Parameters
        ----------
        U
            The solution vector of the converged increment, modified in place.
        """

        for fieldName in self.numbering.fields:
            indices, _ = self.numbering.locateFieldOnNodeSet(fieldName, self.incrementalNodes)
            U[indices] = 0.0

    def writeEntryOfNodeField(self, dofVector: DofVector, nodeField, entry: str):
        """Write a vector of the discrete system to one entry of one node field of the model, by node: on the nodes of
        the active domain, and zero on all other nodes of the model.

        Parameters
        ----------
        dofVector
            The vector of the discrete system.
        nodeField
            The node field of the model.
        entry
            The entry.
        """

        nodeField[entry][:] = 0.0
        self.numbering.writeDofVectorToNodeField(dofVector, nodeField, entry, self.activeNodes)

    @property
    def elementsWithNodalInertia(self) -> list:
        """The finite elements, whose inertia, damping and stable time increment a solver assembles element by
        element, at the nodes. The cells, the cell elements and the particles contribute nothing there: the particles
        carry the state, and with it the inertia, which they integrate themselves, or which a solver lumps to the nodes
        of their kernel functions (see :meth:`assembleLumpedMassOfEntitiesWithOwnInertia`).

        Returns
        -------
        list
            The finite elements, in the order in which the solver computes them.
        """

        return self.locations.finiteElements

    def dofsWithNodalInertia(self, fieldNames: list) -> np.ndarray:
        """The degrees of freedom of the given fields whose inertia is that of the nodes: those of the nodes that carry
        their total solution, i.e. of the finite elements and of the cell elements. Only the finite elements assemble
        a mass there (:attr:`elementsWithNodalInertia`). The nodes of the kernel functions are left out: the particles
        carry the state, and with it the inertia, which they integrate themselves, or which a solver lumps to these
        nodes (see :meth:`assembleLumpedMassOfEntitiesWithOwnInertia`).

        Parameters
        ----------
        fieldNames
            The names of the fields.

        Returns
        -------
        np.ndarray
            The indices of the degrees of freedom in the vectors of the discrete system, sorted.
        """

        isWithNodalInertia = np.zeros(self.numbering.nDof, dtype=bool)
        if self.nodesWithPersistentFieldValues:
            for fieldName in fieldNames:
                indices, _ = self.numbering.locateFieldOnNodeSet(fieldName, self.nodesWithPersistentFieldValues)
                isWithNodalInertia[indices] = True

        return np.flatnonzero(isWithNodalInertia)

    def assembleLumpedMassOfEntitiesWithOwnInertia(
        self, forEachChunk: Callable[[list, Callable], None]
    ) -> DofVector | None:
        """The lumped mass of the particles at the nodes of their current kernel functions,
        :math:`m_A = \\sum_p \\rho_0\\, V_{0,p}\\, T_A(\\boldsymbol{x}_p)`, as each particle reports it
        (``computeLumpedInertia``), summed over the particles; see
        :meth:`~edelweissfe.models.discretesystem.DiscreteSystem.assembleLumpedMassOfEntitiesWithOwnInertia`. The one
        assembly of this mass: the explicit solver divides by it, the implicit dynamic solver adds it to the mass of
        the nodes when it integrates the particles' inertia there.

        A node carries inertia if its mass is positive beyond round-off, more than
        :data:`~edelweissmeshfree.models.mpmdiscretesystem.MASS_AT_ROUND_OFF` of the largest one; every other node
        gets zero, the mass of a node without inertia, which both solvers take as such.

        Parameters
        ----------
        forEachChunk
            The solver's loop over chunks of independent entities: the particles are evaluated in it, each into its own
            part of a scatter vector.

        Returns
        -------
        DofVector | None
            The lumped mass, one value per degree of freedom; None if the model has no particles.
        """

        if not self.particles:
            return None

        M = self.locations.constructDofVector()
        M[:] = 0.0
        scatterM = M.createScatterVector()

        def assemble(particles):
            for particle in particles:
                particle.computeLumpedInertia(scatterM[particle])

        forEachChunk(self.particles, assemble)
        scatterM.assembleInto(M)

        if M.size:
            M[M <= MASS_AT_ROUND_OFF * np.max(M)] = 0.0

        return M

    def carriersOfTheMotion(self) -> set[str]:
        """Where the model holds the motion of its particles; see
        :meth:`~edelweissfe.models.discretesystem.DiscreteSystem.carriersOfTheMotion`.
        ``"entities"`` if a particle that integrates its inertia (one with Newmark parameters) carries a velocity or an
        acceleration in its state, ``"nodes"`` if an incremental node of the active domain carries one in the entries
        ``V`` and ``A`` of the model's node fields.

        The implicit dynamic solver keeps the motion in the particles with ``inertia=entities``, at the nodes
        with ``inertia=nodes``. The explicit solver keeps it in both: the velocity of the nodes is its state,
        and every particle takes its own from them as it advances.

        Returns
        -------
        set[str]
            ``"entities"``, ``"nodes"``, both or none.
        """

        carriers = set()

        if any(
            np.any(particle.getResultArray("velocity")) or np.any(particle.getResultArray("acceleration"))
            for particle in self.particles
            if "newmark-beta beta" in particle.propertyNames
        ):
            carriers.add("entities")

        if self.incrementalNodes:
            rate = self.locations.constructDofVector()
            for fieldName in self.numbering.fields:
                for entry in ("V", "A"):
                    rate[:] = 0.0
                    self.numbering.writeNodeFieldToDofVector(
                        rate, self.model.nodeFields[fieldName], entry, self.incrementalNodes
                    )
                    if np.any(rate):
                        carriers.add("nodes")
                        return carriers

        return carriers

    def updateMaterialPoints(self, dU: DofVector, timeStep: TimeStep, forEachChunk: Callable[[list, Callable], None]):
        """Update the material points with the trial solution increment, before the solver computes the cells, which
        integrate them: the material points return to their converged state, every active cell (and cell element)
        interpolates the solution increment to the material points it holds, and the material points evaluate their
        material. Each of the three is a loop over independent entities -- a material point lies in one cell -- in the
        solver's loop over chunks, on its threads for a parallel solver.

        Parameters
        ----------
        dU
            The trial solution increment.
        timeStep
            The time step.
        forEachChunk
            The solver's loop over chunks of independent entities, see
            :meth:`~edelweissfe.solvers.nonlinearimplicitstatic.NIST.forEachChunk`.
        """

        time = timeStep.totalTime
        dT = timeStep.timeIncrement

        def prepare(materialPoints):
            for materialPoint in materialPoints:
                materialPoint.prepareYourself(time, dT)

        def interpolate(cells):
            for cell in cells:
                cell.interpolateFieldsToMaterialPoints(dU[cell])

        def compute(materialPoints):
            for materialPoint in materialPoints:
                materialPoint.computeYourself(time, dT)

        forEachChunk(self.materialPoints, prepare)
        forEachChunk(self.cellsInterpolatingToMaterialPoints, interpolate)
        forEachChunk(self.materialPoints, compute)
