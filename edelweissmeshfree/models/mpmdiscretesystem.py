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
:class:`~edelweissmeshfree.models.mpmmodel.MPMModel` gives EdelweissFE's solvers whenever they (re)build their discrete
system, see :meth:`~edelweissmeshfree.models.mpmmodel.MPMModel.buildDiscreteSystem`."""

from collections.abc import Callable

import numpy as np
from edelweissfe.models.discretesystem import DiscreteSystem
from edelweissfe.numerics.dofvector import DofVector
from edelweissfe.sets.nodeset import NodeSet
from edelweissfe.timesteppers.timestep import TimeStep

from edelweissmeshfree.numerics.doflocations import MPMDofLocations

#: A lumped coefficient of a node (a mass, a damping) below this fraction of the largest one of its field is zero up to
#: round-off: the particles reach its kernel function only where it vanishes. A solver that divides by it would give the
#: node an acceleration of round-off over round-off.
COEFFICIENT_AT_ROUND_OFF = 1e-12

#: The fraction of the stable time increment of the particles' tangent that the explicit scheme of the particles is
#: stable for. Their NSNI stabilization acts on the increment of the step, not on the total displacement, so the scheme
#: is not exactly the central difference method with their tangent: measured on 2D and 3D blocks of particles (support
#: factors 1.6 to 3.4, quadratic and cubic kernels), it was stable up to 0.875 to 0.98 of the tangent's limit.
STABLE_FRACTION_OF_THE_TANGENT_LIMIT = 0.85


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
    def entitiesWithOwnInertia(self) -> list:
        """The particles: they carry their state, and with it their inertia, and their kernel functions move with them;
        see :attr:`~edelweissfe.models.discretesystem.DiscreteSystem.entitiesWithOwnInertia`.

        Returns
        -------
        list
            The particles.
        """

        return self.particles

    def advanceEntitiesWithOwnInertia(
        self, dU: DofVector, timeStep: TimeStep, forEachChunk: Callable[[list, Callable], None]
    ):
        """Every particle takes the increment of the nodes of the kernel functions it was assigned at the start of the
        increment (``updatePhysicsExplicit``), and moves; see
        :meth:`~edelweissfe.models.discretesystem.DiscreteSystem.advanceEntitiesWithOwnInertia`. The particles are
        independent of each other, so they are advanced in the solver's loop over chunks.

        Parameters
        ----------
        dU
            The increment of the solution.
        timeStep
            The increment.
        forEachChunk
            The solver's loop over chunks of independent entities.
        """

        time = timeStep.totalTime
        dT = timeStep.timeIncrement

        def advance(particles):
            for particle in particles:
                particle.updatePhysicsExplicit(dU[particle], time, dT)

        forEachChunk(self.particles, advance)

    def assembleInternalForceAndMomentumOfEntitiesWithOwnInertia(
        self, forEachChunk: Callable[[list, Callable], None]
    ) -> tuple[DofVector, DofVector] | None:
        """The internal force of the particles at the state they accepted (``computePhysicsKernelsExplicit``) and their
        lumped momentum (``computeLumpedMomentum``), at the nodes of their current kernel functions; see
        :meth:`~edelweissfe.models.discretesystem.DiscreteSystem.assembleInternalForceAndMomentumOfEntitiesWithOwnInertia`.
        Each particle adds into its own part of a scatter vector, in the solver's loop over chunks.

        Parameters
        ----------
        forEachChunk
            The solver's loop over chunks of independent entities.

        Returns
        -------
        tuple[DofVector, DofVector] | None
            The internal force and the lumped momentum; None if the model has no particles.
        """

        if not self.particles:
            return None

        internalForce = self.locations.constructDofVector()
        momentum = self.locations.constructDofVector()
        internalForce[:] = momentum[:] = 0.0

        scatterInternalForce = internalForce.createScatterVector()
        scatterMomentum = momentum.createScatterVector()

        def evaluate(particles):
            for particle in particles:
                particle.computePhysicsKernelsExplicit(scatterInternalForce[particle])
                particle.computeLumpedMomentum(scatterMomentum[particle])

        forEachChunk(self.particles, evaluate)

        scatterInternalForce.assembleInto(internalForce)
        scatterMomentum.assembleInto(momentum)

        return internalForce, momentum

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
        assembly of this mass: the explicit dynamic solver ``NED`` divides by it, the implicit dynamic solver adds it to
        the mass of the nodes when it integrates the particles' inertia there.

        A node carries inertia if its mass is positive beyond round-off, more than
        :data:`~edelweissmeshfree.models.mpmdiscretesystem.COEFFICIENT_AT_ROUND_OFF` of the largest one of its field;
        every other node gets zero, the mass of a node without inertia, which both solvers take as such.

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

        return self._assembleLumpedCoefficientOfParticles(
            lambda particle, coefficient: particle.computeLumpedInertia(coefficient), forEachChunk
        )

    def assembleLumpedDampingOfEntitiesWithOwnInertia(
        self, forEachChunk: Callable[[list, Callable], None]
    ) -> DofVector | None:
        """The lumped damping of the particles at the nodes of their current kernel functions, as each particle reports
        it (``computeLumpedDamping``, e.g. :math:`\\eta\\, V_{0,p}\\, T_A(\\boldsymbol{x}_p)` on a nonlocal field with
        the nonlocal viscosity :math:`\\eta`), summed over the particles; see
        :meth:`~edelweissfe.models.discretesystem.DiscreteSystem.assembleLumpedDampingOfEntitiesWithOwnInertia`. Zero
        up to round-off is zero, as for the mass.

        Parameters
        ----------
        forEachChunk
            The solver's loop over chunks of independent entities.

        Returns
        -------
        DofVector | None
            The lumped damping, one value per degree of freedom; None if the model has no particles.
        """

        return self._assembleLumpedCoefficientOfParticles(
            lambda particle, coefficient: particle.computeLumpedDamping(coefficient), forEachChunk
        )

    def _assembleLumpedCoefficientOfParticles(
        self, computeCoefficient: Callable, forEachChunk: Callable[[list, Callable], None]
    ) -> DofVector | None:
        """Assemble a lumped coefficient of the particles (their mass or their damping): each particle adds its own
        into its part of a scatter vector, in the solver's loop over chunks. A coefficient of a node at or below
        :data:`~edelweissmeshfree.models.mpmdiscretesystem.COEFFICIENT_AT_ROUND_OFF` of the largest one of its field is
        zero: field by field, since the coefficients of two fields (a mass and a micro-inertia) are of other units.

        Parameters
        ----------
        computeCoefficient
            ``computeCoefficient(particle, coefficient)`` adds the particle's coefficient to its part.
        forEachChunk
            The solver's loop over chunks of independent entities.

        Returns
        -------
        DofVector | None
            The lumped coefficient, one value per degree of freedom; None if the model has no particles.
        """

        if not self.particles:
            return None

        coefficient = self.locations.constructDofVector()
        coefficient[:] = 0.0
        scatterCoefficient = coefficient.createScatterVector()

        def assemble(particles):
            for particle in particles:
                computeCoefficient(particle, scatterCoefficient[particle])

        forEachChunk(self.particles, assemble)
        scatterCoefficient.assembleInto(coefficient)

        values = coefficient.asPlainArray()
        for indices in self.numbering.idcsOfFieldsInDofVector.values():
            ofTheField = values[indices]
            if ofTheField.size:
                ofTheField[ofTheField <= COEFFICIENT_AT_ROUND_OFF * np.max(ofTheField)] = 0.0
                values[indices] = ofTheField

        return coefficient

    def stableTimeIncrementOfEntitiesWithOwnInertia(
        self, forEachChunk: Callable[[list, Callable], None]
    ) -> float | None:
        """A bound of the stable time increment of ``NED``'s explicit scheme for the particles, from their tangents at
        the current state (``computeTangentOfTheCurrentState``) and the assembled lumped mass and damping. Every
        particle's tangent is evaluated once.

        It is a property of the ASSEMBLED system: a particle's own bound, :math:`\\lambda_\\mathrm{max}(\\boldsymbol{M}_p^{-1}
        \\boldsymbol{K}_p)` as for a finite element, is unbounded, because a particle's stiffness couples nodes at which
        its own lumped mass vanishes (the mass is taken at its centre, its smoothed gradients at its faces).

        **What is bounded.** The degrees of freedom with an inertia are integrated by central differences:
        :math:`\\boldsymbol{M}\\ddot{\\boldsymbol{u}} + \\boldsymbol{C}\\dot{\\boldsymbol{u}} + \\boldsymbol{K}
        \\boldsymbol{u} = \\boldsymbol{0}` is stable for :math:`\\Delta t \\le 2/\\omega_\\mathrm{max}`, with
        :math:`\\omega_\\mathrm{max}^2` the largest eigenvalue of :math:`\\boldsymbol{M}^{-1}\\boldsymbol{K}`. The damping
        does not lower this limit, since ``NED`` takes it with the mean of the velocities before and after the update
        (a damping evaluated with the old velocity alone would). The degrees of freedom without inertia but with a
        damping (a first-order nonlocal field) are integrated by forward Euler, stable for :math:`\\Delta t \\le
        2/\\lambda_\\mathrm{max}(\\boldsymbol{C}^{-1}\\boldsymbol{K})`. The coupling between the two sets is not
        included.

        **Why the absolute value of the unsymmetric tangent.** The tangent of gradient-enhanced particles is
        unsymmetric, and its coupling blocks differ by orders of magnitude once scaled by the masses: the stress depends
        on the nonlocal field as strongly as on the strain (on a boundary of GM7, through the over-nonlocal strain), the
        nonlocal field depends on the strain with a factor of one, and the micro-inertia of the nonlocal field is far
        below the mass. An eigenvalue :math:`\\lambda` of the coupled system sees the geometric mean of the two coupling
        blocks, the symmetric part of the tangent their arithmetic mean: a bound of the symmetric part follows the large
        block alone and falls far below the stable increment whenever a particle loads a boundary. Instead, every
        eigenvalue, real or complex, satisfies :math:`|\\lambda| \\le \\rho(|\\boldsymbol{S}|)`, the spectral radius of
        the absolute value of :math:`\\boldsymbol{S} = \\boldsymbol{D}\\boldsymbol{K}\\boldsymbol{D}` (similar to
        :math:`\\boldsymbol{M}^{-1}\\boldsymbol{K}`, with :math:`\\boldsymbol{D} = \\boldsymbol{M}^{-1/2}`, or
        :math:`\\boldsymbol{C}^{-1/2}` on the first-order set). :math:`|\\boldsymbol{K}|` is accumulated particle by
        particle, which bounds the absolute value of the sum.

        **The bound of** :math:`\\rho(|\\boldsymbol{S}|)` **from one evaluation of the tangents.** The Collatz-Wielandt
        bound :math:`\\rho(|\\boldsymbol{S}|) \\le \\max_i (|\\boldsymbol{S}|\\boldsymbol{w})_i / w_i` holds for every
        positive :math:`\\boldsymbol{w}`; the weights are taken constant on each field: :math:`\\boldsymbol{w} =
        \\sum_f c_f \\boldsymbol{1}_f`. One evaluation of the tangents gives the row sums of :math:`|\\boldsymbol{S}|`
        over the columns of each field, :math:`Z_{if} = \\sum_{j \\in f} |S_{ij}|`, and with them the field matrix
        :math:`Q_{gf} = \\max_{i \\in g} Z_{if}`, the largest row sum of the block of fields :math:`g, f`. With
        :math:`\\boldsymbol{c}` the Perron vector of :math:`\\boldsymbol{Q}`, every row :math:`i` of field :math:`g`
        gives :math:`(|\\boldsymbol{S}|\\boldsymbol{w})_i \\le (\\boldsymbol{Q}\\boldsymbol{c})_g = \\rho(\\boldsymbol{Q})
        w_i`, so :math:`\\rho(|\\boldsymbol{S}|) \\le \\rho(\\boldsymbol{Q})`: the Gershgorin bound per block of fields,
        coupled by the largest eigenvalue of the small field matrix, which takes the geometric mean of the coupling
        blocks. With a single field it is the Gershgorin bound.

        The result is scaled by :data:`STABLE_FRACTION_OF_THE_TANGENT_LIMIT`, since the explicit scheme of the
        particles is not exactly integrated with their tangent: their NSNI stabilization acts on the increment.

        Parameters
        ----------
        forEachChunk
            The solver's loop over chunks of independent entities.

        Returns
        -------
        float | None
            The stable time increment; None if the model has no particles.
        """

        if not self.particles:
            return None

        M = np.asarray(self.assembleLumpedMassOfEntitiesWithOwnInertia(forEachChunk))
        C = np.asarray(self.assembleLumpedDampingOfEntitiesWithOwnInertia(forEachChunk))

        secondOrder = M > 0.0
        firstOrder = ~secondOrder & (C > 0.0)
        scaling = np.zeros_like(M)
        scaling[secondOrder] = 1.0 / np.sqrt(M[secondOrder])
        scaling[firstOrder] = 1.0 / np.sqrt(C[firstOrder])

        # The blocks of the field matrix: a field's degrees of freedom of each order (a field is one or the other).
        nDof = len(M)
        groupOfDof = np.full(nDof, -1)
        orderOfGroup = []
        for indices in self.numbering.idcsOfFieldsInDofVector.values():
            inField = np.zeros(nDof, dtype=bool)
            inField[indices] = True
            for order, ofTheOrder in (("second", secondOrder), ("first", firstOrder)):
                inGroup = inField & ofTheOrder
                if inGroup.any():
                    groupOfDof[inGroup] = len(orderOfGroup)
                    orderOfGroup.append(order)
        nGroups = len(orderOfGroup)

        # Z_if = sum_{j in f} |S_ij|, assembled column by column
        rowSums = [self.locations.constructDofVector() for _ in range(nGroups)]
        for rowSum in rowSums:
            rowSum[:] = 0.0
        scatterRowSums = [rowSum.createScatterVector() for rowSum in rowSums]

        def addAbsoluteTangentRowSums(particles):
            for particle in particles:
                nDofOfParticle = particle.nDof
                K = np.zeros(nDofOfParticle * nDofOfParticle)
                particle.computeTangentOfTheCurrentState(K)
                indices = self.locations.ofEntity[particle]
                scaledIndicatorOfGroup = np.zeros((nDofOfParticle, nGroups))
                inAGroup = groupOfDof[indices] >= 0
                scaledIndicatorOfGroup[inAGroup, groupOfDof[indices][inAGroup]] = scaling[indices][inAGroup]
                # the tangent is column-major: row i is the equation of degree of freedom i
                absoluteTangentRowSums = np.abs(K.reshape(nDofOfParticle, nDofOfParticle, order="F")) @ (
                    scaledIndicatorOfGroup
                )
                for group, scatterRowSum in enumerate(scatterRowSums):
                    ofTheParticle = scatterRowSum[particle]
                    ofTheParticle += absoluteTangentRowSums[:, group]

        forEachChunk(self.particles, addAbsoluteTangentRowSums)
        Z = np.zeros((nDof, nGroups))
        for group, (rowSum, scatterRowSum) in enumerate(zip(rowSums, scatterRowSums)):
            scatterRowSum.assembleInto(rowSum)
            Z[:, group] = scaling * np.asarray(rowSum)

        Q = np.array([[np.max(Z[groupOfDof == g, f]) for f in range(nGroups)] for g in range(nGroups)])

        def perronRoot(order: str) -> float:
            """The largest eigenvalue of the nonnegative field matrix of the blocks of the given order."""
            groups = [group for group, ofGroup in enumerate(orderOfGroup) if ofGroup == order]
            return float(np.max(np.abs(np.linalg.eigvals(Q[np.ix_(groups, groups)])))) if groups else 0.0

        stable = np.inf
        omegaSquared = perronRoot("second")
        if omegaSquared > 0.0:
            stable = 2.0 / np.sqrt(omegaSquared)
        rate = perronRoot("first")
        if rate > 0.0:
            stable = min(stable, 2.0 / rate)

        return STABLE_FRACTION_OF_THE_TANGENT_LIMIT * stable

    def carriersOfTheMotion(self) -> set[str]:
        """Where the model holds the motion of its particles; see
        :meth:`~edelweissfe.models.discretesystem.DiscreteSystem.carriersOfTheMotion`.
        ``"entities"`` if a particle that integrates its inertia (one with Newmark parameters) carries a velocity or an
        acceleration in its state, ``"nodes"`` if an incremental node of the active domain carries one in the entries
        ``V`` and ``A`` of the model's node fields.

        The implicit dynamic solver keeps the motion in the particles with ``inertia=entities``, at the nodes
        with ``inertia=nodes``. The explicit dynamic solver ``NED`` keeps it in both: the velocity of the nodes is its
        state, and every particle takes its own from them as it advances.

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
            :meth:`~edelweissfe.solvers.base.nonlinearsolverbase.NonlinearSolverBase.forEachChunk`.
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
