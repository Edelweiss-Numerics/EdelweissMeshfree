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

"""Explicit dynamics of a meshfree (RKPM) model: central differences on a discretization that moves with its particles.

The scheme is the leapfrog of EdelweissFE's
:class:`~edelweissfe.solvers.nonlinearexplicitdynamic.NED`: the velocity is staggered half an increment behind the
displacement, and the acceleration is the net force :math:`\\boldsymbol{f}` (external minus internal, with the loads and
constraints) divided by a lumped mass,

.. math::

    \\boldsymbol{v}_{n+1/2} = \\boldsymbol{v}_{n-1/2}
        + \\tfrac{1}{2} \\left( \\Delta t_{n+1} + \\Delta t_n \\right) \\boldsymbol{M}^{-1} \\boldsymbol{f}_n, \\qquad
    \\Delta \\boldsymbol{u}_{n+1} = \\Delta t_{n+1}\\, \\boldsymbol{v}_{n+1/2},

with :math:`\\Delta t_0 = 0`: the first increment of motion starts from the velocity at rest,
:math:`\\boldsymbol{v}_{1/2} = \\boldsymbol{v}_0 + \\tfrac{1}{2} \\Delta t_1 \\boldsymbol{M}^{-1} \\boldsymbol{f}_0`.

What differs is the discretization it is applied to. NED integrates nodes that carry their total solution on a fixed
mesh, and an element advances its state and returns its force in one call. The kernel functions of an RKPM model move
with its particles, and the state lives in the particles (updated Lagrangian), so one increment has three parts:

1. **advance**: every particle takes the increment :math:`\\Delta \\boldsymbol{u}_{n+1}` on the kernel functions it was
   assigned at :math:`t_n` (``updatePhysicsExplicit``), and moves;
2. **reconnect**: the model assigns the particles to the kernel functions they reach now
   (:meth:`~edelweissmeshfree.models.mpmmodel.MPMModel.updateConnectivity`), and the equation system follows;
3. **evaluate**: on the new kernel functions, the particles return their internal force, their lumped mass and their
   lumped momentum, and the loads and constraints their forces, at the configuration just accepted.

The lumped mass is therefore assembled anew in every increment, and the nodal values are coefficients of the current
kernel functions. The velocity is the state the next increment starts from, so the model owns it, as it owns the
velocity of a finite element model (see EdelweissFE's documentation page *Node field results*): the velocity of every
node of the kernel functions is written to the entry ``V`` of the node fields at acceptance, and read back by node
whenever the equation system is rebuilt. A node keeps its velocity while the active nodes change around it, and a node
whose kernel function has just become active starts at rest. If the option ``projectVelocityEveryIncrement`` says so,
the velocity is instead projected from the particles' momentum in every increment,
:math:`\\boldsymbol{v} = \\boldsymbol{M}^{-1} \\boldsymbol{p}`.

At acceptance, the results reach the model in two parts, each through the layout of the equation system that holds
them: the increment ``dU`` (and the velocity it was taken with) on the kernel functions of its start, before
reconnecting; the force ``P`` and the velocity the next increment starts from on the kernel functions of its end.
``V`` is the staggered velocity :math:`\\boldsymbol{v}_{n+1/2}`, as in NED. ``P`` is the internal force with the
inertial force, :math:`\\boldsymbol{f}_\\mathrm{int} + \\boldsymbol{M} \\boldsymbol{a}_n`, where
:math:`\\boldsymbol{a}_n = \\boldsymbol{M}^{-1} \\boldsymbol{f}_n` is the acceleration of the scheme: at a node with
mass, this is the force of the loads and constraints. The solution ``U`` of the kernel functions' nodes is zero: they
carry no total solution.

The time increment is the step's: particles report no critical time step. The first increment of a step must be the
zero increment, which evaluates the forces at rest and projects the initial velocity from the particles' momentum, so
that the first increment of motion starts from both. A resumed step continues from the velocity of the model and the
net force and inverse mass of its checkpoint instead.
"""

from copy import deepcopy
from dataclasses import dataclass

import edelweissfe.utils.performancetiming as performancetiming
import numpy as np
from edelweissfe.numerics.dofmanager import DofVector
from edelweissfe.numerics.parallelizationutilities import (
    getNumberOfWorkerThreads,
    mapChunksInParallel,
    reportThreadAvailability,
)
from edelweissfe.solvers.base.nonlinearsolverbase import NonlinearSolverBase
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.utils.exceptions import CutbackRequest, StepFailed
from edelweissfe.utils.schema import schemaField

from edelweissmeshfree.constraints.base.mpmconstraintbase import MPMConstraintBase
from edelweissmeshfree.models.mpmmodel import MPMModel


@dataclass(frozen=True)
class MeshfreeExplicitDynamicSchema:
    """The options of :class:`MeshfreeExplicitDynamic`; mirrors its ``SolverSpecificOptions``."""

    projectVelocityEveryIncrement: bool = schemaField(
        description=(
            "Project the velocity from the particles' momentum in every increment, not only when the equation system "
            "numbers other nodes than before. Projecting dissipates more; it is what the material point method does."
        ),
        dtype=bool,
        default=False,
    )


class MeshfreeExplicitDynamic(NonlinearSolverBase):
    """Explicit dynamics of a meshfree model whose particles carry their own state; see the module documentation.

    It is an EdelweissFE solver: an EdelweissFE step (e.g.
    :class:`~edelweissfe.steps.adaptivestep.AdaptiveStep`) runs its increments, and its output managers and restart
    checkpoints are EdelweissFE's. The particles are evaluated on all threads (``OMP_NUM_THREADS``).

    Parameters
    ----------
    jobInfo
        The job information.
    journal
        The journal.
    **kwargs
        The options, see :class:`MeshfreeExplicitDynamicSchema`.
    """

    identification = "MeshfreeExplicitDynamic"

    schema = MeshfreeExplicitDynamicSchema

    SolverSpecificOptions = {
        "projectVelocityEveryIncrement": False,
    }

    #: The state an increment starts from that lives within one equation system, and so is not the model's: the last
    #: increment of motion (its size enters the velocity update), and the net force and the inverse lumped mass of the
    #: configuration last accepted. The velocity is the model's (entry ``V`` of its node fields).
    checkpointedState = {
        "prevTimeStep": TimeStep,
        "_netForce": np.ndarray,
        "_Minv": np.ndarray,
    }

    def __init__(self, jobInfo, journal, **kwargs):
        self.journal = journal
        self.options = deepcopy(self.SolverSpecificOptions)
        self._updateOptions(kwargs, journal, strict=True)

        #: The last increment of motion; None before the first one of a step.
        self.prevTimeStep = None
        #: The layout of the equation system on the current kernel functions; None until the first increment.
        self._layout = None
        #: The velocity, staggered half an increment behind the displacement.
        self._V = None
        #: The net force, external minus internal, at the configuration last accepted; None before the first
        #: evaluation of a step.
        self._netForce = None
        #: The inverse of the lumped mass at the configuration last accepted; zero where the mass is not positive.
        self._Minv = None
        #: The increment of the solution.
        self._dU = None
        #: The particles, a list the threads take chunks of.
        self._particles = None

    def validateModelCapabilities(self, model: MPMModel):
        """Refuse what this solver does not integrate, on top of the base checks: everything but particles, i.e.
        finite elements, cells with their material points, and cell elements; constraints with scalar variables of
        their own (Lagrange multipliers), which carry no inertia and therefore have no equation of motion; and
        constraints that do not act in an explicit analysis
        (:attr:`~edelweissmeshfree.constraints.base.mpmconstraintbase.MPMConstraintBase.actsInExplicitAnalysis`): the
        constraints are evaluated at the configuration just accepted, where the nodes of the kernel functions carry
        neither a solution nor an increment, so a constraint on the increment would act as if nothing had moved.

        A constraint that is switched off at the start of the step takes no part in it: switched off from outside, it
        stays off, and the constraints that switch themselves on by their reach, the contacts, act in an explicit
        analysis.

        Parameters
        ----------
        model
            The model.
        """

        super().validateModelCapabilities(model)

        notParticles = {
            "finite elements": model.elements,
            "material points": model.materialPoints,
            "cell elements": model.cellElements,
        }
        for what, entities in notParticles.items():
            if entities:
                raise NotImplementedError(f"The model has {what}; {self.identification} integrates particles only.")

        for name, constraint in model.constraints.items():
            # a constraint that is switched off has no place in the equation system
            if not constraint.active:
                continue
            if constraint.getNumberOfAdditionalNeededScalarVariables():
                raise NotImplementedError(
                    f"Constraint '{name}' introduces scalar variables, which carry no inertia, so "
                    f"{self.identification} has no equation of motion for them. Only constraints that act through "
                    "nodal forces (penalty formulations) are supported."
                )
            if not (isinstance(constraint, MPMConstraintBase) and constraint.actsInExplicitAnalysis):
                raise NotImplementedError(
                    f"Constraint '{name}' ({type(constraint).__name__}) does not act in an explicit analysis: "
                    f"{self.identification} evaluates the constraints at the configuration just accepted, where the "
                    "nodes of the kernel functions carry neither a solution nor an increment, so a constraint that "
                    "reads them would act as if nothing had moved. A constraint that acts here measures its condition "
                    "at the particles' current positions, see MPMConstraintBase.actsInExplicitAnalysis."
                )

    def beginStep(self, step, model: MPMModel, fieldOutputController, outputmanagers):
        """Start the step; see
        :meth:`~edelweissfe.solvers.base.nonlinearsolverbase.NonlinearSolverBase.beginStep`. The equation system is
        built by the first :meth:`prepareIncrement`. A resumed step gets the state of :attr:`checkpointedState` back
        right after this.

        Parameters
        ----------
        step
            The step.
        model
            The model.
        fieldOutputController
            The field output controller.
        outputmanagers
            The output managers.
        """

        self.validateModelCapabilities(model)
        reportThreadAvailability(getNumberOfWorkerThreads(), self.journal, self.identification)

        self.applyStepActionsAtStepStart(model, step)

        self.prevTimeStep = None
        self._layout = None
        self._V = self._netForce = self._Minv = None
        self._particles = list(model.particles.values())

    def prepareIncrement(self, step, model: MPMModel, isRetry: bool):
        """Build the equation system at the start of the step, with the velocity of the model; afterwards it follows
        the particles in :meth:`acceptIncrement`. A resumed step takes over the net force and inverse mass of its
        checkpoint.

        Parameters
        ----------
        step
            The step.
        model
            The model.
        isRetry
            Never True: an explicit increment is not retried.
        """

        if self._layout is not None:
            return

        model.updateConnectivity()
        self._adoptLayout(model.buildEquationSystem(withSystemMatrix=False))

        if self._netForce is not None:
            # resumed: the checkpoint holds the vectors of the equation system just rebuilt from the same state
            dofManager = self._layout.dofManager
            if len(self._netForce) != dofManager.nDof:
                raise ValueError(
                    "The checkpoint holds a net force of {:} entries, but the equation system rebuilt from the "
                    "resumed model has {:}.".format(len(self._netForce), dofManager.nDof)
                )
            self._netForce = self._asDofVector(self._netForce)
            self._Minv = self._asDofVector(self._Minv)

    def attemptIncrement(self, step, model: MPMModel, timeStep: TimeStep):
        """Advance: the central-difference update of the velocity and the increment, and the particles take the
        increment on the kernel functions they were assigned at the start of it.

        Parameters
        ----------
        step
            The step.
        model
            The model.
        timeStep
            The increment.

        Raises
        ------
        StepFailed
            If a particle cannot take the increment: the time increment is set by stability, not by convergence, so it
            is not cut back.
        """

        dT = timeStep.timeIncrement

        if self._netForce is None and dT != 0.0:
            raise ValueError(
                "The first increment of a step must be the zero increment for {:}, which evaluates the forces at rest "
                "and projects the initial velocity, before anything moves.".format(self.identification)
            )

        dU = self._dU
        dU[:] = 0.0
        if dT != 0.0:
            with performancetiming.timeit("kinematic update"):
                # The velocity lives at the half increments, the force at the full ones, so the velocity advances by
                # the time between two half increments. The first increment of motion starts from the velocity at
                # rest, a full increment earlier: it advances by half an increment (the previous one counts zero).
                previousTimeIncrement = self.prevTimeStep.timeIncrement if self.prevTimeStep is not None else 0.0
                self._V += self._Minv * self._netForce * (0.5 * (dT + previousTimeIncrement))
                np.multiply(self._V, dT, out=dU)

        self.applyStepActionsAtIncrementStart(model, timeStep, step.actions)

        time = timeStep.totalTime

        def advance(particles):
            for particle in particles:
                particle.updatePhysicsExplicit(dU[particle], time, dT)

        try:
            with performancetiming.timeit("advance particles"):
                mapChunksInParallel(self._particles, advance)
        except CutbackRequest as e:
            raise StepFailed(
                "A particle cannot take increment {:}: {:}. The explicit time increment is set by stability, not by "
                "convergence, so it is not cut back.".format(timeStep.number, e)
            ) from e

    def acceptIncrement(self, step, model: MPMModel, timeStep: TimeStep):
        """Accept the advanced particles, reconnect, and evaluate on the new kernel functions what the next increment
        starts from: the net force, the inverse lumped mass and, when due, the velocity projected from the momentum.
        The results reach the model in two parts, see the module documentation: the increment and its velocity before
        reconnecting, through the layout they were computed on; the force and the velocity the next increment starts
        from after evaluating, through the layout of the new kernel functions.

        Parameters
        ----------
        step
            The step.
        model
            The model.
        timeStep
            The increment.
        """

        if timeStep.timeIncrement > 0.0:
            self.prevTimeStep = timeStep

        self.updateRigidBodies(model, timeStep)

        with performancetiming.timeit("accept state"):
            model.advanceToTime(timeStep.totalTime)

        with performancetiming.timeit("write results to nodes"):
            self._layout.writeResultsToModel(self._noSolution(), self._dU, None, V=self._V)

        self._reconnect(model)

        firstEvaluationOfStep = self._netForce is None
        self._netForce, self._Minv, momentum, internalAndInertialForce = self._evaluate(model, step, timeStep)

        if firstEvaluationOfStep or self.options["projectVelocityEveryIncrement"]:
            self._V = momentum * self._Minv

        with performancetiming.timeit("write results to nodes"):
            self._layout.writeResultsToModel(self._noSolution(), None, internalAndInertialForce, V=self._V)

    def endStep(self, step, model: MPMModel):
        """Nothing to finish.

        Parameters
        ----------
        step
            The step.
        model
            The model.
        """

    def solveIncrement(self, *args):
        """Not used: an increment is :meth:`attemptIncrement` and :meth:`acceptIncrement`."""

        raise NotImplementedError(f"{self.identification} solves an increment in attemptIncrement.")

    @performancetiming.timeit("reconnect")
    def _reconnect(self, model: MPMModel):
        """Assign the particles to the kernel functions they reach now, and let the equation system follow: if the
        model's active nodes cannot change (:attr:`~edelweissmeshfree.models.mpmmodel.MPMModel.activeNodesAreFixed`),
        only the particles that changed and the constraints are located anew, and the velocity stays as it is;
        otherwise the equation system is rebuilt, and reads the velocity of its nodes from the model.

        Parameters
        ----------
        model
            The model.
        """

        if not model.updateConnectivity():
            return

        if model.activeNodesAreFixed:
            self._layout.relocateParticlesAndConstraints()
            self._dU = self._layout.dofManager.constructDofVector()
            self._V = self._asDofVector(self._V)
            return

        self._adoptLayout(model.buildEquationSystem(withSystemMatrix=False))

    def _adoptLayout(self, layout):
        """Work on a (re)built layout: vectors sized and located by its DOF manager, and the velocity of its nodes read
        from the model, by node.

        Parameters
        ----------
        layout
            The layout of the equation system.
        """

        self._layout = layout
        self.theDofManager = layout.dofManager
        self._dU = layout.dofManager.constructDofVector()
        self._V = layout.dofManager.constructDofVector()
        self._V[:] = 0.0
        layout.readStateFromModel(self._noSolution(), V=self._V)

    def _noSolution(self) -> DofVector:
        """The solution of the current equation system: zero, since the nodes of the kernel functions carry no total
        solution (they move with the particles), and this solver integrates nothing else.

        Returns
        -------
        DofVector
            The zero vector.
        """

        noSolution = self._layout.dofManager.constructDofVector()
        noSolution[:] = 0.0
        return noSolution

    def _asDofVector(self, values: np.ndarray) -> DofVector:
        """A vector of the current equation system holding the given values.

        Parameters
        ----------
        values
            The values, one per degree of freedom.

        Returns
        -------
        DofVector
            The vector.
        """

        vector = self._layout.dofManager.constructDofVector()
        vector[:] = values
        return vector

    @performancetiming.timeit("evaluate")
    def _evaluate(self, model: MPMModel, step, timeStep: TimeStep) -> tuple[DofVector, DofVector, DofVector]:
        """Evaluate at the configuration just accepted: the particles' internal forces, lumped masses and lumped
        momenta, on all threads, then the loads and constraints. All of them add residual contributions (an internal
        force, the negative of an external one), and the net force is their negative.

        Parameters
        ----------
        model
            The model.
        step
            The step, for its loads.
        timeStep
            The increment just accepted.

        Returns
        -------
        tuple[DofVector, DofVector, DofVector, DofVector]
            The net force, the inverse lumped mass (zero where the mass is not positive), the lumped momentum, and the
            internal force with the inertial force, :math:`\\boldsymbol{f}_\\mathrm{int} + \\boldsymbol{M}
            \\boldsymbol{a}`, the result ``P`` the model receives.
        """

        dofManager = self._layout.dofManager
        R = dofManager.constructDofVector()
        M = dofManager.constructDofVector()
        momentum = dofManager.constructDofVector()
        R[:] = M[:] = momentum[:] = 0.0

        scatterR = R.createScatterVector()
        scatterM = M.createScatterVector()
        scatterMomentum = momentum.createScatterVector()

        def evaluate(particles):
            for particle in particles:
                particle.computePhysicsKernelsExplicit(scatterR[particle])
                particle.computeLumpedInertia(scatterM[particle])
                particle.computeLumpedMomentum(scatterMomentum[particle])

        mapChunksInParallel(self._particles, evaluate)

        scatterR.assembleInto(R)
        scatterM.assembleInto(M)
        scatterMomentum.assembleInto(momentum)

        internalForce = dofManager.constructDofVector()
        internalForce[:] = R

        # The configuration is accepted: no increment is left, and the nodes of the kernel functions carry no total
        # solution. Loads and constraints evaluate at the particles' positions.
        noSolution = self._noSolution()
        R, _ = self.assembleLoads(self.collectLoads(step.actions), noSolution, noSolution, R, None, timeStep)
        R = self.assembleConstraintForces(model.constraints, noSolution, noSolution, R, timeStep)

        P = dofManager.constructDofVector()
        np.negative(R, out=P)

        # A node whose kernel function the particles reach only where it vanishes has no positive mass (exactly zero,
        # or zero up to round-off): it carries no inertia, hence no equation of motion, and keeps a zero velocity.
        Minv = dofManager.constructDofVector()
        Minv[:] = 0.0
        hasInertia = M > 0.0
        Minv[hasInertia] = 1.0 / M[hasInertia]

        acceleration = Minv * P
        internalAndInertialForce = dofManager.constructDofVector()
        internalAndInertialForce[:] = internalForce + M * acceleration

        return P, Minv, momentum, internalAndInertialForce
