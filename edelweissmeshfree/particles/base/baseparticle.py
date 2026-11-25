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
"""
Together with meshfree kernel functions (:class:`~BaseMeshfreeKernelFunction`), particles represent the basic building blocks of a particle based method. They
are used for computing the quadrature.

Implementing your own particles can be done easily by subclassing from
the abstract base class :class:`~BaseParticle`.

In an explicit analysis, the methods of a particle are called in an order its implementation relies on (advance,
accept twice, reconnect, then the force and the momentum, the lumped inertia and damping); the calls and the contracts
that follow from their order are written down once, in Marmot's documentation: `the explicit life cycle of a particle
<https://materialmodelingtoolbox.github.io/Marmot/pages/features/meshfreeparticles.html#explicit-life-cycle-of-a-particle>`_.
"""

from abc import abstractmethod

import numpy as np
from edelweissfe.nodecouplingentity.base.nodecouplingentity import (
    BaseNodeCouplingEntity,
)

from edelweissmeshfree.meshfree.kernelfunctions.base.basemeshfreekernelfunction import (
    BaseMeshfreeKernelFunction,
)


class BaseParticle(BaseNodeCouplingEntity):
    """The BaseParticle class is an abstract base class for all particles.

    If you want to implement a new particle, you have to inherit from this class.

    Particles communicate with field variables (:class:`~FieldVariable`) through meshfree kernel functions (:class:`~BaseMeshfreeKernelFunction`),
    which are attached to nodes.
    During a simulation, particles are used to compute the quadrature of the weak form of the governing equations.
    Accordingly, like elements and cells, they are responsible for computing the residual and stiffness matrix.

    Naturally, and in contrast to elements, cells and cell elements, particles have an identical set of fields on each attached node.
    Furthermore, due to the varying number attached nodes, no permutation is allowed.
    For particles generally a node-wise layout is assumed.

    For instance: [node_1_displacement, node_1_temperature, node_2_displacement, node_2_temperature, ...]
    """

    #: The kernel functions of a particle reach other nodes as it moves, so its connectivity changes with the motion;
    #: see :attr:`~edelweissfe.nodecouplingentity.base.nodecouplingentity.BaseNodeCouplingEntity.connectivityChangesWithMotion`.
    connectivityChangesWithMotion = True

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Defines the dimension of the particle, e.g., 1, 2 or 3."""

    @property
    @abstractmethod
    def ensightType(self) -> str:
        """The shape of the particle in Ensight Gold notation."""

    @property
    @abstractmethod
    def baseFields(self) -> list[str]:
        """Defines which fields are attached to the particle."""

    @property
    @abstractmethod
    def propertyNames(self) -> list[str]:
        """Defines which properties are valid for the particle."""

    @abstractmethod
    def setProperty(self, propertyName: str, propertyValue: np.ndarray):
        """Set a single property of the particle.
        Parameters
        ----------
        propertyName
            The name of the property to be set.
        propertyValue
            The value of the property to be set.
        """

    @abstractmethod
    def setProperties(self, properties: np.ndarray):
        """Set all properties of the particle.
        The order of the properties is defined by the order of the
        :attr:`propertyNames` property.

        Parameters
        ----------
        properties
            The properties to be set.
        """

    @property
    @abstractmethod
    def kernelFunctions(self) -> list[BaseMeshfreeKernelFunction]:
        """The kernel functions assigned to the particle."""

    @abstractmethod
    def getVertexCoordinates(
        self,
    ) -> np.ndarray:
        """The vertices defining the shape of the particle.

        Returns
        -------
        np.ndarray
            All coordinates for all bounding vertices."""

    @abstractmethod
    def getFaceCoordinates(
        self,
        faceID: int,
    ) -> np.ndarray:
        """The vertices defining the shape of the particle.
        Parameters
        ----------
        faceID
            The ID of the face to be returned.

        Returns
        -------
        np.ndarray
            All coordinates for all bounding vertices."""

    @abstractmethod
    def getEvaluationCoordinates(
        self,
    ) -> np.ndarray:
        """Get the evaluation coordinates of the particle.

        Returns
        -------
        np.ndarray
            All coordinates for all bounding vertices."""

    @abstractmethod
    def getCenterCoordinates(
        self,
    ) -> np.ndarray:
        """Get the center coordinates of the particle.

        Returns
        -------
        np.ndarray
            The center coordinates."""

    @abstractmethod
    def getVolumeUndeformed(
        self,
    ) -> float:
        """Get the undeformed volume of the particle.

        Returns
        -------
        float
            The undeformed volume."""

    @abstractmethod
    def acceptStateAndPosition(
        self,
    ):
        """Accept the computed state (in nonlinear iteration schemes) and the position."""

    @abstractmethod
    def getResultArray(self, result: str, getPersistentView: bool = True, qp: int = 0) -> np.ndarray:
        """Get the array of a result, possibly as a persistent view which is continiously
        updated by a MaterialPoint.

        Parameters
        ----------
        result
            The name of the result.
        getPersistentView
            If true, the returned array should be continiously updated by the element.
        qp
            The quadrature point index. This is only relevant for higher order particles.

        Returns
        -------
        np.ndarray
            The result.
        """

    @abstractmethod
    def initializeYourself(
        self,
    ):
        """Initalize the particle to be ready for computing."""

    @abstractmethod
    def prepareYourself(self, timeTotal: float, dTime: float):
        """Prepare the particle to be ready for computing."""

    @abstractmethod
    def setInitialCondition(self, stateType: str, values: np.ndarray):
        """Assign initial conditions.

        Parameters
        ----------
        stateType
            The type of initial state.
        values
            The numpy array describing the initial state.
        """

    @abstractmethod
    def computeKernels(
        self,
        K: np.ndarray,
        P: np.ndarray,
        U: np.ndarray,
        dU: np.ndarray,
        timeTotal: float,
        dTime: float,
    ):
        """Add the internal forces of the particle to the residual, and their tangent: the signature of
        :meth:`edelweissfe.elements.base.baseelement.BaseElement.computeKernels`, so that a solver evaluates
        particles and elements alike.

        A particle is incremental on the current configuration: its kernel functions move with it, and their
        nodal values are increments, so a particle reads ``dU``, not ``U``.

        Parameters
        ----------
        K
            The tangent ``dP/dU`` to augment, the stiffness matrix.
        P
            The residual contribution to augment, the internal forces.
        U
            The current solution.
        dU
            The current solution increment.
        timeTotal
            The current total time.
        dTime
            The time increment.
        """

    @abstractmethod
    def computeBodyLoad(
        self,
        loadType: str,
        load: np.ndarray,
        P: np.ndarray,
        K: np.ndarray,
        timeTotal: float,
        dTime: float,
    ):
        """Add the residual contribution of a body load on the particle, the negative of the external load
        :math:`-\\boldsymbol{f}_\\mathrm{ext}`, and its tangent, which is zero for a dead load.

        Parameters
        ----------
        loadType
            The type of load to be computed (e.g., 'bodyforce')
        load
            The float (vector) describing the load.
        P
            The residual contribution to augment, :math:`-\\boldsymbol{f}_\\mathrm{ext}`.
        K
            The tangent ``dP/dU`` to augment.
        timeTotal
            The current total time.
        dTime
            The time increment.
        """

    @abstractmethod
    def computeDistributedLoad(
        self,
        loadType: str,
        surfaceID: int,
        load: np.ndarray,
        P: np.ndarray,
        K: np.ndarray,
        timeTotal: float,
        dTime: float,
    ):
        """Add the residual contribution of a distributed (surface) load on a surface of the particle, the negative of
        the external load :math:`-\\boldsymbol{f}_\\mathrm{ext}`, and its tangent, which is nonzero only for a load
        that follows the deformation.

        Parameters
        ----------
        loadType
            The type of load to be computed (e.g., 'pressure')
        surfaceID
            The ID describing the surface of the particle.
        load
            The float (vector) describing the load.
        P
            The residual contribution to augment, :math:`-\\boldsymbol{f}_\\mathrm{ext}`.
        K
            The tangent ``dP/dU`` to augment.
        timeTotal
            The current total time.
        dTime
            The time increment.
        """

    @abstractmethod
    def updatePhysicsExplicit(
        self,
        dU: np.ndarray,
        timeTotal: float,
        dTime: float,
    ):
        """Explicit dynamics: take the solution increment on the kernel functions currently assigned, i.e. update the
        state and the position of the particle, and its velocity and acceleration; see the explicit dynamic solver
        :class:`~edelweissfe.solvers.nonlinearexplicitdynamic.NED`. A particle that cannot take the increment requests
        a cutback (:class:`~edelweissfe.utils.exceptions.CutbackRequest`), which fails the step.

        Parameters
        ----------
        dU
            The solution increment.
        timeTotal
            The current total time.
        dTime
            The time increment.
        """

    @abstractmethod
    def computePhysicsKernelsExplicit(
        self,
        P: np.ndarray,
    ):
        """Explicit dynamics: add the internal force of the current state, on the kernel functions currently
        assigned, to the residual contribution; no tangent.

        Parameters
        ----------
        P
            The residual contribution to augment.
        """

    @abstractmethod
    def computeLumpedInertia(
        self,
        M: np.ndarray,
    ):
        """Explicit dynamics: add the lumped mass of the particle, on the kernel functions currently assigned: the mass
        the particle integrates itself, which a solver reads through
        :meth:`~edelweissmeshfree.models.mpmdiscretesystem.MPMDiscreteSystem.assembleLumpedMassOfEntitiesWithOwnInertia`.

        Parameters
        ----------
        M
            The lumped mass to augment, one entry per degree of freedom.
        """

    @abstractmethod
    def computeLumpedMomentum(
        self,
        momentum: np.ndarray,
    ):
        """Explicit dynamics: add the lumped momentum of the particle, on the kernel functions currently assigned,
        from which a solver projects the velocity of the nodes.

        Parameters
        ----------
        momentum
            The lumped momentum to augment, one entry per degree of freedom.
        """

    def computeLumpedDamping(
        self,
        C: np.ndarray,
    ):
        """Explicit dynamics: add the lumped damping of the particle, on the kernel functions currently assigned: the
        coefficient of each field's first time derivative, as :meth:`computeLumpedInertia` holds the one of the second.
        A gradient-enhanced particle reports the nonlocal viscosity there, which integrates a first-order nonlocal field
        on its own and damps a second-order one. The default adds nothing, which is no damping.

        Parameters
        ----------
        C
            The lumped damping to augment, one entry per degree of freedom.
        """

    @abstractmethod
    def assembleStiffness(
        self,
        K: np.ndarray,
    ):
        """Explicit dynamics: add the stiffness of the particle, the tangent of the internal force with respect to the
        degrees of freedom of the kernel functions currently assigned, without inertia, of a zero increment of the
        last accepted configuration and the working state: after an accept, the stiffness of the accepted state. It
        leaves the particle as it was found. The stiffness the bound of :meth:`addBoundOfAbsoluteStiffnessTimes`
        bounds, e.g. to check that bound against.

        The zero increment is taken over a unit time increment, at time zero: the time increment is only positive, as a
        material may divide by it, and the tangent of a rate-independent material does not depend on it. For a
        rate-dependent material, e.g. with a Duvaut-Lions viscosity, it is the tangent relaxed over that time, softer
        than the response within an explicit increment, and the bound taken from it may be too large.

        Parameters
        ----------
        K
            The stiffness to augment, column-major, ``nDof x nDof``.
        """

    @abstractmethod
    def addBoundOfAbsoluteStiffnessTimes(
        self,
        W: np.ndarray,
        bound: np.ndarray,
    ):
        """Explicit dynamics: add an upper bound of :math:`|\\boldsymbol{K}|\\, \\boldsymbol{W}`, the absolute value of
        the stiffness of :meth:`assembleStiffness` (see there for the state it is of), times nonnegative weights. It
        leaves the particle as it was found. The stiffness itself is not assembled: the bound costs as much as the
        nodes, the stiffness as their square. It is what the critical time step of the particles is estimated with,
        after the accept of an increment
        (:meth:`~edelweissmeshfree.models.mpmdiscretesystem.MPMDiscreteSystem.criticalTimeStepOfEntitiesWithOwnInertia`).

        Parameters
        ----------
        W
            The nonnegative weights, ``nDof x nColumns``, column-major (Fortran order).
        bound
            The bound to augment, ``nDof x nColumns``, column-major (Fortran order).
        """

    @abstractmethod
    def assignKernelFunctions(self, kernelFunctions: list[BaseMeshfreeKernelFunction]):
        """Assign the meshfree kernel functions whose support covers the particle, and evaluate the shape functions
        with them. In a simulation, only the particle manager of the particle calls it, for a particle whose kernel
        functions changed: the manager decides from its own record of the last search whether they changed.

        Parameters
        ----------
        kernelFunctions
            The kernel functions.
        """

    @abstractmethod
    def updateShapeFunctions(self):
        """Evaluate the shape functions anew at the particle's current position, with the kernel functions it has been
        assigned. The particle managers call it in every increment for the particles whose kernel functions did not
        change, because the particles have moved, and :meth:`assignKernelFunctions` for the others."""

    @abstractmethod
    def getInterpolationVector(self, coordinate: np.ndarray) -> np.ndarray:
        """The shape functions of the assigned kernel functions, evaluated exactly at an arbitrary location,
        :math:`N_B(\\boldsymbol{x})`: e.g. where a field of the kernel functions is wanted somewhere in the body. Not the
        particle's own interpolation at its own points (its center, a vertex, a face center), which is
        :meth:`getInterpolationVectorLinearizedAboutCenter`; away from the center the two differ.

        Parameters
        ----------
        coordinate
            The location.

        Returns
        -------
        np.ndarray
            The interpolation vector for all nodes.
        """

    @abstractmethod
    def getInterpolationVectorLinearizedAboutCenter(self, coordinate: np.ndarray) -> np.ndarray:
        r"""The shape functions linearized about the center of the particle,
        :math:`G_B(\boldsymbol{x}) = N_B(\boldsymbol{x}_c) + \nabla N_B(\boldsymbol{x}_c) \cdot (\boldsymbol{x} -
        \boldsymbol{x}_c)`: the particle's own interpolation at its own points.

        A particle knows every field it carries only by its value and its (smoothed) gradient at its center, so at a
        point of its own (its center, a vertex, a face center) its interpolation of any field is this linearization:
        for the displacement, the motion of the point in the next increment,
        :math:`\Delta\boldsymbol{x} = \sum_B G_B \Delta\boldsymbol{U}_B` (the particle moves as a whole, its center
        with the shape functions there and its domain with the homogeneous deformation gradient); for a nonlocal field
        or a temperature, the value the particle carries at the point. Away from the center it is not
        :meth:`getInterpolationVector` at the point, which is exact at an arbitrary location. A constraint that
        measures a field at a point of the particle and distributes its force there with this vector is symmetric and
        conservative. The vector sums to one.

        Parameters
        ----------
        coordinate
            The point, in the current configuration of the particle.

        Returns
        -------
        np.ndarray
            The interpolation vector for all nodes.
        """

    @abstractmethod
    def vci_getNumberOfConstraints(
        self,
    ) -> int:
        """Get the number of constraints if a variationally consistent integration is used.

        Returns
        -------
        int
            The number of VCI constraints.
        """

    @abstractmethod
    def vci_compute_Test_P_BoundaryIntegral(
        self, R_AiC: np.ndarray, boundarySurfaceVector: np.ndarray, boundaryFaceID: int
    ):
        """Compute the boundary integral for the test function.

        Parameters
        ----------
        R_AiC
            The integral.
        boundarySurfaceVector
            The surface vector defining the area and orientation of the boundary.
            For higher order (non point) particles, the surface vector might be omitted and computed internally from the boundary face ID.
        boundaryFaceID
            The ID of the boundary face (for point shaped particles, this one is ignored).
        """

    @abstractmethod
    def vci_compute_TestGradient_P_Integral(self, R_AiC: np.ndarray):
        """Compute the volume integral for the test function gradient.

        Parameters
        ----------
        R_AiC
            The integral.
        """

    @abstractmethod
    def vci_compute_Test_PGradient_Integral(self, R_AiC: np.ndarray):
        """Compute the volume integral for the test function gradient.

        Parameters
        ----------
        R_AiC
            The integral.
        """

    @abstractmethod
    def vci_compute_MMatrix(self, M_ACD: np.ndarray):
        """Compute the symmetric M-matrix.

        Parameters
        ----------
        M_ACD
            The M-matrix
        """

    @abstractmethod
    def vci_assignTestFunctionCorrectionTerms(self, eta_AjC: np.ndarray):
        """Assign the shape function correction terms in
        for of a 3D numpy array.

        Parameters
        ----------
        corrections
            The correction terms.
        """

    @abstractmethod
    def getRestartData(self) -> np.ndarray:
        """The state of the last accepted increment, for a restart checkpoint. A particle is updated-Lagrangian: its
        position and shape follow from this state.

        Returns
        -------
        np.ndarray
            The state.
        """

    @abstractmethod
    def readRestartData(self, restartData: np.ndarray):
        """Restore the state :meth:`getRestartData` returned, and with it the position and the shape.

        Parameters
        ----------
        restartData
            The state.
        """
