# cython: freethreading_compatible = True
#Initalize  -*- coding: utf-8 -*-
#  ---------------------------------------------------------------------
#
#  _____    _      _              _         __  __ ____  __  __
# | ____|__| | ___| |_      _____(_)___ ___|  \/  |  _ \|  \/  |
# |  _| / _` |/ _ \ \ \ /\ / / _ \ / __/ __| |\/| | |_) | |\/| |
# | |__| (_| |  __/ |\ V  V /  __/ \__ \__ \ |  | |  __/| |  | |
# |_____\__,_|\___|_| \_/\_/ \___|_|___/___/_|  |_|_|   |_|  |_|
#
#
#  Unit of Strength of Materials and Structural Analysis
#  University of Innsbruck,
#  2023 - today
#
#  Matthias Neuner matthias.neuner@uibk.ac.at
#
#  This file is part of EdelweissMPM.
#
#  This library is free software; you can redistribute it and/or
#  modify it under the terms of the GNU Lesser General Public
#  License as published by the Free Software Foundation; either
#  version 2.1 of the License, or (at your option) any later version.
#
#  The full text of the license can be found in the file LICENSE.md at
#  the top level directory of EdelweissMPM.
#  ---------------------------------------------------------------------

import numpy as np

cimport cython
cimport libcpp.cast
cimport numpy as np

cimport edelweissmeshfree.particles.marmot.marmotparticlewrapper

from edelweissfe.numerics.systemmatrixcontributor import denseBlockLocalEntries
from edelweissfe.utils.bufferlength cimport requireLength
from edelweissfe.utils.exceptions import CutbackRequest, ModuleNotBuilt

from cython.operator cimport dereference
from libc.stdlib cimport free, malloc
from libcpp.memory cimport allocator, make_unique, unique_ptr

from edelweissmeshfree.materialpoints.marmotmaterialpoint.mp cimport (
    MarmotMaterialPoint,
    MarmotMaterialPointWrapper,
)
from edelweissmeshfree.meshfree.approximations.marmot.marmotmeshfreeapproximation cimport (
    MarmotMeshfreeApproximation,
    MarmotMeshfreeApproximationWrapper,
)
from edelweissmeshfree.meshfree.kernelfunctions.marmot.marmotmeshfreekernelfunction cimport (
    MarmotMeshfreeKernelFunction,
    MarmotMeshfreeKernelFunctionWrapper,
)


@cython.final # no subclassing -> cpdef with nogil possible
cdef class MarmotParticleWrapper:
    """This class as a wrapper for MarmotParticles.

    For the documentation of MarmotParticles, please refer to `Marmot <https://github.com/MAteRialMOdelingToolbox/Marmot/>`_.
    """


    def __cinit__(self,
                  str particleType,
                  int particleNumber,
                  double[:,::1] vertexCoordinates,
                  double volume,
                  MarmotMeshfreeApproximationWrapper marmotMeshfreeApproximationWrapper,
                  dict material
                  ):


        ##TODO: This this crap:
        # mpCenter =  np.mean( vertexCoordinates, axis=0).reshape(-1, vertexCoordinates.shape[1])

        # self._mp = MarmotMaterialPointWrapper("GradientEnhancedMicropolar/PlaneStrain", particleNumber, mpCenter , volume, material)

        self._vertexCoordinates = np.copy(vertexCoordinates)
        self._vertexCoordinatesView = self._vertexCoordinates





        self._centerCoordinates = np.zeros(vertexCoordinates.shape[1])
        self._centerCoordinatesView = self._centerCoordinates

        # self._marmotMaterialPoint = <MarmotMaterialPoint*> self._mp._marmotMaterialPoint
        # the Marmot particle refers to the approximation, which the wrapper owns: keep the wrapper alive with the particle
        self._approximation = marmotMeshfreeApproximationWrapper

        self._assignedKernelFunctions = list()

        cdef str materialName = material['material']
        self.materialProperties = material['properties']
        self.materialPropertiesView = self.materialProperties
        cdef int nMaterialProperties = len(self.materialProperties)

        # a particle that is not built may be skipped (a test is); a particle that fails to be created is an error
        if not MarmotParticleFactory.isRegistered(particleType.encode('utf-8')):
            raise ModuleNotBuilt("Marmot particle {:} is not built in this environment".format(particleType))
        self._marmotParticle = MarmotParticleFactory.createParticle(particleType.encode('utf-8'),
                                                                   particleNumber,
                                                                   &self._vertexCoordinatesView[0,0],
                                                                   self._vertexCoordinates.size,
                                                                   volume,
                                                                   materialName.upper().encode('utf-8'),
                                                                   &self.materialPropertiesView[0],
                                                                   nMaterialProperties,
                                                                   self._approximation._marmotMeshfreeApproximation[0],
                                                                   )

        self._nStateVars =           self._marmotParticle.getNumberOfRequiredStateVars()

        self._stateVars =            np.zeros(self._nStateVars)
        self._stateVarsTemp =        np.zeros(self._nStateVars)

        self._marmotParticle.assignStateVars(&self._stateVarsTemp[0], self._nStateVars)

        self._particleType          = particleType
        self._number                = particleNumber
        self._ensightType           = self._marmotParticle.getParticleShape().decode('utf-8')
        self._nVertices             = self._marmotParticle.getNumberOfVertices()
        self._nDim                  = self._marmotParticle.getDimension()
        self._nBaseDof              = self._marmotParticle.getNBaseDof()
        # self._assignedKernelFunctions = list()

        cdef vector[string] fields           = self._marmotParticle.getFields()
        self._baseFields                         = [  s.decode('utf-8')  for s in fields ]

        cdef dict supportedBodyLoads = self._marmotParticle.getSupportedBodyLoadTypes()
        self._supportedBodyLoads = {k.decode() :  v for k, v in supportedBodyLoads.items()  }

        cdef dict supportedDistributedLoads = self._marmotParticle.getSupportedDistributedLoadTypes()
        self._supportedDistributedLoads = {k.decode() :  v for k, v in supportedDistributedLoads.items()  }

        self._nEvaluationPoints = self._marmotParticle.getNumberOfEvaluationPoints()

        self._evaluationCoordinates = np.zeros( ( self._nEvaluationPoints, self._nDim) )
        self._evaluationCoordinatesView = self._evaluationCoordinates

    @property
    def nodes(self):
        return self._nodes

    @property
    def nNodes(self):
        return len(self._nodes)

    @property
    def dimension(self):
        return self._nDim

    @property
    def baseFields(self):
        return self._baseFields

    @property
    def fields(self):
        return self._fields

    @property
    def propertyNames(self) -> list[str]:
        """Get the list of valid properties for this particle, as strings (the names :meth:`setProperty` takes)."""

        return [name.decode("utf-8") for name in self._marmotParticle.getPropertyNames()]

    def setProperty(self, str propertyName, value):
        """Set a property of the particle."""

        cdef double[::1] _value = np.array(value, dtype=np.float64).reshape((1,))

        self._marmotParticle.setProperty(propertyName.encode('utf-8'), &_value[0])

    def setProperties(self, double[::1] properties):
        """Set a list of properties of the particle."""
        self._marmotParticle.setProperties(&properties[0], len(properties))


    @property
    def nDof(self):
        return self._numberOfDofs()

    cdef Py_ssize_t _numberOfDofs(self):
        """The dofs of the particle: those of a node times its kernel functions; changes as they are reassigned."""
        return self._nBaseDof * self._nAssignedKernelFunctions

    @property
    def ensightType(self):
        return self._ensightType

    @property
    def number(self):
        return self._number

    @property
    def kernelFunctions(self):
        return self._assignedKernelFunctions

    @property
    def dofIndicesPermutation(self):
        return None

    def computeKernels(self,
                       double[::1] Kc,
                       double[::1] Pc,
                       const double[::1] Uc,
                       const double[::1] dUc,
                       double timeNew,
                       double dTime):
        """Add the internal forces of the particle to the residual, and their tangent; see
        :meth:`edelweissmeshfree.particles.base.baseparticle.BaseParticle.computeKernels`. A particle is incremental,
        so Marmot gets the solution increment. A failure in Marmot, e.g. a material that cannot integrate the
        increment, requests a cutback, as for an EdelweissFE element.

        ``Kc`` has ``nDof * nDof`` entries, ``Pc`` and ``dUc`` have ``nDof``; ``nDof`` changes with every
        reassignment of the kernel functions. A buffer of another length raises a ``ValueError``."""

        cdef Py_ssize_t nDof = self._numberOfDofs()
        requireLength("computeKernels", "Kc", Kc.shape[0], nDof * nDof)
        requireLength("computeKernels", "Pc", Pc.shape[0], nDof)
        requireLength("computeKernels", "dUc", dUc.shape[0], nDof)

        self._initializeStateVarsTemp()
        try:
            with nogil:
                self._marmotParticle.computePhysicsKernels(&dUc[0], &Pc[0], &Kc[0], timeNew, dTime)
        except (RuntimeError, ValueError) as e:
            raise CutbackRequest(str(e), 0.5)


    def updatePhysicsExplicit(self,
                              double[::1] dUc,
                              double timeNew,
                              double dTime, ):
        """Take the solution increment; see
        :meth:`edelweissmeshfree.particles.base.baseparticle.BaseParticle.updatePhysicsExplicit`. A failure in Marmot
        requests a cutback, as in :meth:`computeKernels`. ``dUc`` has ``nDof`` entries."""

        requireLength("updatePhysicsExplicit", "dUc", dUc.shape[0], self._numberOfDofs())

        self._initializeStateVarsTemp()
        try:
            self._marmotParticle.updatePhysicsExplicit(&dUc[0], timeNew, dTime)
        except (RuntimeError, ValueError) as e:
            raise CutbackRequest(str(e), 0.5)

    def computePhysicsKernelsExplicit(self,
                                      double[::1] Pc ):
        """Add the internal force of the current state; see
        :meth:`edelweissmeshfree.particles.base.baseparticle.BaseParticle.computePhysicsKernelsExplicit`. ``Pc`` has
        ``nDof`` entries."""

        requireLength("computePhysicsKernelsExplicit", "Pc", Pc.shape[0], self._numberOfDofs())
        self._marmotParticle.computePhysicsKernelsExplicit(&Pc[0])

    def computeLumpedInertia(self, double[::1] mLumped ):
        """Add the lumped mass; see
        :meth:`edelweissmeshfree.particles.base.baseparticle.BaseParticle.computeLumpedInertia`. ``mLumped`` has
        ``nDof`` entries."""

        requireLength("computeLumpedInertia", "mLumped", mLumped.shape[0], self._numberOfDofs())
        self._marmotParticle.computeLumpedInertia(&mLumped[0])

    @property
    def connectivityChangesWithMotion(self):
        """True: the kernel functions of a particle reach other nodes as it moves."""
        return True

    @property
    def scalarVariables(self):
        """No scalar variables, as for every particle; see
        :attr:`edelweissfe.nodecouplingentity.base.nodecouplingentity.BaseNodeCouplingEntity.scalarVariables`."""
        return ()

    def computeLumpedMomentum(self, double[::1] pLumped ):
        """Add the lumped momentum; see
        :meth:`edelweissmeshfree.particles.base.baseparticle.BaseParticle.computeLumpedMomentum`. ``pLumped`` has
        ``nDof`` entries."""

        requireLength("computeLumpedMomentum", "pLumped", pLumped.shape[0], self._numberOfDofs())
        self._marmotParticle.computeLumpedMomentum(&pLumped[0])

    def computeLumpedDamping(self, double[::1] cLumped ):
        """Add the lumped damping; see
        :meth:`edelweissmeshfree.particles.base.baseparticle.BaseParticle.computeLumpedDamping`. ``cLumped`` has
        ``nDof`` entries."""

        requireLength("computeLumpedDamping", "cLumped", cLumped.shape[0], self._numberOfDofs())
        self._marmotParticle.computeLumpedDamping(&cLumped[0])

    def assembleStiffness(self, double[::1] K):
        """Add the stiffness; see
        :meth:`edelweissmeshfree.particles.base.baseparticle.BaseParticle.assembleStiffness`: Marmot's assembling
        visitor of the stiffness the particle describes, which leaves the particle as it was found; over a unit time
        increment at time zero. ``K`` has ``nDof * nDof`` entries."""

        cdef Py_ssize_t nDof = self._numberOfDofs()
        requireLength("assembleStiffness", "K", K.shape[0], nDof * nDof)
        assembleStiffness(dereference(self._marmotParticle), &K[0], 0.0, 1.0)

    def addBoundOfAbsoluteStiffnessTimes(self, double[::1, :] W, double[::1, :] bound):
        """Add an upper bound of the absolute value of the stiffness times the weights; see
        :meth:`edelweissmeshfree.particles.base.baseparticle.BaseParticle.addBoundOfAbsoluteStiffnessTimes`: Marmot's
        visitor of the stiffness the particle describes, which leaves the particle as it was found; over a unit time
        increment at time zero, as :meth:`assembleStiffness`. ``W`` and ``bound`` both have ``nDof`` rows and the
        same number of columns."""

        cdef Py_ssize_t nDof = self._numberOfDofs()
        requireLength("addBoundOfAbsoluteStiffnessTimes", "W rows", W.shape[0], nDof)
        requireLength("addBoundOfAbsoluteStiffnessTimes", "bound rows", bound.shape[0], nDof)
        requireLength("addBoundOfAbsoluteStiffnessTimes", "bound columns", bound.shape[1], W.shape[1])
        addBoundOfAbsoluteStiffnessTimes(
            dereference(self._marmotParticle), &W[0, 0], W.shape[1], &bound[0, 0], 0.0, 1.0
        )

    cpdef void _initializeStateVarsTemp(self, ) nogil:
        self._stateVarsTemp[:] = self._stateVars

    def computeBodyLoad(self,
                        str loadType,
                        double[::1] load,
                        double[::1] Pc,
                        double[::1] Kc,
                        double timeNew,
                        double dTime):
        """Add a body load; see :meth:`edelweissmeshfree.particles.base.baseparticle.BaseParticle.computeBodyLoad`.
        ``load`` has ``dimension`` entries, ``Pc`` has ``nDof`` and ``Kc`` has ``nDof * nDof``."""

        cdef Py_ssize_t nDof = self._numberOfDofs()
        requireLength("computeBodyLoad", "load", load.shape[0], self._nDim)
        requireLength("computeBodyLoad", "Pc", Pc.shape[0], nDof)
        requireLength("computeBodyLoad", "Kc", Kc.shape[0], nDof * nDof)

        self._marmotParticle.computeBodyLoad( self._supportedBodyLoads[loadType.upper()], &load[0], &Pc[0], &Kc[0], timeNew, dTime)

    def computeDistributedLoad(self,
                               str loadType,
                               int surfaceID,
                               double[::1] load,
                               double[::1] Pc,
                               double[::1] Kc,
                               double timeNew,
                               double dTime):
        """Add a distributed load; see
        :meth:`edelweissmeshfree.particles.base.baseparticle.BaseParticle.computeDistributedLoad`. ``Pc`` has
        ``nDof`` entries and ``Kc`` has ``nDof * nDof``. How many values ``load`` holds depends on the load type,
        which only the Marmot particle knows, so it is not checked here."""

        cdef Py_ssize_t nDof = self._numberOfDofs()
        requireLength("computeDistributedLoad", "Pc", Pc.shape[0], nDof)
        requireLength("computeDistributedLoad", "Kc", Kc.shape[0], nDof * nDof)

        self._marmotParticle.computeDistributedLoad( self._supportedDistributedLoads[loadType.upper()], surfaceID, &load[0], &Pc[0], &Kc[0], timeNew, dTime)

    def getInterpolationVector(self, double[::1] coordinates) -> np.ndarray:
        """The shape functions of the nodes at ``coordinates``, which has ``dimension`` entries."""

        requireLength("getInterpolationVector", "coordinates", coordinates.shape[0], self._nDim)
        cdef np.ndarray N = np.zeros(len(self._nodes))
        cdef double[::1] Nview_ = N
        self._marmotParticle.getInterpolationVector(&Nview_[0], &coordinates[0])
        return N

    def getInterpolationVectorLinearizedAboutCenter(self, double[::1] coordinates) -> np.ndarray:
        """The shape functions linearized about the center; see
        :meth:`edelweissmeshfree.particles.base.baseparticle.BaseParticle.getInterpolationVectorLinearizedAboutCenter`.
        ``coordinates`` has ``dimension`` entries."""

        requireLength("getInterpolationVectorLinearizedAboutCenter", "coordinates", coordinates.shape[0], self._nDim)
        cdef np.ndarray G = np.zeros(len(self._nodes))
        cdef double[::1] Gview_ = G
        self._marmotParticle.getInterpolationVectorLinearizedAboutCenter(&Gview_[0], &coordinates[0])
        return G

    def assignKernelFunctions(self, list marmotMeshfreeKernelFunctionWrappers):
        self._assignedKernelFunctions = marmotMeshfreeKernelFunctionWrappers
        self._nAssignedKernelFunctions = len(marmotMeshfreeKernelFunctionWrappers)

        cdef vector[const MarmotMeshfreeKernelFunction*] kernelFunctions

        cdef MarmotMeshfreeKernelFunctionWrapper k
        for k in marmotMeshfreeKernelFunctionWrappers:
            kernelFunctions.push_back(k._marmotMeshfreeKernelFunction)

        self._marmotParticle.assignMeshfreeKernelFunctions(kernelFunctions)

        self._nodes = [kf.node for kf in self._assignedKernelFunctions]
        self._fields = [ self._baseFields for n in self._nodes ]

    def updateShapeFunctions(self):
        """Evaluate the shape functions anew at the particle's current position, with the kernel functions it has.

        Marmot holds the kernel functions, so no Python object is touched: the particles of several threads do not
        contend for the reference counts of the kernel functions they share.
        """
        with nogil:
            self._marmotParticle.updateShapeFunctions()

    def prepareYourself(self, timeTotal: float, dTime: float):
        self._stateVarsTemp[:] = self._stateVars
        self._marmotParticle.acceptStateAndPosition() # this is to set the state of the particle to "ready"
        self._stateVarsTemp[:] = self._stateVars

    def acceptStateAndPosition(self,):
        self._marmotParticle.acceptStateAndPosition() # this is to set the state of the particle to "ready"
        self._stateVars[:] = self._stateVarsTemp

    def initializeYourself(self):
        self._stateVarsTemp[:] = self._stateVars
        self._marmotParticle.initializeYourself()
        self.acceptStateAndPosition()

    def getResultArray(self, result:str, getPersistentView:bool=True, qp:int=0):
        """Get the array of a result, possibly as a persistent view which is continiously
        updated by the underlying MarmotParticle."""

        cdef string result_ =  result.encode('UTF-8')
        return np.array(  self.getStateView(result_, qp), copy= not getPersistentView, )

    cdef double[::1] getStateView(self, string result, int qp):
        """Directly access the state vars of the underlying MarmotElement"""

        cdef StateView res = self._marmotParticle.getStateView(result, qp)

        return <double[:res.stateSize]> ( res.stateLocation )

    def getVertexCoordinates(self):
        """Get the underlying MarmotParticle vertex coordinates."""

        self._marmotParticle.getVertexCoordinates(&self._vertexCoordinatesView[0,0])

        return self._vertexCoordinates

    def getFaceCoordinates(self, int faceID):
        """Get the underlying coordinates of a given face of the MarmotParticle."""

        cdef np.ndarray faceCoordinates = np.zeros( self._nDim )
        cdef double[::1] faceCoordinatesView = faceCoordinates
        self._marmotParticle.getFaceCoordinates(faceID, &faceCoordinatesView[0])

        return faceCoordinates

    def getEvaluationCoordinates(self):

        self._marmotParticle.getEvaluationCoordinates(&self._evaluationCoordinatesView[0,0])

        return self._evaluationCoordinates

    def getCenterCoordinates(self):
        """Get the underlying MarmotParticle center coordinates."""

        self._marmotParticle.getCenterCoordinates(&self._centerCoordinatesView[0])

        return self._centerCoordinates

    def getVolumeUndeformed(self):
        """Get the underlying MarmotParticle undeformed volume."""

        return self._marmotParticle.getVolumeUndeformed()

    def setInitialCondition(self, str stateType, value):
        """Set initial condition of the particle."""

        cdef double[::1] _value = np.array(value, dtype=np.float64).reshape((1,))

        self._marmotParticle.setInitialCondition(stateType.encode('utf-8'), &_value[0])
        self.acceptStateAndPosition()

    def vci_getNumberOfConstraints(self, ):
        return self._marmotParticle.vci_getNumberOfConstraints()

    cdef int _requireVCIShape(self, str method, str name, double[:,:,::1] array, Py_ssize_t nRows) except -1:
        """Require the shape of a VCI array: one block per node, of ``nRows`` x the number of VCI constraints."""

        requireLength(method, name + " nodes", array.shape[0], self._nAssignedKernelFunctions)
        requireLength(method, name + " rows", array.shape[1], nRows)
        requireLength(method, name + " constraints", array.shape[2], self._marmotParticle.vci_getNumberOfConstraints())
        return 0

    def vci_compute_Test_P_BoundaryIntegral(self, double[:,:,::1] r_AiC, double[::1] boundarySurfaceVector, int boundaryFaceID ):
        """Accumulate the boundary integral of a face into ``r_AiC`` (nodes x dimension x constraints);
        ``boundarySurfaceVector`` has ``dimension`` entries."""

        self._requireVCIShape("vci_compute_Test_P_BoundaryIntegral", "r_AiC", r_AiC, self._nDim)
        requireLength("vci_compute_Test_P_BoundaryIntegral", "boundarySurfaceVector", boundarySurfaceVector.shape[0], self._nDim)
        self._marmotParticle.vci_compute_Test_P_BoundaryIntegral(&r_AiC[0,0,0], &boundarySurfaceVector[0], boundaryFaceID )

    def vci_compute_TestGradient_P_Integral(self, double[:,:,::1]r_AiC):
        """Accumulate the volume integral of the test function gradients into ``r_AiC`` (nodes x dimension x
        constraints)."""

        self._requireVCIShape("vci_compute_TestGradient_P_Integral", "r_AiC", r_AiC, self._nDim)
        self._marmotParticle.vci_compute_TestGradient_P_Integral(&r_AiC[0,0,0])

    def vci_compute_Test_PGradient_Integral(self, double[:,:,::1] r_AiC):
        """Accumulate the volume integral of the basis gradients into ``r_AiC`` (nodes x dimension x constraints)."""

        self._requireVCIShape("vci_compute_Test_PGradient_Integral", "r_AiC", r_AiC, self._nDim)
        self._marmotParticle.vci_compute_Test_PGradient_Integral(&r_AiC[0,0,0])

    def vci_compute_MMatrix(self, double[:,:,::1] m_ACD):
        """Accumulate the moment matrix into ``m_ACD`` (nodes x constraints x constraints)."""

        self._requireVCIShape("vci_compute_MMatrix", "m_ACD", m_ACD, self._marmotParticle.vci_getNumberOfConstraints())
        self._marmotParticle.vci_compute_MMatrix(&m_ACD[0,0,0])

    def vci_assignTestFunctionCorrectionTerms(self, double[:,:,::1] eta_AjC):
        """Assign the correction terms ``eta_AjC`` (nodes x dimension x constraints)."""

        self._requireVCIShape("vci_assignTestFunctionCorrectionTerms", "eta_AjC", eta_AjC, self._nDim)
        self._marmotParticle.vci_assignTestFunctionCorrectionTerms(&eta_AjC[0,0,0])

    def getRestartData(self,):
        return self._stateVars

    def readRestartData(self, data):
        """Restore the state :meth:`getRestartData` returned. The geometry of the underlying MarmotParticle follows
        from its state, and is brought to it as at the start of an increment (see :meth:`prepareYourself`)."""

        np.asarray(self._stateVars)[:] = data[:]
        self.prepareYourself(0.0, 0.0)

    @property
    def couplesAllItsDofs(self) -> bool:
        """``True``: the block is the dense ``nDof x nDof`` matrix, column-major."""
        return True

    def localEntriesOfBlock(self) -> tuple[np.ndarray, np.ndarray]:
        """The entries of the dense block, see
        :func:`~edelweissfe.numerics.systemmatrixcontributor.denseBlockLocalEntries`."""
        return denseBlockLocalEntries(self.nDof)

    def shapeBlock(self, values: np.ndarray) -> np.ndarray:
        """Keep the view flat for Cython 1-D memoryview compatibility."""
        return values

    def __dealloc__(self):
        del self._marmotParticle

