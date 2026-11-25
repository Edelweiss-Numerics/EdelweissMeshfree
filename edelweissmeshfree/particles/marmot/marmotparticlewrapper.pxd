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
cimport numpy as np
from libcpp cimport bool
from libcpp.string cimport string
from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector

import numpy as np

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
)


cdef extern from "Marmot/MarmotUtils.h":
    cdef struct StateView "Marmot::Utilities::StateView":
        double *stateLocation
        int stateSize

cdef extern from "Marmot/MarmotParticleLibrary.h" nogil:
    cdef cppclass MarmotParticleFactory "Marmot::Registry::ParticleFactory":
        @staticmethod
        bool isRegistered(const string& particleName)

        @staticmethod
        MarmotParticle* createParticle(const string& particleName,
                               int particleNumber,
                               const double* particleCoordinates,
                               int sizeParticleCoordinates,
                               double volume,
                               const string& materialName,
                               const double* materialProperties,
                               int sizeMaterialProperties,
                               const MarmotMeshfreeApproximation& approximation) except +

# Every method of a Marmot particle may throw: a default that refuses (e.g. the explicit path of a particle that
# has none), or a check of its arguments. Each is declared "except +", which turns the C++ exception into a Python
# exception (RuntimeError, ValueError, ...); without it, the exception would end the process in std::terminate.
cdef extern from "Marmot/MarmotParticle.h":
    cdef cppclass MarmotParticle "Marmot::Discretizations::Meshfree::Particle" nogil:

        const vector[string]& getFields() except +

        void getVertexCoordinates(double* ) except +

        void getFaceCoordinates(int, double* ) except +

        void getEvaluationCoordinates(double* ) except +

        void getVisualizationVertexCoordinates(double* ) except +

        void getCenterCoordinates(double* ) except +

        string getParticleShape() except +

        void assignMeshfreeKernelFunctions (const vector[const MarmotMeshfreeKernelFunction*]& meshfreeKernelFunctions) except +

        void updateShapeFunctions() except +

        void computePhysicsKernels(   const double* dUc,
                                            double* Pc,
                                            double* Kc,
                                            double timeNewTotal,
                                            double dT) except +

        void computeBodyLoad(               int type,
                                            const double* load,
                                            double* Pc,
                                            double* Kc,
                                            double timeNewTotal,
                                            double dT) except +

        void computeDistributedLoad(        int type,
                                            int surfaceID,
                                            const double* load,
                                            double* Pc,
                                            double* Kc,
                                            double timeNewTotal,
                                            double dT) except +

        void updatePhysicsExplicit( const double* dUc,
                                            double timeNewTotal,
                                            double dT) except +

        void computePhysicsKernelsExplicit( double* Pc) except +

        void computeLumpedInertia ( double* mLumped) except +

        void computeLumpedMomentum ( double* pLumped) except +

        const unordered_map[string, int]& getSupportedBodyLoadTypes() except +

        const unordered_map[string, int]& getSupportedDistributedLoadTypes() except +

        void getInterpolationVector( double* N, const double* coordinates) except +

        void getInterpolationVectorLinearizedAboutCenter( double* G, const double* coordinates) except +

        int getNumberOfRequiredStateVars() except +

        void assignStateVars( double* stateVars, int nStateVars ) except +

        void initializeYourself() except +

        void acceptStateAndPosition() except +

        StateView getStateView( const string& stateName, int qp) except +

        int getDimension() except +

        int getNumberOfVertices() except +

        int getNBaseDof() except +

        int getNumberOfEvaluationPoints() except +

        int vci_getNumberOfConstraints() except +

        double getVolumeUndeformed() except +

        void vci_compute_Test_P_BoundaryIntegral(double* f_AiC_RowMajor, const double* boundarySurfaceVector, int boundaryFaceID) except +

        void vci_compute_TestGradient_P_Integral(double* f_AiC_RowMajor) except +

        void vci_compute_Test_PGradient_Integral(double* f_AiC_RowMajor) except +

        void vci_compute_MMatrix(double* MMatrix_ACD_RowMajor) except +

        void vci_assignTestFunctionCorrectionTerms(const double* eta_AjC_RowMajor) except +

        void setProperties( const double* properties, int nProperties ) except +

        void setProperty( const string& propertyName, const double* property ) except +

        void setInitialCondition( const string& stateName, const double* stateValue ) except +

        vector[ string ] getPropertyNames() except +

        void computeLumpedDamping ( double* cLumped) except +

# The host's visitors of the stiffness the particle describes, outside the particle: its assembly and the bound of
# its absolute value.
cdef extern from "Marmot/MarmotParticleStiffnessVisitors.h" namespace "Marmot::Discretizations::Meshfree" nogil:
    void assembleStiffness( MarmotParticle& particle, double* K, double timeNew, double dT) except +

    void addBoundOfAbsoluteStiffnessTimes( MarmotParticle& particle,
                                           const double* columnWeights,
                                           int nColumns,
                                           double* bound,
                                           double timeNew,
                                           double dT) except +





cdef class MarmotParticleWrapper:

    cdef MarmotParticle* _marmotParticle

    # the approximation the Marmot particle refers to, held so that it lives as long as the particle
    cdef MarmotMeshfreeApproximationWrapper _approximation

    cdef np.ndarray materialProperties
    cdef double[::1] materialPropertiesView


    cdef int _number,
    cdef str _particleType,
    cdef str _ensightType
    cdef int _nVertices

    cdef list _baseFields
    cdef list _fields

    cdef int _nBaseDof # the number of dofs we have for a single attached kernel function / node
    cdef int _nAssignedKernelFunctions # the number of attached kernel functions

    cdef public double[::1] _stateVars
    cdef public double[::1] _stateVarsTemp
    cdef public double[::1] _stateVarsOld

    cdef int _nStateVars
    cdef int _nDim

    cdef list[MarmotMeshfreeKernelFunctionWrapper] _assignedKernelFunctions
    cdef list _nodes

    cdef dict _supportedBodyLoads

    cdef dict _supportedDistributedLoads

    cdef public list _assignedShapeFunctions

    cdef public double[::1]  _materialProperties

    cdef np.ndarray _vertexCoordinates
    cdef double[:,::1] _vertexCoordinatesView

    cdef np.ndarray _evaluationCoordinates
    cdef double[:,::1] _evaluationCoordinatesView
    cdef int _nEvaluationPoints

    cdef np.ndarray _centerCoordinates
    cdef double[::1] _centerCoordinatesView

    # nogil methods are already declared here

    cpdef void _initializeStateVarsTemp(self, ) nogil

    cdef double[::1] getStateView(self, string stateName, int qp)

    cdef Py_ssize_t _numberOfDofs(self)

    cdef int _requireVCIShape(self, str method, str name, double[:,:,::1] array, Py_ssize_t nRows) except -1
