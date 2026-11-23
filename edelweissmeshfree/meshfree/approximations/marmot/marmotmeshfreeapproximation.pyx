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

import numpy as np

cimport cython
cimport libcpp.cast
cimport numpy as np

cimport edelweissmeshfree.meshfree.approximations.marmot.marmotmeshfreeapproximation
from edelweissmeshfree.meshfree.kernelfunctions.marmot.marmotmeshfreekernelfunction cimport (
    MarmotMeshfreeKernelFunction,
    MarmotMeshfreeKernelFunctionWrapper,
)


@cython.final # no subclassing -> cpdef with nogil possible
cdef class MarmotMeshfreeApproximationWrapper:
    """A meshfree approximation of Marmot: the reproducing kernel approximation with explicit gradients
    (``"ReproducingKernel"``) or with implicit gradients (``"ReproducingKernelImplicitGradient"``).

    Parameters
    ----------
    approximationType
        ``"ReproducingKernel"`` or ``"ReproducingKernelImplicitGradient"``.
    dim
        The spatial dimension.
    completenessOrder
        The order of the polynomials the shape functions reproduce (keyword, required).
    momentMatrixConditionLimit
        The limit on the condition of the moment matrix, scaled to a unit diagonal, above which the completeness order
        of a point is lowered, for the linear basis to the order 0, the Shepard functions (keyword, optional). Such a
        point lies on one side of all kernels that cover it, which lie nearly in one plane: e.g. a point of a strongly stretched
        particle reaching into a void. There the linear shape functions extrapolate and grow without bound. The default
        ``inf`` sets no limit; with a limit, every point below it is evaluated bit for bit as without one. See Marmot's
        documentation of the meshfree approximation for the condition measure, what the lower order costs, and the
        range of useful limits.
    """

    def __cinit__(
        self,
        str approximationType,
        int dim,
        *,
        int completenessOrder,
        double momentMatrixConditionLimit=float("inf"),
    ):

        validApproximationTypes = ["ReproducingKernel", "ReproducingKernelImplicitGradient"]

        self._nDim = dim

        if approximationType == "ReproducingKernel":
                self._marmotMeshfreeApproximation = <MarmotMeshfreeApproximation*> (
                                                                                    new
                                                                                    MarmotMeshfreeReproducingKernelApproximation( dim, completenessOrder, momentMatrixConditionLimit ) )
        elif approximationType == "ReproducingKernelImplicitGradient":
                self._marmotMeshfreeApproximation = <MarmotMeshfreeApproximation*> (
                                                                                    new
                                                                                    MarmotMeshfreeReproducingKernelApproximationImplicit( dim, completenessOrder, momentMatrixConditionLimit ) )
        else:
                raise NotImplementedError("Approximation type {:} not found in library. Valid options are {:}".format(approximationType, validApproximationTypes))

    def computeShapeFunctions(self, double[::1] coordinates, list marmotMeshfreeKernelFunctionWrappers):
        cdef vector[const MarmotMeshfreeKernelFunction*] mMFKFs
        cdef MarmotMeshfreeKernelFunctionWrapper mMFKFWrapper

        for mMFKFWrapper in marmotMeshfreeKernelFunctionWrappers:
            mMFKFs.push_back(<MarmotMeshfreeKernelFunction*> mMFKFWrapper._marmotMeshfreeKernelFunction)

        cdef result = np.empty(len(marmotMeshfreeKernelFunctionWrappers), dtype=np.float64)
        cdef double[::1] result_view = result
        self._marmotMeshfreeApproximation.computeShapeFunctions(&coordinates[0], mMFKFs, &result_view[0])

        return result

    def computeShapeFunctionsAndGradients(self, double[::1] coordinates, list marmotMeshfreeKernelFunctionWrappers):
        cdef vector[const MarmotMeshfreeKernelFunction*] mMFKFs
        cdef MarmotMeshfreeKernelFunctionWrapper mMFKFWrapper

        for mMFKFWrapper in marmotMeshfreeKernelFunctionWrappers:
            mMFKFs.push_back(<MarmotMeshfreeKernelFunction*> mMFKFWrapper._marmotMeshfreeKernelFunction)

        cdef result = np.empty(len(marmotMeshfreeKernelFunctionWrappers), dtype=np.float64)
        cdef resultGradient = np.empty( (self._nDim, len(marmotMeshfreeKernelFunctionWrappers) ), dtype=np.float64, order='F')
        cdef double[::1] result_view = result
        cdef double[::1, :] resultGradient_view = resultGradient
        self._marmotMeshfreeApproximation.computeShapeFunctionsAndGradients(&coordinates[0], mMFKFs, &result_view[0], &resultGradient_view[0, 0])

        return result, resultGradient.T

    def __dealloc__(self):
        del self._marmotMeshfreeApproximation

