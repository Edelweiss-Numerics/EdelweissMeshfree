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
from libcpp.string cimport string
from libcpp.vector cimport vector

# from edelweissfe.points.node import Node

import numpy as np


cdef extern from "Marmot/MarmotMeshfreeKernelFunction.h" namespace "Marmot::Meshfree":
    cdef cppclass MarmotMeshfreeKernelFunction nogil:
        # MarmotMeshfreeKernelFunction(const double *coord, int dim, double supportRadius, int continuityOrder, int completeOrder)

        double computeKernelFunction( const double *coord)

        void getBoundingBox( double *min, double *max ) const

        int isInSupport( const double *coord ) const

        void moveTo ( const double *coordinates )

cdef extern from "Marmot/MarmotMeshfreeKernelFunctionBSpline2ndOrderBoxed.h" namespace "Marmot::Meshfree":
    cdef cppclass MarmotMeshfreeKernelFunctionBSpline2ndOrderBoxed nogil:
        MarmotMeshfreeKernelFunctionBSpline2ndOrderBoxed(double *coord, int dim, double supportRadius)

cdef extern from "Marmot/MarmotMeshfreeKernelFunctionBSpline3rdOrderBoxed.h" namespace "Marmot::Meshfree":
    cdef cppclass MarmotMeshfreeKernelFunctionBSpline3rdOrderBoxed nogil:
        MarmotMeshfreeKernelFunctionBSpline3rdOrderBoxed(double *coord, int dim, double supportRadius)


cdef class MarmotMeshfreeKernelFunctionWrapper:

    cdef _node
    cdef np.ndarray _center
    cdef np.ndarray _displacement
    cdef MarmotMeshfreeKernelFunction* _marmotMeshfreeKernelFunction

    cdef int _dimension
    cdef np.ndarray _boundingBoxMin
    cdef np.ndarray _boundingBoxMax



