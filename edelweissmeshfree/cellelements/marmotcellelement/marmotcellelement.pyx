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

cimport edelweissmeshfree.cellelements.marmotcellelement.marmotcellelement

from edelweissfe.utils.exceptions import CutbackRequest

from libc.stdlib cimport free, malloc
from libcpp.memory cimport allocator, make_unique, unique_ptr

from edelweissmeshfree.cells.marmotcell.marmotcell cimport MarmotCell, MarmotCellWrapper
from edelweissmeshfree.materialpoints.marmotmaterialpoint.mp cimport (
    MarmotMaterialPointWrapper,
)

from edelweissmeshfree.materialpoints.marmotmaterialpoint.mp import (
    MarmotMaterialPointWrapper,
)


@cython.final # no subclassing -> cpdef with nogil possible
cdef class MarmotCellElementWrapper(MarmotCellWrapper):
    """This cell as a wrapper for MarmotCellElements.
    This class is not intended to be used directly, but to be subclassed by the actual cell element classes, such as LagrangianMarkerCellElement.

    For the documentation of MarmotCellElements, please refer to `Marmot <https://github.com/MAteRialMOdelingToolbox/Marmot/>`_.

    Parameters
    ----------
    cellElementType
        The Marmot element which should be represented, e.g., CPE4.
    cellElementNumber
        The (unique) label of this CellElement.
    nodes
        The list of nodes for this CellElement.
    quadratureType
        The type of quadrature to be used.
    quadratureOrder
        The order of the quadrature to be used."""
    def __init__(self, cellType, cellNumber, nodes, quadratureType, quadratureOrder):
        super().__init__(cellType, cellNumber, nodes)

    @property
    def nMaterialPoints(self):
        return self._nMaterialPoints

    @property
    def elNumber(self):
        return self._cellNumber

    def getRequestedMaterialPointCoordinates(self, ):
        """Get the coordinates of the material points in the cell."""
        cdef int dim = self._nodeCoordinates.shape[1]
        cdef np.ndarray mpCoordinates = np.zeros( (self._nMaterialPoints, dim) )
        cdef double[:,::1] _coordsView = mpCoordinates
        self._marmotCellElement.getRequestedMaterialPointCoordinates(&_coordsView[0, 0])
        return mpCoordinates

    def getRequestedMaterialPointVolumes(self, ):
        """Get the volumes of the material points in the cell."""
        cdef np.ndarray vols = np.zeros(self._nMaterialPoints)
        cdef double[::1] _volsView = vols
        self._marmotCellElement.getRequestedMaterialPointVolumes(&_volsView[0])
        return vols

    def getRequestedMaterialPointType(self, ):
        """Get the type of the material points in the cell."""
        return MarmotMaterialPointWrapper

    def computeKernels(self,
                       double[::1] Kc,
                       double[::1] Pc,
                       const double[::1] Uc,
                       const double[::1] dUc,
                       double timeNew,
                       double dTime):
        """Add the internal forces of the material points to the residual, and their tangent; see
        :meth:`edelweissmeshfree.cells.base.cell.CellBase.computeKernels`. Unlike a cell, a cell element deforms like
        an element, so Marmot gets the current solution, not its increment. A failure in Marmot requests a cutback,
        as for an EdelweissFE element."""

        try:
            with nogil:
                self._marmotCell.computeMaterialPointKernels(&Uc[0], &Pc[0], &Kc[0], timeNew, dTime)
        except (RuntimeError, ValueError) as e:
            raise CutbackRequest(str(e), 0.5)

    def acceptLastState(self, ):
        pass
