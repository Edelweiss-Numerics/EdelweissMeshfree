# -*- coding: utf-8 -*-
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
"""The search for the kernel functions that cover a point: the kernel functions sorted into a regular grid of bins by
their boxes (the broad phase), and the test whether a point lies inside a box (the narrow phase for kernel functions
whose support is their box).

Both loops run in compiled code, without the Python interpreter: a search visits every evaluation point of every
particle in every increment, and a few Python operations per point would cost more than all the rest of an explicit
increment.
"""

import numpy as np

cimport cython
from libcpp.algorithm cimport sort
from libcpp.vector cimport vector


# final: the methods are called directly, not through the table of a class that might derive from it, in the loop over
# the points
@cython.final
cdef class BinsOfKernelFunctions:
    """The boxes of the kernel functions sorted into a regular grid of bins: a kernel function is in every bin its box
    overlaps.

    The grid spans all boxes; a bin is half as wide as the mean box along each dimension. The bins are held in the
    compressed layout of a sparse matrix: the kernel functions of bin ``b`` are
    ``kernelsInBins[firstKernelOfBin[b] : firstKernelOfBin[b + 1]]``, in ascending order. The bins are numbered with the
    first dimension fastest: the flat index of a bin is the sum, over the dimensions, of its index along the dimension
    times the stride of the dimension (:attr:`strides`).

    A point lies in one bin, and every box that holds the point overlaps that bin. So the kernel functions whose box
    holds a point are among the kernel functions of its bin, and only they are tested
    (:meth:`kernelsWhoseBoxesHoldThePoints`).

    Parameters
    ----------
    lowerCorners
        The lower corner of the box of every kernel function, one row per kernel function.
    upperCorners
        The upper corners.
    """

    cdef readonly int dimension
    cdef double[:, ::1] lowerCorners, upperCorners
    cdef Py_ssize_t[::1] kernelsInBins, firstKernelOfBin
    cdef double[::1] _gridLowerCorner, _gridUpperCorner, _binSize
    # along each of three dimensions: a lacking dimension has one layer of bins, and the stride 0
    cdef Py_ssize_t _numberOfBins[3]
    cdef Py_ssize_t _strides[3]

    def __init__(self, lowerCorners, upperCorners):
        lower = np.ascontiguousarray(lowerCorners, dtype=np.float64)
        upper = np.ascontiguousarray(upperCorners, dtype=np.float64)
        self.lowerCorners, self.upperCorners = lower, upper
        self.dimension = lower.shape[1]
        cdef Py_ssize_t nKernels = lower.shape[0]
        cdef int d

        if not 1 <= self.dimension <= 3:
            raise ValueError("The bins are for one, two or three dimensions.")

        if nKernels:
            self._gridLowerCorner = np.min(lower, axis=0)
            self._gridUpperCorner = np.max(upper, axis=0)
            # Half the mean box: smaller bins hold fewer kernel functions for a point to test, but every box then lies
            # in more bins (from 2^d to 3^d for a box of mean size).
            self._binSize = np.mean(upper - lower, axis=0) / 2.0
        else:
            self._gridLowerCorner = np.zeros(self.dimension)
            self._gridUpperCorner = np.zeros(self.dimension)
            self._binSize = np.ones(self.dimension)

        for d in range(3):
            self._numberOfBins[d], self._strides[d] = 1, 0
        for d in range(self.dimension):
            # as many layers of bins as the upper corner of the grid needs to lie in the last one
            self._numberOfBins[d] = (
                <Py_ssize_t> ((self._gridUpperCorner[d] - self._gridLowerCorner[d]) / self._binSize[d]) + 1
                if nKernels
                else 0
            )
            self._strides[d] = 1 if d == 0 else self._strides[d - 1] * self._numberOfBins[d - 1]

        self._sortTheKernelsIntoTheBins()

    @property
    def gridLowerCorner(self):
        """The lower corner of the grid of bins, the lowest corner of all boxes."""
        return np.asarray(self._gridLowerCorner)

    @property
    def gridUpperCorner(self):
        """The upper corner of the grid of bins, the highest corner of all boxes."""
        return np.asarray(self._gridUpperCorner)

    @property
    def binSize(self):
        """The size of a bin along each dimension."""
        return np.asarray(self._binSize)

    @property
    def numberOfBins(self):
        """The number of layers of bins along each dimension."""
        return np.array([self._numberOfBins[d] for d in range(self.dimension)], dtype=np.intp)

    @property
    def strides(self):
        """The stride of each dimension in the flat index of a bin."""
        return np.array([self._strides[d] for d in range(self.dimension)], dtype=np.intp)

    cdef _sortTheKernelsIntoTheBins(self):
        """Fill the bins by a counting sort: count the kernel functions of each bin, place the first kernel function of
        each bin after those of all bins before it, then place the kernel functions in ascending order."""

        cdef Py_ssize_t nBins = self._numberOfBins[0] * self._numberOfBins[1] * self._numberOfBins[2]
        cdef Py_ssize_t nKernels = self.lowerCorners.shape[0]
        cdef Py_ssize_t[::1] count = np.zeros(nBins + 1, dtype=np.intp)
        cdef Py_ssize_t[:, ::1] lowest = np.zeros((nKernels, 3), dtype=np.intp)
        cdef Py_ssize_t[:, ::1] highest = np.zeros((nKernels, 3), dtype=np.intp)
        cdef Py_ssize_t k, binX, binY, binZ, b
        cdef int d

        # The bins a box overlaps along each dimension, from the one of its lower to the one of its upper corner (along
        # a lacking dimension, the one layer of bins). Rounding down is monotonic, so the bin of a point inside a box
        # lies between the bins of the box's corners: the box is in the bin of every point it holds.
        for k in range(nKernels):
            for d in range(self.dimension):
                lowest[k, d] = self._binIndexAlong(d, self.lowerCorners[k, d])
                highest[k, d] = self._binIndexAlong(d, self.upperCorners[k, d])

        # counted one place further, so that the cumulative sum gives the first place of every bin
        for k in range(nKernels):
            for binZ in range(lowest[k, 2], highest[k, 2] + 1):
                for binY in range(lowest[k, 1], highest[k, 1] + 1):
                    for binX in range(lowest[k, 0], highest[k, 0] + 1):
                        count[self._flatBinIndex(binX, binY, binZ) + 1] += 1

        self.firstKernelOfBin = np.cumsum(count).astype(np.intp)
        cdef Py_ssize_t[::1] nextPlaceInBin = np.asarray(self.firstKernelOfBin).copy()
        self.kernelsInBins = np.empty(self.firstKernelOfBin[nBins], dtype=np.intp)

        # the kernel functions in ascending order, so every bin lists its kernel functions in ascending order
        for k in range(nKernels):
            for binZ in range(lowest[k, 2], highest[k, 2] + 1):
                for binY in range(lowest[k, 1], highest[k, 1] + 1):
                    for binX in range(lowest[k, 0], highest[k, 0] + 1):
                        b = self._flatBinIndex(binX, binY, binZ)
                        self.kernelsInBins[nextPlaceInBin[b]] = k
                        nextPlaceInBin[b] += 1

    cdef inline Py_ssize_t _binIndexAlong(self, int d, double x) noexcept nogil:
        """The index of the layer of bins along dimension ``d`` that holds the coordinate ``x``: -1 below the grid (and
        for a NaN), the number of layers above it.

        Rounding down is monotonic, so the layer of a coordinate between two others lies between their layers. The
        explicit ends keep the conversion to an integer in range: converting a NaN or a too large double is undefined in
        C."""

        cdef double distance = (x - self._gridLowerCorner[d]) / self._binSize[d]
        if not distance >= 0.0:
            return -1
        if distance >= self._numberOfBins[d]:
            return self._numberOfBins[d]
        return <Py_ssize_t> distance

    cdef inline Py_ssize_t _flatBinIndex(self, Py_ssize_t binX, Py_ssize_t binY, Py_ssize_t binZ) noexcept nogil:
        """The flat index of the bin with the given indices along the dimensions (0 along a lacking dimension)."""

        return binX * self._strides[0] + binY * self._strides[1] + binZ * self._strides[2]

    cdef inline Py_ssize_t _binOfPoint(self, const double* point) noexcept nogil:
        """The flat index of the bin that holds the point; -1 beyond the grid, where no box holds it."""

        cdef Py_ssize_t[3] binAlong = [0, 0, 0]
        cdef int d
        for d in range(self.dimension):
            binAlong[d] = self._binIndexAlong(d, point[d])
            if binAlong[d] < 0 or binAlong[d] >= self._numberOfBins[d]:
                return -1
        return self._flatBinIndex(binAlong[0], binAlong[1], binAlong[2])

    cdef inline bint _boxHoldsPoint(self, Py_ssize_t k, const double* point) noexcept nogil:
        """Whether the box of kernel function ``k`` holds the point strictly inside."""

        cdef int d
        for d in range(self.dimension):
            if not (self.lowerCorners[k, d] < point[d] < self.upperCorners[k, d]):
                return False
        return True

    def kernelsInBin(self, Py_ssize_t binIndex):
        """The kernel functions whose boxes overlap a bin.

        Parameters
        ----------
        binIndex
            The flat index of the bin.

        Returns
        -------
        np.ndarray
            The indices of the kernel functions, in ascending order.
        """

        return np.asarray(self.kernelsInBins[self.firstKernelOfBin[binIndex] : self.firstKernelOfBin[binIndex + 1]])

    def kernelsWhoseBoxesHoldThePoints(self, const double[:, ::1] points, const Py_ssize_t[::1] firstPointOfGroup):
        """For every group of points (the evaluation points of a particle), the kernel functions whose box holds one of
        its points strictly inside: each once, in ascending order. Strictly: a point on the boundary of a box lies where
        a kernel function with box support is zero.

        Parameters
        ----------
        points
            The points, one row each, the points of a group consecutive.
        firstPointOfGroup
            The first point of every group, and the number of points at the end: the points of group ``g`` are
            ``points[firstPointOfGroup[g] : firstPointOfGroup[g + 1]]``.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``firstKernelOfGroup`` and ``kernels``, in the same layout: the kernel functions of group ``g`` are
            ``kernels[firstKernelOfGroup[g] : firstKernelOfGroup[g + 1]]``.
        """

        cdef Py_ssize_t nGroups = firstPointOfGroup.shape[0] - 1
        cdef Py_ssize_t[::1] firstKernelOfGroup = np.zeros(nGroups + 1, dtype=np.intp)
        cdef vector[Py_ssize_t] kernels
        # whether a kernel function is already found for the group at hand: reset after each group
        cdef unsigned char[::1] isFound = np.zeros(self.lowerCorners.shape[0], dtype=np.uint8)
        cdef Py_ssize_t g, p, n, k, b

        with nogil:
            for g in range(nGroups):
                for p in range(firstPointOfGroup[g], firstPointOfGroup[g + 1]):
                    b = self._binOfPoint(&points[p, 0])
                    if b < 0:
                        continue
                    for n in range(self.firstKernelOfBin[b], self.firstKernelOfBin[b + 1]):
                        k = self.kernelsInBins[n]
                        if not isFound[k] and self._boxHoldsPoint(k, &points[p, 0]):
                            isFound[k] = 1
                            kernels.push_back(k)

                # ascending numbers are the label order, the order a particle lists its kernel functions in
                sort(kernels.begin() + firstKernelOfGroup[g], kernels.end())
                for n in range(firstKernelOfGroup[g], <Py_ssize_t> kernels.size()):
                    isFound[kernels[n]] = 0
                firstKernelOfGroup[g + 1] = kernels.size()

        kernelsOfGroups = np.empty(kernels.size(), dtype=np.intp)
        cdef Py_ssize_t[::1] kernelsOfGroupsView = kernelsOfGroups
        for n in range(<Py_ssize_t> kernels.size()):
            kernelsOfGroupsView[n] = kernels[n]

        return np.asarray(firstKernelOfGroup), kernelsOfGroups
