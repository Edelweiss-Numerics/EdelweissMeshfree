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

from libcpp.algorithm cimport sort
from libcpp.vector cimport vector


cdef class BinsOfKernelFunctions:
    """The boxes of the kernel functions sorted into a regular grid of bins: a kernel function is in every bin its box
    overlaps.

    The grid spans all boxes; a bin is half as wide as the mean box along each dimension. The bins are held in the compressed layout of a
    sparse matrix: the kernel functions of bin ``b`` are ``kernelsInBins[firstKernelOfBin[b] : firstKernelOfBin[b +
    1]]``, in ascending order. The bins are numbered with the first dimension fastest.

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
    cdef readonly object lowerCorners, upperCorners, gridLowerCorner, gridUpperCorner, binSize, numberOfBins, strides
    cdef readonly object kernelsInBins, firstKernelOfBin

    cdef double[:, ::1] _lower, _upper
    cdef double[::1] _origin, _binSize
    cdef Py_ssize_t[::1] _numberOfBins, _strides, _kernelsInBins, _firstKernelOfBin

    def __init__(self, lowerCorners, upperCorners):
        self.lowerCorners = np.ascontiguousarray(lowerCorners, dtype=np.float64)
        self.upperCorners = np.ascontiguousarray(upperCorners, dtype=np.float64)
        self.dimension = self.lowerCorners.shape[1]
        nKernels = self.lowerCorners.shape[0]

        if not 1 <= self.dimension <= 3:
            raise ValueError("The bins are for one, two or three dimensions.")

        if nKernels:
            self.gridLowerCorner = np.min(self.lowerCorners, axis=0)
            self.gridUpperCorner = np.max(self.upperCorners, axis=0)
            self.binSize = np.mean(self.upperCorners - self.lowerCorners, axis=0) / 2.0
        else:
            self.gridLowerCorner = np.zeros(self.dimension)
            self.gridUpperCorner = np.zeros(self.dimension)
            self.binSize = np.ones(self.dimension)
        self._lower, self._upper = self.lowerCorners, self.upperCorners
        self._origin, self._binSize = self.gridLowerCorner, self.binSize

        # along each dimension, as many layers of bins as the upper corner of the grid needs to lie in the last one
        self.numberOfBins = np.array(
            [self._binAlong(d, self.gridUpperCorner[d]) + 1 if nKernels else 0 for d in range(self.dimension)],
            dtype=np.intp,
        )
        # the flat index of a bin is the sum of its index along each dimension times the stride of the dimension
        self.strides = np.array([np.prod(self.numberOfBins[:d]) for d in range(self.dimension)], dtype=np.intp)
        self._numberOfBins, self._strides = self.numberOfBins, self.strides

        self._sortTheKernelsIntoTheBins()

    def _sortTheKernelsIntoTheBins(self):
        """Fill the bins by a counting sort: count the kernel functions of each bin, place the first kernel function of
        each bin after those of all bins before it, then place the kernel functions in ascending order."""

        cdef Py_ssize_t nBins = int(np.prod(self.numberOfBins))
        cdef Py_ssize_t nKernels = self._lower.shape[0]
        cdef Py_ssize_t[::1] count = np.zeros(nBins + 1, dtype=np.intp)
        cdef Py_ssize_t[:, ::1] lowest = np.zeros((nKernels, 3), dtype=np.intp)
        cdef Py_ssize_t[:, ::1] highest = np.zeros((nKernels, 3), dtype=np.intp)
        cdef Py_ssize_t[3] strides = [0, 0, 0]
        cdef Py_ssize_t k, d, i, j, l, b

        for d in range(self.dimension):
            strides[d] = self._strides[d]

        # the bins a box overlaps along each dimension, from the one of its lower to the one of its upper corner; a
        # lacking dimension has one layer of bins
        for k in range(nKernels):
            for d in range(self.dimension):
                lowest[k, d] = self._binAlong(d, self._lower[k, d])
                highest[k, d] = self._binAlong(d, self._upper[k, d])

        for k in range(nKernels):
            for l in range(lowest[k, 2], highest[k, 2] + 1):
                for j in range(lowest[k, 1], highest[k, 1] + 1):
                    for i in range(lowest[k, 0], highest[k, 0] + 1):
                        count[i * strides[0] + j * strides[1] + l * strides[2] + 1] += 1

        firstKernelOfBin = np.cumsum(count).astype(np.intp)
        cdef Py_ssize_t[::1] nextPlaceInBin = firstKernelOfBin.copy()
        cdef Py_ssize_t[::1] kernelsInBins = np.empty(firstKernelOfBin[nBins], dtype=np.intp)

        for k in range(nKernels):
            for l in range(lowest[k, 2], highest[k, 2] + 1):
                for j in range(lowest[k, 1], highest[k, 1] + 1):
                    for i in range(lowest[k, 0], highest[k, 0] + 1):
                        b = i * strides[0] + j * strides[1] + l * strides[2]
                        kernelsInBins[nextPlaceInBin[b]] = k
                        nextPlaceInBin[b] += 1

        self.firstKernelOfBin = firstKernelOfBin
        self.kernelsInBins = np.asarray(kernelsInBins)
        self._firstKernelOfBin, self._kernelsInBins = self.firstKernelOfBin, self.kernelsInBins

    cdef inline Py_ssize_t _binAlong(self, int d, double x) noexcept nogil:
        """The index of the layer of bins along dimension ``d`` that holds the coordinate ``x``; -1 below the grid.
        Rounding is monotonic: a coordinate between two others lies in a layer between theirs."""

        cdef double distance = (x - self._origin[d]) / self._binSize[d]
        if distance < 0.0:
            return -1
        return <Py_ssize_t> distance

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

        return self.kernelsInBins[self.firstKernelOfBin[binIndex] : self.firstKernelOfBin[binIndex + 1]]

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
        cdef Py_ssize_t nKernels = self._lower.shape[0]
        cdef int dim = self.dimension
        cdef Py_ssize_t[::1] firstKernelOfGroup = np.zeros(nGroups + 1, dtype=np.intp)
        cdef vector[Py_ssize_t] kernels
        # whether a kernel function is already found for the group at hand: reset after each group
        cdef unsigned char[::1] isFound = np.zeros(nKernels, dtype=np.uint8)
        cdef Py_ssize_t g, p, d, n, k, b, layer, firstOfGroup
        cdef bint isInside

        with nogil:
            for g in range(nGroups):
                firstOfGroup = kernels.size()
                for p in range(firstPointOfGroup[g], firstPointOfGroup[g + 1]):
                    # the bin of the point; a point beyond the grid lies in no box
                    b = 0
                    for d in range(dim):
                        layer = self._binAlong(d, points[p, d])
                        if layer < 0 or layer >= self._numberOfBins[d]:
                            b = -1
                            break
                        b += layer * self._strides[d]
                    if b < 0:
                        continue

                    for n in range(self._firstKernelOfBin[b], self._firstKernelOfBin[b + 1]):
                        k = self._kernelsInBins[n]
                        if isFound[k]:
                            continue
                        isInside = True
                        for d in range(dim):
                            if not (self._lower[k, d] < points[p, d] < self._upper[k, d]):
                                isInside = False
                                break
                        if isInside:
                            isFound[k] = 1
                            kernels.push_back(k)

                sort(kernels.begin() + firstOfGroup, kernels.end())
                for n in range(firstOfGroup, <Py_ssize_t> kernels.size()):
                    isFound[kernels[n]] = 0
                firstKernelOfGroup[g + 1] = kernels.size()

        kernelsOfGroups = np.empty(kernels.size(), dtype=np.intp)
        cdef Py_ssize_t[::1] kernelsOfGroupsView = kernelsOfGroups
        for n in range(<Py_ssize_t> kernels.size()):
            kernelsOfGroupsView[n] = kernels[n]

        return np.asarray(firstKernelOfGroup), kernelsOfGroups
