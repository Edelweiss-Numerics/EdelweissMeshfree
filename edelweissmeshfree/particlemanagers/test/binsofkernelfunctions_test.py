# -*- coding: utf-8 -*-
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
"""The bins of the kernel functions, the search of the particle manager: every kernel function is in exactly the bins
its box overlaps, and for every group of points the search returns exactly the kernel functions whose box holds one of
the points strictly inside, each once, in ascending order."""

import itertools

import numpy as np
import pytest

from edelweissmeshfree.particlemanagers.binsofkernelfunctions import (
    BinsOfKernelFunctions,
)


def _boxes(dimension: int, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    centres = rng.uniform(0.0, 10.0, (n, dimension))
    halfWidths = rng.uniform(0.5, 2.0, (n, 1)) * np.ones(dimension)
    return centres - halfWidths, centres + halfWidths


@pytest.mark.parametrize("dimension", [1, 2, 3])
def test_every_kernel_is_in_exactly_the_bins_its_box_overlaps(dimension):
    lower, upper = _boxes(dimension, 60, np.random.default_rng(dimension))
    bins = BinsOfKernelFunctions(lower, upper)

    lowest = np.floor((lower - bins.gridLowerCorner) / bins.binSize).astype(int)
    highest = np.floor((upper - bins.gridLowerCorner) / bins.binSize).astype(int)
    assert np.all(highest < bins.numberOfBins), "the grid holds every box"
    expected = [[] for _ in range(int(np.prod(bins.numberOfBins)))]
    for kernel, (lo, hi) in enumerate(zip(lowest, highest)):
        for index in itertools.product(*[range(lo[d], hi[d] + 1) for d in range(dimension)]):
            expected[sum(i * bins.strides[d] for d, i in enumerate(index))].append(kernel)

    for binIndex, kernelsInBin in enumerate(expected):
        assert bins.kernelsInBin(binIndex).tolist() == kernelsInBin


@pytest.mark.parametrize("dimension", [1, 2, 3])
def test_the_search_returns_the_kernels_whose_boxes_hold_a_point_of_the_group(dimension):
    """Groups of 1 to 6 points, some beyond the grid on either side, some on the boundary of a box (not inside)."""

    rng = np.random.default_rng(10 + dimension)
    lower, upper = _boxes(dimension, 80, rng)
    bins = BinsOfKernelFunctions(lower, upper)

    pointsOfGroups = [rng.uniform(-4.0, 14.0, (rng.integers(1, 7), dimension)) for _ in range(200)]
    for points in pointsOfGroups[::10]:
        points[0, rng.integers(dimension)] = lower[rng.integers(len(lower)), rng.integers(dimension)]
    firstPointOfGroup = np.cumsum([0] + [len(points) for points in pointsOfGroups], dtype=np.intp)

    firstKernelOfGroup, kernels = bins.kernelsWhoseBoxesHoldThePoints(np.concatenate(pointsOfGroups), firstPointOfGroup)

    for g, points in enumerate(pointsOfGroups):
        holds = np.all((points[:, None, :] > lower[None]) & (points[:, None, :] < upper[None]), axis=2)
        expected = np.flatnonzero(np.any(holds, axis=0))
        assert kernels[firstKernelOfGroup[g] : firstKernelOfGroup[g + 1]].tolist() == expected.tolist()


@pytest.mark.parametrize("dimension", [1, 2, 3])
def test_a_point_beyond_the_grid_of_bins_is_in_no_box(dimension):
    """On either side of every dimension: a particle that left the supports of all kernel functions has no kernel
    functions, and the manager reports it by its own message."""

    bins = BinsOfKernelFunctions(*_boxes(dimension, 20, np.random.default_rng(20 + dimension)))
    extent = bins.gridUpperCorner - bins.gridLowerCorner

    points = []
    for d in range(dimension):
        for side in (-1.0, 1.0):
            point = bins.gridLowerCorner + 0.5 * extent
            point[d] = (bins.gridLowerCorner if side < 0 else bins.gridUpperCorner)[d] + side * 0.1 * extent[d]
            points.append(point)

    firstKernelOfGroup, kernels = bins.kernelsWhoseBoxesHoldThePoints(
        np.array(points), np.arange(len(points) + 1, dtype=np.intp)
    )
    assert kernels.size == 0 and np.all(firstKernelOfGroup == 0)
