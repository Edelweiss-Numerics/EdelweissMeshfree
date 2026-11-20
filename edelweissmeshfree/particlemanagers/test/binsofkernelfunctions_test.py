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
"""The bins of the kernel functions, the broad phase of the particle manager's search: built as arrays, every kernel is
in exactly the bins its box overlaps, so a query returns every kernel whose box overlaps the query box."""

import itertools

import numpy as np
import pytest

from edelweissmeshfree.particlemanagers.kdbinorganizedparticlemanager import (
    _FastKDBinOrganizer,
)


class _BoxKernel:
    """A kernel function as the bins see it: its bounding box."""

    def __init__(self, lower: np.ndarray, upper: np.ndarray):
        self._box = (lower, upper)

    def getBoundingBox(self):
        return self._box


def _kernels(dimension: int, n: int, rng: np.random.Generator) -> list:
    centres = rng.uniform(0.0, 10.0, (n, dimension))
    halfWidths = rng.uniform(0.5, 2.0, (n, 1)) * np.ones(dimension)
    return [_BoxKernel(c - h, c + h) for c, h in zip(centres, halfWidths)]


@pytest.mark.parametrize("dimension", [1, 2, 3])
def test_every_kernel_is_in_exactly_the_bins_its_box_overlaps(dimension):
    rng = np.random.default_rng(dimension)
    kernels = _kernels(dimension, 60, rng)
    bins = _FastKDBinOrganizer(kernels, dimension)

    lowest = ((bins._mins - bins._boundingBoxMin) / bins._binSize).astype(int)
    highest = ((bins._maxs - bins._boundingBoxMin) / bins._binSize).astype(int)
    expected = [[] for _ in range(int(np.prod(bins._nBins)))]
    for kernel, (lo, hi) in enumerate(zip(lowest, highest)):
        for index in itertools.product(*[range(lo[d], hi[d] + 1) for d in range(dimension)]):
            expected[sum(i * bins._strides[d] for d, i in enumerate(index))].append(kernel)

    for binIndex, kernelsInBin in enumerate(expected):
        assert bins.kernelsInBin(binIndex).tolist() == kernelsInBin


@pytest.mark.parametrize("dimension", [1, 2, 3])
def test_a_query_returns_every_kernel_whose_box_overlaps_the_query_box(dimension):
    rng = np.random.default_rng(10 + dimension)
    kernels = _kernels(dimension, 80, rng)
    bins = _FastKDBinOrganizer(kernels, dimension)

    for _ in range(50):
        lower = rng.uniform(0.0, 10.0, dimension)
        upper = lower + rng.uniform(0.0, 1.0, dimension)
        candidates = bins.getCandidateIndices(lower, upper)

        overlapping = np.flatnonzero(np.all((upper >= bins._mins) & (lower <= bins._maxs), axis=1))
        assert np.all(np.diff(candidates) > 0), "each candidate once, in ascending order"
        assert set(overlapping) <= set(candidates)
