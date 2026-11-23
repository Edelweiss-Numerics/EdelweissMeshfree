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
"""The limit on the condition of the moment matrix reaches Marmot's reproducing kernel approximations: a point on one
side of a nearly flat layer of kernels gets the Shepard functions above the limit, a point inside the kernels is
evaluated bit for bit as without a limit.

Skipped where Marmot is not built.
"""

import numpy as np
import pytest
from edelweissfe.points.node import Node

SUPPORT_RADIUS = 1.6
# one layer of kernels of spacing 1, every other centre lifted out of the plane z = 0 by a small jitter: a wall
CENTRES = [np.array([i, j, 0.02 * ((i + j) % 2)], dtype=float) for j in range(5) for i in range(5)]
ONE_SIDED_POINT = np.array([2.1, 1.8, 0.8])  # in the void next to the wall
INSIDE_POINT = np.array([2.1, 1.8, 0.01])  # within the wall


def _kernelsAndApproximation(approximationType: str, **kwargs):
    from edelweissmeshfree.meshfree.approximations.marmot import (
        marmotmeshfreeapproximation,
    )
    from edelweissmeshfree.meshfree.kernelfunctions.marmot.marmotmeshfreekernelfunction import (
        MarmotMeshfreeKernelFunctionWrapper,
    )

    if marmotmeshfreeapproximation.__file__.endswith(".py"):
        pytest.skip("Marmot is not built")

    kernels = [
        MarmotMeshfreeKernelFunctionWrapper(
            Node(label, centre), "BSplineBoxed", supportRadius=SUPPORT_RADIUS, continuityOrder=3
        )
        for label, centre in enumerate(CENTRES)
    ]
    approximation = marmotmeshfreeapproximation.MarmotMeshfreeApproximationWrapper(
        approximationType, 3, completenessOrder=1, **kwargs
    )
    return kernels, approximation


@pytest.mark.parametrize("approximationType", ["ReproducingKernel", "ReproducingKernelImplicitGradient"])
def test_a_one_sided_point_gets_the_shepard_functions_above_the_limit(approximationType):
    kernels, unlimited = _kernelsAndApproximation(approximationType)
    _, limited = _kernelsAndApproximation(approximationType, momentMatrixConditionLimit=1e4)

    N, _ = unlimited.computeShapeFunctionsAndGradients(ONE_SIDED_POINT, kernels)
    NLimited, dNLimited = limited.computeShapeFunctionsAndGradients(ONE_SIDED_POINT, kernels)

    # the linear shape functions extrapolate: of the size of the distance over the jitter
    assert np.abs(N).max() > 10.0
    phi = np.array([kernel.computeKernelFunction(ONE_SIDED_POINT) for kernel in kernels])
    np.testing.assert_allclose(NLimited, phi / phi.sum(), rtol=1e-12, atol=1e-15)
    assert np.abs(dNLimited).max() * SUPPORT_RADIUS < 10.0


@pytest.mark.parametrize("approximationType", ["ReproducingKernel", "ReproducingKernelImplicitGradient"])
def test_a_point_inside_the_kernels_is_bitwise_unchanged_by_the_limit(approximationType):
    kernels, unlimited = _kernelsAndApproximation(approximationType)
    _, limited = _kernelsAndApproximation(approximationType, momentMatrixConditionLimit=1e4)

    N, dN = unlimited.computeShapeFunctionsAndGradients(INSIDE_POINT, kernels)
    NLimited, dNLimited = limited.computeShapeFunctionsAndGradients(INSIDE_POINT, kernels)

    np.testing.assert_array_equal(NLimited, N)
    np.testing.assert_array_equal(dNLimited, dN)


def test_a_limit_below_one_is_refused():
    with pytest.raises(ValueError):
        _kernelsAndApproximation("ReproducingKernel", momentMatrixConditionLimit=0.5)
