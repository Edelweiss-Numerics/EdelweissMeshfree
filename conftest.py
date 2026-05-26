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
import pytest


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    """Automatically skip tests that raise NotImplementedError.

    This handles the case where optional Marmot modules (particles,
    materialpoints, cells) are not installed in the current environment.
    """
    outcome = yield
    if outcome.excinfo is not None and issubclass(outcome.excinfo[0], NotImplementedError):
        pytest.skip(str(outcome.excinfo[1]))


@pytest.fixture
def assert_gold():
    """Fixture that provides a helper for comparing numerical results against gold files.

    Prints a human-readable summary of absolute error, relative error and the norm of
    the difference before delegating to ``numpy.testing.assert_allclose``, so that
    failures include actionable diagnostic information.

    Usage::

        def test_sim(assert_gold):
            res  = run_sim()
            gold = np.loadtxt("gold.csv")
            assert_gold(res, gold)          # default tolerances
            assert_gold(res, gold, atol=1e-7)  # custom tolerances
    """

    def _assert_gold(res, gold, rtol=1e-5, atol=1e-8):
        res = np.asarray(res)
        gold = np.asarray(gold)
        abs_err = np.abs(res - gold)
        with np.errstate(divide="ignore", invalid="ignore"):
            rel_err = np.where(np.abs(gold) > 0, abs_err / np.abs(gold), abs_err)
        print(f"  Max absolute error : {abs_err.max():.3e}")
        print(f"  Max relative error : {rel_err.max():.3e}")
        print(f"  Norm of difference : {np.linalg.norm(res - gold):.3e}")
        np.testing.assert_allclose(res, gold, rtol=rtol, atol=atol)

    return _assert_gold
