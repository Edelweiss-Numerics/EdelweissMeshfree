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
"""The defaults of the quasi-static solver's options cannot be changed by a simulation.

They are a class attribute, shared by every simulation in the Python process. Example 124 once wrote its loose
per-field tolerances into the per-field dicts of a copy of them; the copy was shallow, so these were the dicts of the
class, and every later simulation in the same process inherited the tolerances. The defaults are read-only at every
level.
"""

import pytest

from edelweissmeshfree.solvers.nqs import NonlinearQuasistaticSolver


def test_no_default_is_a_mutable_container():
    for name, default in NonlinearQuasistaticSolver.validOptions.items():
        assert not isinstance(default, (dict, list, set)), name


def test_the_defaults_cannot_be_changed_through_a_copy():
    options = NonlinearQuasistaticSolver.validOptions.copy()

    with pytest.raises(TypeError):
        options["spec. relative flux residual tolerances"]["nonlocal damage"] = 1e10

    assert NonlinearQuasistaticSolver.validOptions["spec. relative flux residual tolerances"] == {}


def test_the_defaults_cannot_be_changed_directly():
    with pytest.raises(TypeError):
        NonlinearQuasistaticSolver.validOptions["max. iterations"] = 1
