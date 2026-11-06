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
"""The convergence criterion of the former meshfree solvers as a parameter set of EdelweissFE's legacy criterion."""

import pytest

from edelweissmeshfree.solvers.meshfreeconvergencecriterion import (
    jobInfoWithMeshfreeTolerances,
    meshfreeFieldTolerances,
)


def test_every_field_but_the_scalar_variables_gets_the_meshfree_tolerances():
    jobInfo = jobInfoWithMeshfreeTolerances()

    for name, tolerance in meshfreeFieldTolerances.items():
        for field, value in jobInfo[name].items():
            if field != "scalar variables":
                assert value == tolerance


def test_a_tolerance_given_by_name_replaces_the_meshfree_one_of_every_field():
    jobInfo = jobInfoWithMeshfreeTolerances(fluxResidualTolerance=5e-3)

    assert jobInfo["fluxResidualTolerance"]["displacement"] == 5e-3
    assert jobInfo["fluxResidualTolerance"]["scalar variables"] != 5e-3
    assert (
        jobInfo["absoluteFluxResidualTolerance"]["displacement"]
        == meshfreeFieldTolerances["absoluteFluxResidualTolerance"]
    )


def test_an_unknown_tolerance_is_refused():
    with pytest.raises(ValueError, match="no such tolerance"):
        jobInfoWithMeshfreeTolerances(fluxTolerance=5e-3)
