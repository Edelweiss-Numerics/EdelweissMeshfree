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
"""The compiled Marmot cells, cell elements and particles implement the protocol of :class:`BaseNodeCouplingEntity`
without deriving from it: a Cython extension type cannot derive from a Python class. Every public member of the base
must be there.

Skipped where Marmot is not built (the Python stand-ins of the extensions are no entities), as the suites already do.
"""

import itertools

import numpy as np
import pytest
from edelweissfe.nodecouplingentity.base.nodecouplingentity import (
    BaseNodeCouplingEntity,
)
from edelweissfe.points.node import Node


def _compiled(module):
    """The module, or skip the test if it is the Python stand-in of an extension that is not built."""

    if module.__file__.endswith(".py"):
        pytest.skip("Marmot is not built")

    return module


def _wrapperClasses() -> list[type]:
    from edelweissmeshfree.cellelements.marmotcellelement import (
        lagrangianmarmotcellelement,
    )
    from edelweissmeshfree.cells.marmotcell import (
        bsplinemarmotcell,
        lagrangianmarmotcell,
    )
    from edelweissmeshfree.particles.marmot import marmotparticlewrapper

    return [
        _compiled(lagrangianmarmotcell).LagrangianMarmotCellWrapper,
        _compiled(bsplinemarmotcell).BSplineMarmotCellWrapper,
        _compiled(lagrangianmarmotcellelement).LagrangianMarmotCellElementWrapper,
        _compiled(marmotparticlewrapper).MarmotParticleWrapper,
    ]


def _cell():
    from edelweissmeshfree.cells.marmotcell import lagrangianmarmotcell

    nodes = [Node(label, np.array([*xy, 0.0])) for label, xy in enumerate([(0, 0), (1, 0), (1, 1), (0, 1)])]

    return _compiled(lagrangianmarmotcell).LagrangianMarmotCellWrapper("Displacement/Quad4", 1, nodes)


def _particle():
    from edelweissmeshfree.meshfree.approximations.marmot.marmotmeshfreeapproximation import (
        MarmotMeshfreeApproximationWrapper,
    )
    from edelweissmeshfree.meshfree.kernelfunctions.marmot.marmotmeshfreekernelfunction import (
        MarmotMeshfreeKernelFunctionWrapper,
    )
    from edelweissmeshfree.particles.marmot import marmotparticlewrapper

    # K, G, yield stress (never reached), ..., mass density
    material = {
        "material": "FiniteStrainJ2Plasticity",
        "properties": np.array([1.0, 0.5, 1e9, 1e9, 1, 0, 1, 1.0e-8]),
    }
    particle = _compiled(marmotparticlewrapper).MarmotParticleWrapper(
        "Displacement/SQCNIxNSNI/PlaneStrain/Quad",
        1,
        np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]),
        0.0,
        MarmotMeshfreeApproximationWrapper("ReproducingKernel", 2, completenessOrder=1),
        material,
    )
    particle.assignKernelFunctions(
        [
            MarmotMeshfreeKernelFunctionWrapper(
                Node(label, np.array([*center, 0.0])), "BSplineBoxed", supportRadius=1.5, continuityOrder=3
            )
            for label, center in enumerate(itertools.product((-0.5, 0.5, 1.5), repeat=2))
        ]
    )
    particle.initializeYourself()

    return particle


def test_every_marmot_wrapper_has_every_public_member_of_the_entity_protocol():
    members = [name for name in dir(BaseNodeCouplingEntity) if not name.startswith("_")]

    for wrapper in _wrapperClasses():
        assert [name for name in members if name not in dir(wrapper)] == [], wrapper.__name__


@pytest.mark.parametrize("makeEntity, changesWithMotion", [(_cell, False), (_particle, True)])
def test_a_particle_changes_its_connectivity_with_the_motion_and_a_cell_does_not(makeEntity, changesWithMotion):
    entity = makeEntity()

    assert entity.connectivityChangesWithMotion is changesWithMotion
    assert list(entity.scalarVariables) == []
