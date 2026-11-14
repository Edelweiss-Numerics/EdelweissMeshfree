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
"""The discrete system of a meshfree model, like that of a finite element model, holds the finite elements that carry
kernels (:meth:`~edelweissfe.models.femodel.FEModel.elementsWithKernels`): an entity of the element container that
carries geometry only, e.g. a contact facet, has no block in the system matrix and no location."""

from edelweissfe.journal.journal import Journal

from edelweissmeshfree.models.mpmmodel import MPMModel
from edelweissmeshfree.models.test._finiteelementstrip import addFiniteElementStrip


def test_the_discrete_system_leaves_out_an_element_without_kernels(monkeypatch):
    model = MPMModel(2)
    addFiniteElementStrip(model, 3)
    model.prepareYourself(Journal(verbose=False))
    first, second, third = model.elements.values()
    monkeypatch.setattr(type(second), "hasKernels", property(lambda self: self is not second))

    system = model.buildDiscreteSystem()

    assert system.elements == [first, third] and system.elementsWithNodalInertia == [first, third]
    assert second not in system.locations
