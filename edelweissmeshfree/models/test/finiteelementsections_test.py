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
"""An MPMModel gives every finite element its section once, when it is prepared for the simulation.

The sections belong to the finite elements, which EdelweissFE's model assigns them to; the material points carry their
material from their creation. A section that counts its assignments shows how often each element received its
material, its properties and its state.
"""

from collections import Counter

from edelweissfe.journal.journal import Journal
from edelweissfe.sections.plane import Section

from edelweissmeshfree.models.mpmmodel import MPMModel
from edelweissmeshfree.models.test._finiteelementstrip import addFiniteElementStrip


class _CountingSection(Section):
    """A plane section that counts how often it is assigned to each element."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.assignmentsPerElement = Counter()

    def assignSectionPropertiesToElement(self, element, **kwargs):
        self.assignmentsPerElement[element] += 1
        super().assignSectionPropertiesToElement(element, **kwargs)


def test_every_finite_element_is_given_its_section_once():
    model = MPMModel(2)
    section = addFiniteElementStrip(model, 3, sectionType=_CountingSection)

    model.prepareYourself(Journal(verbose=False))

    assert section.assignmentsPerElement == Counter(model.elements.values())
