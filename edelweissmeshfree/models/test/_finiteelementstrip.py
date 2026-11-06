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
"""A strip of finite elements in a meshfree model, for the tests of the models and the solvers: EdelweissFE's plane
strain quadrilaterals (``CPE4``, linear elastic), unit squares along x, one element high."""

import numpy as np
from edelweissfe.config.elementlibrary import getElementClass
from edelweissfe.config.materiallibrary import getMaterialClass
from edelweissfe.points.node import Node
from edelweissfe.sections.plane import PlaneSectionSchema, Section
from edelweissfe.sets.elementset import ElementSet
from edelweissfe.sets.nodeset import NodeSet

from edelweissmeshfree.models.mpmmodel import MPMModel

YOUNGS_MODULUS = 1000.0
POISSONS_RATIO = 0.3


def addFiniteElementStrip(model: MPMModel, numberOfElements: int, sectionType: type = Section) -> Section:
    """Add the strip to a model: nodes 1 to ``numberOfElements + 1`` at the bottom, the ones above them numbered on
    from there; the element set ``strip``, the node sets ``left``, ``leftBottom`` and ``right``, and the section
    ``strip``.

    Parameters
    ----------
    model
        The model.
    numberOfElements
        The number of elements along x.
    sectionType
        The type of the plane section of the strip.

    Returns
    -------
    Section
        The section of the strip.
    """

    nNodesPerRow = numberOfElements + 1
    for row in range(2):
        for i in range(nNodesPerRow):
            label = 1 + i + row * nNodesPerRow
            model.nodes[label] = Node(label, np.array([float(i), float(row)]))

    ElementClass = getElementClass("CPE4", "edelweiss")
    for i in range(numberOfElements):
        element = ElementClass("CPE4", i + 1)
        bottom, top = 1 + i, 1 + i + nNodesPerRow
        element.setNodes([model.nodes[bottom], model.nodes[bottom + 1], model.nodes[top + 1], model.nodes[top]])
        model.elements[i + 1] = element

    model.elementSets["strip"] = ElementSet("strip", list(model.elements.values()))
    model.nodeSets["left"] = NodeSet("left", [model.nodes[1], model.nodes[1 + nNodesPerRow]])
    model.nodeSets["leftBottom"] = NodeSet("leftBottom", [model.nodes[1]])
    model.nodeSets["right"] = NodeSet("right", [model.nodes[nNodesPerRow], model.nodes[2 * nNodesPerRow]])

    material = getMaterialClass("linearelastic", "edelweiss")(np.array([YOUNGS_MODULUS, POISSONS_RATIO]))
    model.sections["strip"] = sectionType(
        "strip", model, material, [model.elementSets["strip"]], configuration=PlaneSectionSchema(thickness=1.0)
    )

    return model.sections["strip"]
