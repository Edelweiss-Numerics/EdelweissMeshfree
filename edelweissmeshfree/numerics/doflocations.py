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
#  Matthias Neuner |  matthias.neuner@boku.ac.at
#  Thomas Mader    |  thomas.mader@bokut.ac.at
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


from edelweissfe.numerics.doflocations import DofLocations
from edelweissfe.numerics.dofnumbering import DofNumbering


class MPMDofLocations(DofLocations):
    """The locations of the entities of a material point or RKPM model in the numbering of the degrees of freedom of
    its active domain: those of
    :class:`~edelweissfe.numerics.doflocations.DofLocations` -- the finite elements and the constraints -- and those of
    the background cells, the cell elements and the particles, which are computed like elements.

    The locations of the particles are the locations of the nodes their kernel functions reach, node after node, on
    each node the same fields. Moving particles reach other nodes of the same numbering: new locations of the
    particles on the same numbering, see :meth:`withNewLocationsOf`.

    Parameters
    ----------
    numbering : DofNumbering
        The numbering of the degrees of freedom of the active domain.
    elements : list
        A list of elements.
    constraints : list
        A list of constraints.
    cells : list
        A list of cells.
    cellElements : list
        A list of cell elements.
    particles : list
        The list of particles (RKPM).
    """

    def __init__(
        self,
        numbering: DofNumbering,
        elements: list = None,
        constraints: list = None,
        cells: list = None,
        cellElements: list = None,
        particles: list = None,
    ):
        cells = [] if cells is None else list(cells)
        cellElements = [] if cellElements is None else list(cellElements)
        particles = [] if particles is None else list(particles)

        super().__init__(numbering, elements, constraints, [*cells, *cellElements, *particles])

        #: The cells that have a location, the active cells.
        self.cells = cells
        #: The cell elements that have a location.
        self.cellElements = cellElements
        #: The particles that have a location.
        self.particles = particles
