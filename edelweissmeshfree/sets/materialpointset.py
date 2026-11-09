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

from edelweissfe.sets.orderedset import OrderedSet

from edelweissmeshfree.materialpoints.base.mp import MaterialPointBase
from edelweissmeshfree.materialpoints.marmotmaterialpoint.mp import (
    MarmotMaterialPointWrapper,
)


class MaterialPointSet(OrderedSet):
    """A basic material point set.
    It has a name, and contains the unique material points in the order in which they were given.

    That order is the order of iteration, so it is reproducible from run to run: the material point managers
    hand the material points to their cells in this order, and the loads and constraints defined on a set sum
    their contributions in this order. (A plain set iterates in the order of memory addresses, which changes
    from run to run, and with it the rounding of these sums.)

    Parameters
    ----------
    name
        The unique name for this material point set.
    mps
        The material points, in the order of iteration.
    """

    def __init__(
        self,
        name: str,
        mps,
    ):
        self.allowedObjectTypes = [MaterialPointBase, MarmotMaterialPointWrapper]

        super().__init__(name, mps)
