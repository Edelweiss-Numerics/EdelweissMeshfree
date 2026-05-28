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
"""Node field and node field subset specializations for MPM simulations."""

from edelweissfe.fields.nodefield import NodeField, NodeFieldSubset
from edelweissfe.points.node import Node

from edelweissmeshfree.sets.cellelementset import CellElementSet
from edelweissmeshfree.sets.cellset import CellSet

# class MPMNodeFieldSubset:
#    pass


class MPMNodeField(NodeField):
    """Node field specialization for MPM nodes and node-based subsets."""

    def __init__(self, *args, **kwargs):
        """Initialize the MPM node field.

        Parameters
        ----------
        *args
            Positional arguments forwarded to :class:`~edelweissfe.fields.nodefield.NodeField`.
        **kwargs
            Keyword arguments forwarded to :class:`~edelweissfe.fields.nodefield.NodeField`.
        """
        super().__init__(*args, **kwargs)

    def _getNodeFieldSubsetClass(self):
        """Return the node field subset class used for MPM node fields."""
        return MPMNodeFieldSubset


class MPMNodeFieldSubset(NodeFieldSubset):
    """Node field subset specialization supporting MPM cell-based subsets."""

    def __init__(self, *args, **kwargs):
        """Initialize the MPM node field subset.

        Parameters
        ----------
        *args
            Positional arguments forwarded to :class:`~edelweissfe.fields.nodefield.NodeFieldSubset`.
        **kwargs
            Keyword arguments forwarded to :class:`~edelweissfe.fields.nodefield.NodeFieldSubset`.
        """
        super().__init__(*args, **kwargs)

    def _getSubsetNodes(self, subset) -> list[Node]:
        """Return the nodes contained in the requested MPM-aware subset."""
        if type(subset) is CellSet or type(subset) is CellElementSet:
            nodeCandidates = subset.extractNodeSet()
            return [n for n in nodeCandidates if n in self.parentNodeField._indicesOfNodesInArray]

        else:
            return super()._getSubsetNodes(subset)
