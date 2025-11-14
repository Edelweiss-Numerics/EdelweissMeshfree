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


class MaterialPointSet(set):
    """A basic node set.
    It has a label, and a list containing the unique nodes.

    Parameters
    ----------
    label
        The unique label for this element set.
    nodes
        A list of nodes.
    """

    def __init__(
        self,
        name: str,
        mps: list,
    ):
        self.name = name
        super().__init__(mps)

    def __hash__(self):
        return hash(self.name)
