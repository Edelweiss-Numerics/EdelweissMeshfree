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
#  Institute of Structural Engineering, Research Group Computational Mechanics of Materials
#  BOKU University, Vienna
#  2023 - today
#
#  Matthias Neuner matthias.neuner@uibk.ac.at
#  Thomas Mader thomas.mader@boku.ac.at
#
#  This file is part of EdelweissMPM.
#
#  This library is free software; you can redistribute it and/or
#  modify it under the terms of the GNU Lesser General Public
#  License as published by the Free Software Foundation; either
#  version 2.1 of the License, or (at your option) any later version.
#
#  The full text of the license can be found in the file LICENSE.md at
#  the top level directory of EdelweissMPM.
#  ---------------------------------------------------------------------

import typing

import numpy as np
from edelweissfe.journal.journal import Journal
from edelweissfe.points.node import Node
from edelweissfe.sets.nodeset import NodeSet

from edelweissmpm.meshfree.kernelfunctions.base.basemeshfreekernelfunction import (
    BaseMeshfreeKernelFunction,
)
from edelweissmpm.models.mpmmodel import MPMModel

def generateKernelFunctionGridFromInputFile(
        model: MPMModel,
        journal: Journal,
        kernelFunctionFactoryCallback: typing.Callable[[Node], BaseMeshfreeKernelFunction],
        inputFilePath: str,
        name: str = "grid_from_file",
        firstKernelFunctionNumber: int = 1,
        ) -> MPMModel:
    """Generate a grid of particles from an Abaqus input file.
    Returns
    -------
    MPMModel
        The updated model.
    """
    # Read input file
    with open(inputFilePath, 'r') as file:
        lines = file.readlines()
    # Parse nodes and elements
    nodes = []
    elements = []
    reading_nodes = False
    reading_elements = False
    for line in lines:
        line = line.strip()
        parts = line.split(',')
        if line.lower().startswith('*node'):
            reading_nodes = True
            continue
        elif line.lower().startswith('*element,'):
            reading_elements = True
            reading_nodes = False
            continue
        elif line.startswith('*'):
            reading_nodes = False
            reading_elements = False
            continue
        if reading_nodes:
            node_id = int(parts[0])
            x = float(parts[1])
            y = float(parts[2])
            nodes.append((node_id, np.array([x, y])))
        elif reading_elements:
            #print(parts)
            element_id = int(parts[0])
            if len(parts) >= 5:
                # elements with 8 nodes (e.g., C3D8R) can be handled here if needed
                element_nodes = [int(p) for p in parts[1:5]]
            else:
                element_nodes = [int(p) for p in parts[1:]]
            elements.append((element_id, element_nodes))

    # Create kernel functions
    kfNodes = []
    currentKernelFunctionNumber = firstKernelFunctionNumber
    for element in elements:
        element_id, element_nodes = element
        # center coordinates of element nodes
        centerCoordinates = np.mean([nodes[node_id - 1][1] for node_id in element_nodes], axis=0)
        kf = kernelFunctionFactoryCallback(Node(currentKernelFunctionNumber, centerCoordinates))
        model.meshfreeKernelFunctions[currentKernelFunctionNumber] = kf
        currentKernelFunctionNumber += 1
        kfNodes.append(kf.node)

    # check if any of the node labels already exist in the model
    for n in kfNodes:
        if n.label in model.nodes:
            raise ValueError("Node with label {:} already exists in model.".format(n.label))

    model.nodes.update({n.label: n for n in kfNodes})

    #journal.info(f"Generated {len(kernelFunctions)} kernel functions from input file.")

    return model

if __name__ == "__main__":
    from edelweissmpm.meshfree.kernelfunctions.marmot.marmotmeshfreekernelfunction import (
        MarmotMeshfreeKernelFunctionWrapper,
    )
    #visualize read input file
    theModel = MPMModel(dimension=2)
    theJournal = Journal()
    theModel = generateKernelFunctionGridFromInputFile(
        model=theModel,
        journal=theJournal,
        kernelFunctionFactoryCallback=lambda node: MarmotMeshfreeKernelFunctionWrapper(node, "BSplineBoxed", supportRadius=2.0, continuityOrder=3), 
        inputFilePath="abaqusInput_Circle.inp",
        name="grid_from_file",
        firstKernelFunctionNumber=1,
    )

    #print(theModel.nodes)
    import matplotlib.pyplot as plt
    plt.figure()
    for n in theModel.nodes.values():
        plt.plot(n.coordinates[0], n.coordinates[1], 'ko', markersize=.1)
        #plt.text(n.coordinates[0], n.coordinates[1], str(n.label))
    plt.axis('equal')
    plt.savefig("tunnel.pdf")
    #plt.show()
