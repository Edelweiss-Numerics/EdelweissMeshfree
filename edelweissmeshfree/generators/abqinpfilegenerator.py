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

import typing

import numpy as np
from edelweissfe.journal.journal import Journal
from edelweissfe.points.node import Node
from edelweissfe.sets.nodeset import NodeSet

from edelweissmeshfree.meshfree.kernelfunctions.base.basemeshfreekernelfunction import (
    BaseMeshfreeKernelFunction,
)
from edelweissmeshfree.meshfree.kernelfunctions.marmot.marmotmeshfreekernelfunction import (
    MarmotMeshfreeKernelFunctionWrapper,
)
from edelweissmeshfree.models.mpmmodel import MPMModel
from edelweissmeshfree.particles.base.baseparticle import BaseParticle
from edelweissmeshfree.sets.particleset import ParticleSet


def computeParticleArea(vertices):
    """
    Computes the area of an arbitrary polygon by the shoelace formula.
    vertices: list of edge point coordinates [x, y], z. B. [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    """
    n = len(vertices)
    s1 = sum(vertices[i][0] * vertices[(i + 1) % n][1] for i in range(n))
    s2 = sum(vertices[i][1] * vertices[(i + 1) % n][0] for i in range(n))
    return abs(s1 - s2) / 2


def kernelFunctionFactoryCallback(
    node: Node, kernelFunction: str, supportRadius: float, continuityOrder: int
) -> BaseMeshfreeKernelFunction:
    return MarmotMeshfreeKernelFunctionWrapper(
        node, kernelFunction, supportRadius=supportRadius, continuityOrder=continuityOrder
    )


def parse_inp_file(input_file_path):
    nodes = {}
    elements = {}
    surfaces = {}
    nsets = {}
    elsets = {}

    reading_nodes = False
    reading_elements = False
    reading_surfaces = False
    current_nset = None
    current_elset = None
    generate_flag = False

    with open(input_file_path, "r") as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("**"):  # Kommentar oder leer
                continue

            # Abschnittserkennung
            lower = line.lower()
            if lower.startswith("*node"):
                reading_nodes = True
                reading_elements = False
                reading_surfaces = False
                current_nset = None
                current_elset = None
                continue

            elif lower.startswith("*element"):
                reading_nodes = False
                reading_elements = True
                reading_surfaces = False
                current_nset = None
                current_elset = None
                continue

            elif lower.startswith("*surface"):
                reading_nodes = False
                reading_elements = False
                reading_surfaces = True
                current_nset = None
                current_elset = None
                current_surface = line.split("name=")[1].split(",")[0].strip() if "=" in line else "default"
                continue

            elif lower.startswith("*nset"):
                reading_nodes = reading_elements = False
                current_nset = line.split("=")[1].split(",")[0].strip() if "=" in line else "default"
                current_elset = None
                generate_flag = "generate" in lower
                continue

            elif lower.startswith("*elset"):
                reading_nodes = reading_elements = False
                current_elset = line.split("=")[1].split(",")[0].strip() if "=" in line else "default"
                current_nset = None
                generate_flag = "generate" in lower
                continue

            elif line.startswith("*"):  # others
                reading_nodes = reading_elements = False
                current_nset = current_elset = None
                continue

            # --- read nodes ---
            if reading_nodes:
                parts = [p.strip() for p in line.split(",") if p.strip()]
                node_id = int(parts[0])
                coords = np.array(list(map(float, parts[1:])))
                nodes[node_id] = coords
                continue

            # --- read elements ---
            if reading_elements:
                parts = [p.strip() for p in line.split(",") if p.strip()]
                element_id = int(parts[0])
                element_nodes = list(map(int, parts[1:]))
                elements[element_id] = element_nodes
                continue

            # --- read surfaces ---
            # if reading_surfaces:
            #    parts = [p.strip() for p in line.split(',') if p.strip()]
            #    surface_elSet = parts[0]
            #    surface_orientation = int(parts[1][-1])
            #    if current_surface not in surfaces.keys():
            #        surfaces[current_surface] = {surface_elSet: surface_orientation}
            #    else:
            #        surfaces[current_surface][surface_elSet] = surface_orientation

            # --- read node- or element set ---
            if current_nset or current_elset:
                parts = [p.strip() for p in line.split(",") if p.strip()]
                if generate_flag:
                    start, end = map(int, parts[:2])
                    step = int(parts[2]) if len(parts) > 2 else 1
                    ids = list(range(start, end + 1, step))
                    generate_flag = False
                else:
                    ids = list(map(int, parts))

                if current_nset:
                    nsets.setdefault(current_nset, []).extend(ids)
                elif current_elset:
                    elsets.setdefault(current_elset, []).extend(ids)

    return nodes, elements, nsets, elsets, surfaces


def generateKernelFunctionGridFromInputFile(
    inputFilePath: str,
    journal: Journal,
    model: MPMModel,
    # kernelFunctionFactoryCallback: typing.Callable[[Node], BaseMeshfreeKernelFunction],
    kernelFunctionSpecifier: dict,
    particleFactoryCallback: typing.Callable[[np.ndarray, float], BaseParticle],
    firstKernelFunctionNumber: int = 1,
    firstParticleNumber: int = 1,
    name: str = "grid_from_file",
) -> MPMModel:
    """Generate a grid of particles from an Abaqus input file.
    Returns
    -------
    MPMModel
        The updated model.
    """
    nodes, elements, nsets, elsets, surfaces = parse_inp_file(inputFilePath)

    kfNodes = []
    particles = []
    currentKernelFunctionNumber = firstKernelFunctionNumber
    currentParticleNumber = firstParticleNumber
    feNodesParticlesMaps = []
    # print(nsets)

    for nsetName, nodeLabels in nsets.items():
        for n in nodeLabels:
            otherNsetName = nsets.keys()
            # print(otherNsetName)
            for nSetOther in otherNsetName:
                if nsetName == nSetOther:
                    continue
                if n in nsets[nSetOther]:
                    nsets[nsetName].remove(n)

    for element_id, element_nodes in elements.items():

        # center coordinates of element nodes
        maxXDist = max([nodes[node_id][0] for node_id in element_nodes]) - min(
            [nodes[node_id][0] for node_id in element_nodes]
        )
        maxYDist = max([nodes[node_id][1] for node_id in element_nodes]) - min(
            [nodes[node_id][1] for node_id in element_nodes]
        )
        # compute supportRadius in dependence of the maximum x or y distance of each particle
        theSupportRadius = kernelFunctionSpecifier["supportRadiusFactor"] * max(maxXDist, maxYDist)

        centerCoordinates = np.mean([nodes[node_id] for node_id in element_nodes], axis=0)

        kf = kernelFunctionFactoryCallback(
            Node(currentKernelFunctionNumber, centerCoordinates),
            kernelFunction=kernelFunctionSpecifier["kernelFunction"],
            supportRadius=theSupportRadius,
            continuityOrder=kernelFunctionSpecifier["continuityOrder"],
        )
        # kf = kernelFunctionFactoryCallback(Node(currentKernelFunctionNumber, centerCoordinates))
        model.meshfreeKernelFunctions[currentKernelFunctionNumber] = kf
        # create particle for each element
        particleVertices = np.asarray([nodes[node_id] for node_id in element_nodes])
        # create map from fe node id to particle vertex index
        feNodesParticleVerticesMap = {node_id: idx for idx, node_id in enumerate(element_nodes)}
        feNodesParticlesMaps.append(feNodesParticleVerticesMap)

        particleVolume = computeParticleArea(particleVertices)
        particle = particleFactoryCallback(currentParticleNumber, particleVertices, particleVolume)
        model.particles[currentParticleNumber] = particle

        currentKernelFunctionNumber += 1
        currentParticleNumber += 1
        kfNodes.append(kf.node)
        particles.append(particle)

    # check if any of the node labels already exist in the model
    for n in kfNodes:
        if n.label in model.nodes:
            raise ValueError("Node with label {:} already exists in model.".format(n.label))

    model.nodes.update({n.label: n for n in kfNodes})

    model.nodeSets["{:}_all".format(name)] = NodeSet("{:}_all".format(name), kfNodes)

    # create particle- and nodeSets for model
    model.particleSets[f"{name}_all"] = ParticleSet(f"{name}_all", particles)
    for elsetName, elementLabels in elsets.items():

        model.particleSets[f"{name}_{elsetName}"] = ParticleSet(
            f"{name}_{elsetName}", [particles[e - 1] for e in elementLabels]
        )
        model.nodeSets[f"{name}_{elsetName}"] = NodeSet(f"{name}_{elsetName}", [kfNodes[e - 1] for e in elementLabels])

        if not elsetName.startswith("_"):
            model.vertexSets[f"{name}_{elsetName}_verts"] = []

            seen_nodes = []
            for idx, e in enumerate(elementLabels):
                vertices = []
                for n in feNodesParticlesMaps[e - 1].keys():
                    if n in nsets[elsetName]:
                        v = feNodesParticlesMaps[e - 1][n]
                        if n not in seen_nodes:
                            vertices.append(v)
                            seen_nodes.append(n)
                model.vertexSets[f"{name}_{elsetName}_verts"].append(vertices)

    # add surface to model
    # for surfaceName, surfaceData in surfaces.items():
    #    modelSurfaceName = f"{name}_{surfaceName}"
    #    for elsetName, orientation in surfaceData.items():
    #        print(surfaceName, elsetName, orientation)
    #        if modelSurfaceName not in model.surfaces.keys():
    #            model.surfaces[f"{name}_{surfaceName}"] = {orientation: [particles[e-1] for e in elsets[elsetName]]}
    #        else:
    #            model.surfaces[f"{name}_{surfaceName}"][orientation] = [particles[e-1] for e in elsets[elsetName]]

    # journal.info(f"Generated {len(kernelFunctions)} kernel functions from input file.")

    return model


if __name__ == "__main__":
    from edelweissmeshfree.meshfree.approximations.marmot.marmotmeshfreeapproximation import (
        MarmotMeshfreeApproximationWrapper,
    )
    from edelweissmeshfree.meshfree.kernelfunctions.marmot.marmotmeshfreekernelfunction import (
        MarmotMeshfreeKernelFunctionWrapper,
    )
    from edelweissmeshfree.particles.marmot.marmotparticlewrapper import (
        MarmotParticleWrapper,
    )

    # visualize read input file
    theModel = MPMModel(dimension=2)
    theJournal = Journal()

    marmotParticleType = "GradientEnhancedMicropolarSQCNIxSDI/PlaneStrain/Quad"
    theApproximation = MarmotMeshfreeApproximationWrapper("ReproducingKernelImplicitGradient", 2, completenessOrder=2)
    theMaterial = {
        "material": "GMDamagedShearNeoHooke",
        "properties": np.array([240.565, 0.2, 1, 0.1, 0.2, 1.4999, 1.0]),
    }

    theModel = generateKernelFunctionGridFromInputFile(
        inputFilePath="inputBorehole_finer.inp",
        # inputFilePath="inputBorehole.inp",
        journal=theJournal,
        model=theModel,
        kernelFunctionSpecifier={"kernelFunction": "BSplineBoxed", "continuityOrder": 3, "supportRadiusFactor": 1.1},
        particleFactoryCallback=lambda number, vertexCoordinates, volume: MarmotParticleWrapper(
            marmotParticleType, number, vertexCoordinates, volume, theApproximation, theMaterial
        ),
        firstKernelFunctionNumber=1,
        firstParticleNumber=1,
        name="grid_from_file",
    )

    import matplotlib.pyplot as plt

    plt.figure()
    for n in theModel.nodes.values():
        plt.plot(n.coordinates[0], n.coordinates[1], "ko", markersize=1)
        # plt.text(n.coordinates[0], n.coordinates[1], str(n.label))
    for p in theModel.particles.values():
        verts = p.getVertexCoordinates()
        # verts = np.vstack((p.getVertexCoordinates(), p.vertexCoordinates[0]))
        plt.plot(verts[:, 0], verts[:, 1], linestyle="-", linewidth=0.5, color="k")

    # for f in theModel.meshfreeKernelFunctions.values():
    #    boundingBox = f.getBoundingBox()
    #    boundingBoxMin = boundingBox[0]
    #    boundingBoxMax = boundingBox[1]
    #    width = boundingBoxMax[0] - boundingBoxMin[0]
    #    height = boundingBoxMax[1] - boundingBoxMin[1]
    #    #plt.plot(boundingBoxMin[0], boundingBoxMin[1], 'ro', markersize=1)
    #    plt.gca().add_patch(plt.Rectangle(boundingBoxMin, width, height, linestyle='-', linewidth=1, edgecolor='r', facecolor='none'))

    for pSet in theModel.particleSets.values():
        color = np.random.rand(
            3,
        )
        for p in pSet.particles:
            verts = p.getVertexCoordinates()
            plt.fill(verts[:, 0], verts[:, 1], color=color, alpha=0.3)

    for vSetName, vSet in theModel.vertexSets.items():
        # if "right" in vSetName:
        for i, vertices in enumerate(vSet):
            # vertices = [0,1]
            for ver in vertices:
                vert = theModel.particleSets[vSetName.removesuffix("_verts")][i].getVertexCoordinates()[ver]
                plt.plot(vert[0], vert[1], "ro", markersize=5)
                plt.text(vert[0], vert[1], f"{i}|{ver}", color="r")
        # verts = theModel.particleSets[vSetName.removesuffix("_verts")][i].getVertexCoordinates()[ver]
        verts = theModel.particleSets[vSetName.removesuffix("_verts")][i].getVertexCoordinates()[vertices]

    # for nSet in theModel.nodeSets.values():
    #    if isinstance(nSet, list):
    #        continue
    #    #print(nSet)
    #    # get random color
    #    color = np.random.rand(3,)
    #    for n in nSet.nodes:
    #        plt.plot(n.coordinates[0], n.coordinates[1], 'o', color=color, markersize=3)

    # for surfaceName, surfaceData in theModel.surfaces.items():
    #    for orientation, particles in surfaceData.items():
    #        #print(surfaceName, orientation, len(particles))
    #        for p in particles:
    #            verts = p.getVertexCoordinates()
    #            plt.plot(verts[:, 0], verts[:, 1], linestyle='-', linewidth=2, color='b')

    plt.axis("equal")
    plt.show()
