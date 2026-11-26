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
"""The contact points of particles: quadrature points on the faces of particles at the boundary of a body, for the
contact constraints of EdelweissFE, which pair them with a master surface of finite element faces
(:mod:`edelweissfe.constraints.surfacetodeformablesurfacepenalty`) or with a discrete rigid body
(:mod:`edelweissfe.constraints.surfacetodiscreterigidbodypenalty`). They are a third kind of EdelweissFE's contact
points (:mod:`edelweissfe.constraints.base.contactpoints`), next to the nodes and the quadrature points of finite
element faces: a point :math:`\\boldsymbol{x}_q` of a particle face, with :math:`c = G(\\boldsymbol{x}_q)`, at
:math:`\\boldsymbol{x} = \\boldsymbol{x}_q^n + G(\\boldsymbol{x}_q) \\cdot \\Delta\\boldsymbol{U}`, standing for the
area of its share of the face (*The area of a point*, below). The contact with the law ``type=tied``, a tie, takes
them alike: it ties faces of particles to finite elements. A change of the mesh never regenerates them
(:meth:`~edelweissfe.constraints.base.contactpoints.ContactPoints.regenerated`), so a tie hands its state on across a
new mesh of the master only.

A face of a particle moves with its particle: the particle moves every point of its own, a vertex or a point on a
face, by :math:`\\Delta\\boldsymbol{x} = \\sum_B G_B(\\boldsymbol{x}) \\Delta\\boldsymbol{U}_B`, with its
shape functions linearized about its center
(:meth:`~edelweissmeshfree.particles.base.baseparticle.BaseParticle.getInterpolationVectorLinearizedAboutCenter`), which
are affine in :math:`\\boldsymbol{x}`. So a face stays flat if it is flat, and a point at fixed parametric coordinates
:math:`\\xi_q` of the face stays there. Such a point is a contact point: it sits at
:math:`\\boldsymbol{x}_q = \\sum_a N_a(\\xi_q) \\boldsymbol{x}_a` (the face's vertices :math:`\\boldsymbol{x}_a`),
and stands for the area of its share of the face, of the measure :math:`J_q w_q`. The contact measures its motion with
:math:`G(\\boldsymbol{x}_q)`, and its contact force reaches the nodes of the particle's kernel functions with the same
:math:`G`, the transpose of its motion: the penalty stiffness :math:`k\\, G G^T` is symmetric and the contact
conservative. It is always :math:`G`, because a point at fixed :math:`\\xi_q` stays on the face only if it moves as
the particle moves the face: with the shape functions at the point, :math:`N(\\boldsymbol{x}_q)`, it would drift off
the face within an increment and jump back, by :math:`\\boldsymbol{n} \\cdot (G - N) \\Delta\\boldsymbol{U}`, when the
next update takes its origin from the face again.

The particles are updated-Lagrangian: the nodes of their kernel functions carry the increment
:math:`\\Delta\\boldsymbol{U}` since the last update of the connectivity, which a model with particles makes at the
start of every increment, not a total displacement. So the displacement a point moves with is that increment: its origin
:math:`\\boldsymbol{x}_q^n` is where it was at the update, and its displacement from there is :math:`c \\cdot
\\Delta\\boldsymbol{U}` (:meth:`ContactPointsOnParticleFaces.displacementsOfLocatedPoints`). The contact constraints
pass the total displacement ``U`` and the increment ``dU`` alike, and these points take ``dU``.

For a constant pressure on a face, :math:`\\sum_q w_q J_q G(\\boldsymbol{x}_q) = A\\, G(\\boldsymbol{x}_c)` for every
rule that integrates a linear function exactly: one point at the face's center already delivers the resultant and its
first moment, everything a particle with a homogeneous deformation can feel. Two points per edge (2 in 2D, 2 x 2 in
3D) integrate exactly as well the linearly varying gap of a tilted face, and the quadratic shape functions of a master
face of the same size, with which the contact loads the master
(:mod:`edelweissfe.constraints.surfacetodeformablesurfacepenalty`).

**The area of a point** is the area of the body its share of the face stands for, as the particles integrate, measured
from the model when the points are made, by the rule of every contact point
(:mod:`edelweissfe.constraints.base.contactpoints`): its measure :math:`J_q w_q` in the model's own space -- an area in
3D, a length in 2D -- times the out-of-plane extent :math:`b` of the body at the point, :math:`A_q = b\\, J_q w_q`.

- In 3D, :math:`b = 1`.
- A **plane model** with particles is a slice of **unit thickness**, :math:`b = 1`: Marmot's plane-strain particles
  integrate their forces and masses per unit thickness, and so do their points. The finite elements the particles meet
  must then be of plane sections of thickness 1 for the forces to balance; nothing checks this.
- In an **axisymmetric model**, the section r-z of a body of revolution
  (:attr:`~edelweissfe.models.femodel.FEModel.axisymmetric`), the particles are axisymmetric
  (:meth:`~edelweissmeshfree.models.mpmmodel.MPMModel.prepareYourself` checks them), and a face of a particle is an
  edge of the section that sweeps a ring (a conical, cylindrical or plane annular strip) about the axis. A point stands
  for the area of its ring, :math:`b = 2 \\pi r_q`, with the radius :math:`r_q` of the point, so that the contact forces
  are those of the full rings, as the particles' forces and masses are. A point on the axis stands for a ring of radius
  0, no area: the axis is a line, not a surface of the body. A point below the axis by no more than the round-off of
  the diagonal of its particle's bounding box (:data:`~edelweissfe.constraints.base.contactpoints.axisTolerance`, the
  rule of Marmot's axisymmetric particles and elements for their vertices) is on it; one further below is refused.
  The rule is EdelweissFE's, for every kind of contact point
  (:func:`~edelweissfe.constraints.base.contactpoints.circumferenceAt`); the particles give it their diagonals.

The ring weight is linear in :math:`r` along an edge, so one point at the edge's center already gives the exact area of
its ring, :math:`2 \\pi \\bar r L`. The nodal forces of a constant pressure on the ring,
:math:`\\int 2 \\pi r\\, G(\\boldsymbol{x})\\, \\mathrm{d}s`, are quadratic along the edge: two points per edge
integrate them exactly, one point does not.

The load at a free end. The particles balance a homogeneous stress with the traction on their faces distributed with
their shape functions at the centers of the faces, :math:`N(\\boldsymbol{x}_f)`, one point per face: their smoothed
gradients integrate :math:`N` over their faces with that rule, and a particle distributes its own pressure on a face
so. Only that load keeps the homogeneous state. The contact distributes with :math:`G` instead, alike at one or two
points per face; the shape functions at the points, :math:`N(\\boldsymbol{x}_q)`, at two points per face would differ
from it as well, since :math:`N` is not affine. On each face, the difference has no resultant and no moment, and it
does not cancel between neighbouring particles, since :math:`G` is not continuous from one particle to the next. Along
a straight row of equal particles it repeats from one column of kernel nodes to the next, without resultant in each
column, so it changes the pressure alike along the row. Where the row ends at a free corner of the body, the kernel
functions are no longer translates of each other, and the pressure on the last two or three particles deviates from
uniform by a few percent, by the same profile at every particle size. Two points per face, which a quadratic master
needs, also hold the tilt of a face: at a free end the pressure varies within each face as well (example 192, the
contact patch test).
"""

from collections.abc import Iterable

import numpy as np
from edelweissfe.constraints.base.contactpoints import ContactPoints, circumferenceAt
from edelweissfe.generators.surfaceelementgenerator import canonicalParentFace
from edelweissfe.models.femodel import FEModel
from edelweissfe.utils.isoparametricshapes import shapeFunctionsAndDerivatives
from scipy.sparse import csr_matrix

from edelweissmeshfree.constraints.base.pointsofparticles import (
    locateTheNodesOfParticles,
)

#: The vertices of each face of a particle, in the Abaqus numbering of the faces (face ID 1 first) that Marmot's
#: particles and the generators of EdelweissMeshfree use; by the number of vertices of the particle. They are the
#: parent faces of the finite element with the same vertices (``quad4``, ``hexa8``), from EdelweissFE's face tables
#: (:func:`~edelweissfe.generators.surfaceelementgenerator.canonicalParentFace`): each face lists its vertices
#: cyclically, in the order of the face shape functions ``bar2`` and ``quad4``.
verticesOfFace = {
    nVertices: np.array([canonicalParentFace(elementType, face)[1] for face in range(1, nFaces + 1)])
    for nVertices, elementType, nFaces in ((4, "quad4", 4), (8, "hexa8", 6))
}


def quadratureRuleOfFace(nDim: int, nQuadraturePoints: int) -> tuple[np.ndarray, np.ndarray]:
    """The Gauss rule on a face of a particle: Gauss-Legendre on an edge (2D), its tensor product on a face (3D).

    Parameters
    ----------
    nDim
        The spatial dimension.
    nQuadraturePoints
        The number of points on the face: :math:`n` on an edge, :math:`n^2` on a face.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        The parametric coordinates of the points, (nQuadraturePoints, nDim - 1), and their weights.

    Raises
    ------
    ValueError
        If the number of points is not :math:`n` (2D) or :math:`n^2` (3D) for some :math:`n \\ge 1`.
    """

    perEdge = round(nQuadraturePoints ** (1.0 / (nDim - 1))) if nQuadraturePoints >= 1 else 0
    if perEdge < 1 or perEdge ** (nDim - 1) != nQuadraturePoints:
        raise ValueError(
            f"{nQuadraturePoints} quadrature points per face are not a Gauss rule in {nDim}D: use n points per edge, "
            f"{'n' if nDim == 2 else 'n x n'} per face."
        )
    coordinates, weights = np.polynomial.legendre.leggauss(perEdge)
    if nDim == 2:
        return coordinates[:, None], weights
    eta, xi = np.meshgrid(coordinates, coordinates, indexing="ij")
    return np.column_stack((xi.ravel(), eta.ravel())), np.outer(weights, weights).ravel()


class ContactPointsOnParticleFaces(ContactPoints):
    """The quadrature points of faces of particles, used as contact points.

    The points are numbered face after face, in the order the faces are given, and within a face by quadrature point.
    The area each point stands for is measured when the points are made, from the model, in the configuration of the
    particles then, the reference of the contact: its measure times the out-of-plane extent of the body at it
    (see *The area of a point* in the module documentation). A penalty per unit area times it is the constant
    stiffness of the point's spring.

    A located point takes the nodes of its particle's kernel functions into the block of the constraint; a node shared
    by several located points, of one particle or of neighbouring ones, is located once
    (:func:`~edelweissmeshfree.constraints.base.pointsofparticles.locateTheNodesOfParticles`). The interpolation
    vectors of the located points form one sparse matrix :math:`C` over these nodes, made once per update of the
    connectivity: the points move by :math:`C\\, \\Delta\\boldsymbol{U}` from their origins, where they were at
    the update, and their forces reach the nodes with :math:`C^T`. The vectors are :math:`G(\\boldsymbol{x}_q)`, the
    particle's interpolation linearized about its center.

    Parameters
    ----------
    faces
        The faces: a surface of particles (face ID -> particles, e.g. from the generators of EdelweissMeshfree or from
        the side sets of an Exodus mesh), or (particle, face ID) pairs. The face IDs follow the Abaqus numbering, from
        1 (:data:`verticesOfFace`); others are refused.
    model
        The model of the particles, whose kind -- 3D, plane or axisymmetric -- the areas are measured as.
    nQuadraturePoints
        The number of Gauss points per face: :math:`n` on an edge (2D), :math:`n \\times n` on a face (3D). One point,
        the face's center, loads a quadratic master face only at its midside node and does not integrate the ring
        weight of an axisymmetric model: take 2 per edge (2 in 2D, 4 in 3D) for a quadratic master or an axisymmetric
        model (see the module documentation).
    """

    #: The points are located on the kernel functions of their particles, which the model connects at an update of the
    #: connectivity (and a checkpoint does not hold).
    locatedOnlyWhenConnected = True

    def __init__(self, faces: dict | Iterable, model: FEModel, nQuadraturePoints: int):
        pairs = (
            [(particle, faceID) for faceID, particles in faces.items() for particle in particles]
            if isinstance(faces, dict)
            else list(faces)
        )
        if not pairs:
            raise ValueError("No particle faces were given.")

        self.particlesOfFaces = [particle for particle, _ in pairs]
        self.faceIDs = np.array([faceID for _, faceID in pairs], dtype=np.int64)

        self._particles = list(dict.fromkeys(self.particlesOfFaces))
        indexOfParticle = {particle: i for i, particle in enumerate(self._particles)}
        self._particleOfFace = np.array([indexOfParticle[particle] for particle in self.particlesOfFaces])

        vertices = self._verticesOfParticles()
        nVertices, self.nDim = vertices.shape[1:]
        if nVertices not in verticesOfFace or self.nDim != {4: 2, 8: 3}[nVertices]:
            raise ValueError(
                f"Contact points on particle faces need quadrilateral (2D) or hexahedral (3D) particles; these have "
                f"{nVertices} vertices in {self.nDim}D."
            )
        nFacesOfParticle = len(verticesOfFace[nVertices])
        outside = (self.faceIDs < 1) | (self.faceIDs > nFacesOfParticle)
        if outside.any():
            raise ValueError(
                f"Particle face IDs follow the Abaqus numbering, 1 to {nFacesOfParticle} for these particles; got "
                f"{sorted(set(self.faceIDs[outside].tolist()))}."
            )
        parametric, weights = quadratureRuleOfFace(self.nDim, nQuadraturePoints)
        self.nQuadraturePoints = nQuadraturePoints

        self._verticesOfFace = verticesOfFace[nVertices][self.faceIDs - 1]
        shapeOfFace = "bar2" if self.nDim == 2 else "quad4"
        shapeFunctions, derivatives = zip(*(shapeFunctionsAndDerivatives(shapeOfFace, xi) for xi in parametric))
        #: The face's shape functions at its quadrature points, (nQuadraturePoints, nFaceVertices).
        self._shapeFunctions = np.array(shapeFunctions)
        #: Their derivatives with respect to the face's parametric coordinates, (nQuadraturePoints, nFaceVertices,
        #: nDim - 1).
        self._shapeFunctionDerivatives = np.array(derivatives)

        nFaces = len(self.faceIDs)
        self.nPoints = nFaces * nQuadraturePoints
        self._faceOfPoint = np.repeat(np.arange(nFaces), nQuadraturePoints)
        faceVertices = self._coordinatesOfFaceVertices(vertices)
        measures = (self._jacobians(faceVertices) * weights).ravel()
        extents = self._outOfPlaneExtentAt(self._coordinatesOfPointsOnFaces(faceVertices), vertices, model)
        self.areas = measures * extents
        self.axisymmetric = model.axisymmetric

        self.locatePoints(np.zeros(0, dtype=int))

    def _outOfPlaneExtentAt(
        self, coordinates: np.ndarray, verticesOfParticles: np.ndarray, model: FEModel
    ) -> np.ndarray:
        """The out-of-plane extent :math:`b` of the body at the points, by which a point's measure becomes its area: 1
        in 3D and in a plane model, whose particles are slices of unit thickness, :math:`2 \\pi r` in an axisymmetric
        model, by EdelweissFE's rule of the axis (:func:`~edelweissfe.constraints.base.contactpoints.circumferenceAt`),
        with the diagonal of the bounding box of each point's particle for the round-off (see the module
        documentation).

        Parameters
        ----------
        coordinates
            The points, (nPoints, nDim).
        verticesOfParticles
            The vertices of the particles of the faces, (nParticles, nVertices, nDim): the diagonals of their bounding
            boxes give the round-off of the axis.
        model
            The model.

        Returns
        -------
        numpy.ndarray
            :math:`b` at each point.

        Raises
        ------
        ValueError
            If a point of an axisymmetric model lies below the axis by more than the round-off.
        """

        if not model.axisymmetric:
            return np.ones(len(coordinates))

        diagonals = np.linalg.norm(verticesOfParticles.max(axis=1) - verticesOfParticles.min(axis=1), axis=-1)
        particleOfPoint = self._particleOfFace[self._faceOfPoint]
        return circumferenceAt(
            coordinates[:, 0],
            diagonals[particleOfPoint],
            lambda p: f"a face of particle {self._particles[particleOfPoint[p]].number}",
        )

    def _verticesOfParticles(self) -> np.ndarray:
        """The current vertices of the particles of the faces, (nParticles, nVertices, nDim)."""

        return np.array([particle.getVertexCoordinates() for particle in self._particles], dtype=float)

    def _coordinatesOfFaceVertices(self, verticesOfParticles: np.ndarray) -> np.ndarray:
        """The current vertices of each face, (nFaces, nFaceVertices, nDim)."""

        return np.take_along_axis(verticesOfParticles[self._particleOfFace], self._verticesOfFace[:, :, None], axis=1)

    def _jacobians(self, faceVertices: np.ndarray) -> np.ndarray:
        """The area (length in 2D) per unit parametric measure at each quadrature point, (nFaces, nQuadraturePoints)."""

        tangents = np.einsum("qak,fad->fqdk", self._shapeFunctionDerivatives, faceVertices)
        if self.nDim == 2:
            return np.linalg.norm(tangents[..., 0], axis=-1)
        return np.linalg.norm(np.cross(tangents[..., 0], tangents[..., 1]), axis=-1)

    def _coordinatesOfPointsOnFaces(self, faceVertices: np.ndarray) -> np.ndarray:
        """Where the points are on faces with the given vertices (nFaces, nFaceVertices, nDim): (nPoints, nDim)."""

        return np.einsum("qa,fad->fqd", self._shapeFunctions, faceVertices).reshape((-1, self.nDim))

    def _coordinatesOfAllPoints(self) -> np.ndarray:
        """Where all points are in the current configuration of the particles, (nPoints, nDim)."""

        return self._coordinatesOfPointsOnFaces(self._coordinatesOfFaceVertices(self._verticesOfParticles()))

    def describe(self) -> str:
        """The number of points, particle faces and quadrature points per face, for the journal.

        Returns
        -------
        str
            The description.
        """

        return (
            f"{self.nPoints} points ({len(self.faceIDs)} particle faces x {self.nQuadraturePoints} quadrature points)"
        )

    def currentPointCoordinates(self, model: FEModel) -> np.ndarray:
        """Where all points are, in the current configuration of the particles.

        Parameters
        ----------
        model
            The model; unused, the particles know where they are.

        Returns
        -------
        numpy.ndarray
            The coordinates, of shape (nPoints, nDim).
        """

        return self._coordinatesOfAllPoints()

    def locatePoints(self, points: np.ndarray) -> list:
        """Locate the given points: the nodes of the kernel functions of their particles, each once, and the sparse
        matrix of their interpolation vectors :math:`G(\\boldsymbol{x}_q)` at their current positions.

        Parameters
        ----------
        points
            The indices of the contact points to locate, ascending.

        Returns
        -------
        list
            The nodes of the kernel functions of the particles of the points.
        """

        particleOfPoint = self._particleOfFace[self._faceOfPoint[points]]
        particles = np.unique(particleOfPoint)
        localIndexOfNode = {}
        localNodesOfParticle = dict(
            zip(
                particles,
                locateTheNodesOfParticles(
                    [self._particles[particle].kernelFunctions for particle in particles], localIndexOfNode
                ),
            )
        )

        self._originsOfLocated = self._coordinatesOfAllPoints()[points] if len(points) else np.zeros((0, self.nDim))
        columns = [localNodesOfParticle[particle] for particle in particleOfPoint]
        values = [
            self._particles[particle].getInterpolationVectorLinearizedAboutCenter(x).ravel()
            for particle, x in zip(particleOfPoint, self._originsOfLocated)
        ]
        counts = np.array([len(c) for c in columns], dtype=np.intp)
        self._interpolationMatrix = csr_matrix(
            (
                np.concatenate(values) if values else np.zeros(0),
                np.concatenate(columns) if columns else np.zeros(0, dtype=np.intp),
                np.concatenate(([0], np.cumsum(counts))),
            ),
            shape=(len(points), len(localIndexOfNode)),
        )
        return list(localIndexOfNode)

    def originsOfLocatedPoints(self) -> np.ndarray:
        """:math:`\\boldsymbol{x}_q^n`, where the points were at the update, in the configuration of the particles
        then: where they are when the increment they move with is zero.

        Returns
        -------
        numpy.ndarray
            The coordinates, of shape (nLocated, nDim).
        """

        return self._originsOfLocated

    def displacementsOfLocatedPoints(self, U: np.ndarray, dU: np.ndarray) -> np.ndarray:
        """:math:`C\\, \\Delta\\boldsymbol{U}`, the displacement the points move with, the increment since the update.

        Parameters
        ----------
        U
            The total displacement of the slave part of the block; unused: the nodes of the kernel functions carry the
            increment.
        dU
            The increment of the slave part of the block.

        Returns
        -------
        numpy.ndarray
            The displacements, of shape (nLocated, nDim).
        """

        return self._interpolationMatrix @ dU.reshape((-1, self.nDim))

    def addForceResidual(self, P: np.ndarray, points: np.ndarray, forces: np.ndarray, directions: np.ndarray):
        """Add :math:`C^T (f\\, \\boldsymbol{n})`: the forces of the springs at the given points, distributed over the
        nodes of the kernel functions of their particles with their interpolation vectors.

        Parameters
        ----------
        P
            The slave part of the residual contribution; augmented.
        points
            The located points that carry a force.
        forces
            The forces of their springs.
        directions
            The unit directions of their springs.
        """

        forcesOnPoints = np.zeros((self._interpolationMatrix.shape[0], self.nDim))
        forcesOnPoints[points] = forces[:, None] * directions
        P.reshape((-1, self.nDim))[:] += self._interpolationMatrix.T @ forcesOnPoints

    def interpolationMatrixOfLocatedPoints(self) -> csr_matrix:
        """The sparse matrix :math:`C` of the interpolation vectors of the located points
        over the nodes of the kernel functions of their particles, made when they were located; see
        :meth:`~edelweissfe.constraints.base.contactpoints.ContactPoints.interpolationMatrixOfLocatedPoints`.

        Returns
        -------
        scipy.sparse.csr_matrix
            The matrix.
        """

        return self._interpolationMatrix

    def interpolationOfLocatedPoint(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        """The degrees of freedom of the nodes of the kernel functions of a located point's particle, and its
        interpolation vector over them.

        Parameters
        ----------
        k
            The index of the located point.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            The degrees of freedom and the interpolation vector.
        """

        C = self._interpolationMatrix
        nodes = C.indices[C.indptr[k] : C.indptr[k + 1]]
        return (nodes[:, None] * self.nDim + np.arange(self.nDim)).ravel(), C.data[C.indptr[k] : C.indptr[k + 1]]

    def searchLayout(self) -> dict[str, np.ndarray]:
        """The particles, their faces and the quadrature rule identify the contact points in a checkpoint.

        Returns
        -------
        dict[str, numpy.ndarray]
            The numbers of the particles of the faces, the face IDs, and the number of quadrature points per face.
        """

        return {
            "slaveParticles": np.array([particle.number for particle in self.particlesOfFaces], dtype=np.int64),
            "slaveFaces": self.faceIDs,
            "nQuadraturePoints": np.array([self.nQuadraturePoints], dtype=np.int64),
        }
