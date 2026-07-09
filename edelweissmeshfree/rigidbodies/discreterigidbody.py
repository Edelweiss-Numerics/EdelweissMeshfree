import numpy as np
from edelweissfe.config.phenomena import getFieldSize
from edelweissfe.utils.caseinsensitivedict import CaseInsensitiveDict
from edelweissfe.utils.inputlanguage import Module

module = Module("discreterigidbody", "A discrete rigid body entity.")
module.addRequiredArg("nSet", "The node set containing the surface nodes of the rigid body.", str)
module.addRequiredArg("referencePoint", "The node set containing the single reference point.", str)
keyword = "discreterigidbody"


class DiscreteRigidBody:
    """
    Discrete Rigid Body entity for explicit dynamics.

    It reads the total accumulated displacement (and rotation) of its Reference Point (RP)
    from the model's NodeField "U" entries, then kinematically updates all surface node
    coordinates and displacement field variables accordingly.
    """

    def __init__(self, name, model, *args, **kwargs):
        self.name = name
        self.model = model

        kwargs = CaseInsensitiveDict(kwargs)

        self.surfaceNodes = list(model.nodeSets[kwargs["nSet"]])
        rpNodeSet = model.nodeSets[kwargs["referencePoint"]]

        if len(rpNodeSet) > 1:
            raise ValueError("Reference point set must contain exactly one node!")

        self.rpNode = list(rpNodeSet)[0]
        self.domainSize = model.domainSize

        # Precompute initial relative positions of surface nodes w.r.t. the RP
        self.initialRelativePositions = np.array([n.coordinates - self.rpNode.coordinates for n in self.surfaceNodes])

    def _getFieldU(self, fieldName, node):
        """Safely retrieve the accumulated displacement/rotation for a single node.
        Returns a zero vector if the field entry "U" has not been written yet."""
        node_field = self.model.nodeFields.get(fieldName)
        if node_field is None or "U" not in node_field:
            return np.zeros(getFieldSize(fieldName, self.domainSize))
        return node_field.subset(node)["U"][0].copy()

    def updateKinematics(self, timeStep=None):
        """Update surface node coordinates and displacement fields based on the current RP state."""
        u_rp = self._getFieldU("displacement", self.rpNode)
        # Current RP position (initial + total accumulated displacement)
        rp_current = self.rpNode.coordinates + u_rp

        disp_field = self.model.nodeFields.get("displacement")
        has_disp = disp_field is not None and "U" in disp_field

        if self.domainSize == 3:
            theta = self._getFieldU("rotation", self.rpNode)
            R = self._getRotationMatrix3D(theta)
            for i, node in enumerate(self.surfaceNodes):
                X_rel = self.initialRelativePositions[i]
                node.coordinates[:] = rp_current + R.dot(X_rel)
                if has_disp:
                    idx = disp_field._indicesOfNodesInArray[node]
                    initial_coord = self.rpNode.coordinates + X_rel
                    disp_field["U"][idx] = node.coordinates - initial_coord
        else:
            # 2D: rotation is a scalar (rotation around Z)
            rot_u = self._getFieldU("rotation", self.rpNode)
            theta_z = rot_u[0] if len(rot_u) > 0 else 0.0
            c, s = np.cos(theta_z), np.sin(theta_z)
            R = np.array([[c, -s], [s, c]])
            for i, node in enumerate(self.surfaceNodes):
                X_rel = self.initialRelativePositions[i]
                node.coordinates[:] = rp_current + R.dot(X_rel)
                if has_disp:
                    idx = disp_field._indicesOfNodesInArray[node]
                    initial_coord = self.rpNode.coordinates + X_rel
                    disp_field["U"][idx] = node.coordinates - initial_coord

    def _getRotationMatrix3D(self, theta):
        """Construct a 3D rotation matrix from a rotation vector using Rodrigues' formula."""
        angle = np.linalg.norm(theta)
        if angle < 1e-12:
            return np.eye(3)

        axis = theta / angle
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])

        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
        return R
