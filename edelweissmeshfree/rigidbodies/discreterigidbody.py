import numpy as np
from edelweissfe.config.phenomena import getFieldSize
from edelweissfe.utils.caseinsensitivedict import CaseInsensitiveDict
from edelweissfe.utils.inputlanguage import Module

module = Module("discreterigidbody", "A discrete rigid body entity.")
module.addRequiredArg("nSet", "The node set containing the surface nodes of the rigid body.", str)
module.addRequiredArg("referencePoint", "The node set containing the single reference point.", str)
module.addOptionalArg("mass", "The mass of the rigid body.", float, None)
module.addOptionalArg("inertia", "The inertia tensor of the rigid body.", list, None)
module.addOptionalArg("initial_velocity", "The initial velocity vector.", list, None)
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

        self.mass = kwargs.get("mass")
        self.inertia = kwargs.get("inertia")
        self.initial_velocity = kwargs.get("initial_velocity")

        # Abstract the PointMass element
        self.point_mass_element = None
        if self.mass is not None:
            from edelweissfe.elements.pointmass import PointMass

            # generate a safe dummy elNumber (high number)
            el_num = max(model.elements.keys()) + 1 if model.elements else 1000000
            self.point_mass_element = PointMass(
                el_num, [self.rpNode], model, self.mass, self.inertia, self.initial_velocity
            )
            model.elements[el_num] = self.point_mass_element

    def _getFieldU(self, fieldName, node):
        """Safely retrieve the accumulated displacement/rotation for a single node.
        Returns a zero vector if the field entry "U" has not been written yet."""
        node_field = self.model.nodeFields.get(fieldName)
        if node_field is None or "U" not in node_field:
            return np.zeros(getFieldSize(fieldName, self.domainSize))
        return node_field.subset(node)["U"][0].copy()

    def getCurrentKinematics(self):
        """Returns the current RP displacement, rotation matrix, and initial RP coordinate."""
        u_rp = self._getFieldU("displacement", self.rpNode)
        
        if self.domainSize == 3:
            theta = self._getFieldU("rotation", self.rpNode)
            R = self._getRotationMatrix3D(theta)
        else:
            rot_u = self._getFieldU("rotation", self.rpNode)
            theta_z = rot_u[0] if len(rot_u) > 0 else 0.0
            c, s = np.cos(theta_z), np.sin(theta_z)
            R = np.array([[c, -s], [s, c]])
            
        return u_rp, R, self.rpNode.coordinates

    def updateKinematics(self, timeStep=None):
        """Update surface node coordinates and displacement fields based on the current RP state."""
        u_rp, R, rp_initial = self.getCurrentKinematics()
        
        # Current RP position (initial + total accumulated displacement)
        rp_current = rp_initial + u_rp

        disp_field = self.model.nodeFields.get("displacement")
        has_disp = disp_field is not None and "U" in disp_field

        # Vectorized coordinate calculation
        new_coords = rp_current + self.initialRelativePositions.dot(R.T)

        if has_disp:
            disp_u = disp_field["U"]

        # Assign back to nodes
        for i, node in enumerate(self.surfaceNodes):
            node.coordinates[:] = new_coords[i]
            if has_disp:
                idx = disp_field._indicesOfNodesInArray[node]
                disp_u[idx] = new_coords[i] - (self.rpNode.coordinates + self.initialRelativePositions[i])

    def querySurface(self, coords, proximity_factor=None):
        """
        Query the signed distance and outward normals of the surface mesh for the given global coordinates.
        If `proximity_factor` is provided, a broadphase AABB check is performed to skip distant points.
        Returns arrays of shape (N,) and (N, 3). Distances for points outside the proximity bounding box will be np.inf.
        """
        if not hasattr(self, "_query_engine"):
            from edelweissmeshfree.utils.discretesurfacequery import DiscreteSurfaceQuery
            if not hasattr(self, "surface_mesh"):
                raise RuntimeError("DiscreteRigidBody has no surface_mesh to query.")
            self._query_engine = DiscreteSurfaceQuery(mesh=self.surface_mesh)
            
        n_points = coords.shape[0]
        
        # Broadphase AABB Check
        if proximity_factor is not None:
            curr_min, curr_max = self.getAABB()
            aabb_min = curr_min - proximity_factor
            aabb_max = curr_max + proximity_factor
            
            in_aabb = np.all((coords >= aabb_min) & (coords <= aabb_max), axis=1)
            active_indices = np.where(in_aabb)[0]
            
            if len(active_indices) == 0:
                return np.full(n_points, np.inf), np.zeros((n_points, 3))
            
            coords_to_query = coords[active_indices]
        else:
            coords_to_query = coords
            active_indices = np.arange(n_points)
            
        u_rp, R, rp_initial = self.getCurrentKinematics()
        active_dists, active_normals = self._query_engine.query(
            coords_to_query,
            translation=u_rp,
            rotation_matrix=R,
            rotation_center=rp_initial
        )
        
        # Reassemble full arrays
        dists = np.full(n_points, np.inf)
        dists[active_indices] = active_dists
        
        normals = np.zeros((n_points, 3))
        normals[active_indices] = active_normals
        
        return dists, normals

    def getAABB(self):
        """Returns the current Axis-Aligned Bounding Box (min, max) of the surface."""
        coords = np.array([n.coordinates for n in self.surfaceNodes])
        return np.min(coords, axis=0), np.max(coords, axis=0)

    def _getRotationMatrix3D(self, theta):
        """Construct a 3D rotation matrix from a rotation vector using Rodrigues' formula."""
        angle = np.linalg.norm(theta)
        if angle < 1e-12:
            return np.eye(3)

        axis = theta / angle
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])

        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
        return R

    @classmethod
    def from_mesh_file(
        cls,
        name: str,
        model,
        filename: str,
        translation: np.ndarray = None,
        density: float = None,
        mass: float = None,
        inertia: list = None,
        initial_velocity: list = None,
        rp_coordinate: np.ndarray = None,
        start_label: int = 1000000,
    ):
        """
        Creates a Discrete Rigid Body directly from a mesh file.
        Encapsulates node creation, elements generation, and RP kinematics.
        """
        import pyvista as pv
        from edelweissfe.points.node import Node
        from edelweissfe.sets.nodeset import NodeSet
        from edelweissfe.variables.fieldvariable import FieldVariable
        from edelweissfe.elements.discreterigid import DiscreteRigidElement
        from edelweissfe.sets.elementset import ElementSet

        mesh = pv.read(filename)
        if isinstance(mesh, pv.MultiBlock):
            mesh = mesh.combine()

        points = mesh.points.copy()
        if translation is not None:
            points += np.array(translation)

        mesh.points = points
        surf = mesh.extract_surface()
        surf.compute_normals(cell_normals=True, point_normals=False, inplace=True)

        # Handle density -> mass & inertia
        if density is not None:
            volume = surf.volume
            if volume == 0.0:
                volume = mesh.volume
            mass = volume * density
            # Simple fallback for inertia if none provided
            if inertia is None:
                inertia = [mass, mass, mass]

        # Extract faces
        cells = surf.cells if hasattr(surf, 'cells') else mesh.cells
        faces = []
        i = 0
        while i < len(cells):
            n = cells[i]
            faces.append(cells[i + 1 : i + 1 + n])
            i += 1 + n

        # Generate Node Entities
        rigid_nodes = []
        for i, pt in enumerate(points):
            n = Node(start_label + i, pt.copy())
            model.nodes[n.label] = n
            n.fields["displacement"] = FieldVariable(n, "displacement")
            rigid_nodes.append(n)

        nset_name = f"{name}_surface_nodes"
        model.nodeSets[nset_name] = NodeSet(nset_name, rigid_nodes)

        # Generate Element Entities
        rigid_elements = []
        for i, face in enumerate(faces):
            if len(face) == 4:
                el_nodes = [rigid_nodes[face[0]], rigid_nodes[face[1]], rigid_nodes[face[2]], rigid_nodes[face[3]]]
                el = DiscreteRigidElement(start_label + i, el_nodes, model, "quad4")
            else:
                el_nodes = [rigid_nodes[face[0]], rigid_nodes[face[1]], rigid_nodes[face[2]]]
                el = DiscreteRigidElement(start_label + i, el_nodes, model, "tria3")
            model.elements[el.elNumber] = el
            rigid_elements.append(el)
        
        eset_name = f"{name}_surface"
        model.elementSets[eset_name] = ElementSet(eset_name, rigid_elements)

        # Reference Point
        if rp_coordinate is None:
            rp_coordinate = surf.center_of_mass()

        rp = Node(start_label + 999999, np.array(rp_coordinate))
        model.nodes[rp.label] = rp
        
        rp_nset_name = f"{name}_rp"
        model.nodeSets[rp_nset_name] = NodeSet(rp_nset_name, [rp])
        rp.fields["displacement"] = FieldVariable(rp, "displacement")
        rp.fields["rotation"] = FieldVariable(rp, "rotation")

        if "all" in model.nodeSets:
            all_nodes = list(model.nodeSets["all"])
            all_nodes.extend(rigid_nodes)
            all_nodes.append(rp)
            model.nodeSets["all"] = NodeSet("all", all_nodes)

        # Instantiate self
        instance = cls(
            name,
            model,
            nSet=nset_name,
            referencePoint=rp_nset_name,
            mass=mass,
            inertia=inertia,
            initial_velocity=initial_velocity
        )
        
        # Store pyvista surface internally so query engine can reuse it without disk reads
        instance.surface_mesh = surf
        
        model.discreteRigidBodies[name] = instance
        return instance
