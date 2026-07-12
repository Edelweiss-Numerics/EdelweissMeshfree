import edelweissfe.utils.performancetiming as performancetiming
import numpy as np
from edelweissfe.config.phenomena import getFieldSize
from edelweissfe.utils.caseinsensitivedict import CaseInsensitiveDict
from edelweissfe.utils.inputlanguage import Module

from edelweissmeshfree.rigidbodies.rigidbody import RigidBody

module = Module("discreterigidbody", "A discrete rigid body entity.")
module.addRequiredArg("nSet", "The node set containing the surface nodes of the rigid body.", str)
module.addRequiredArg("referencePoint", "The node set containing the single reference point.", str)
module.addOptionalArg("mass", "The mass of the rigid body.", float, None)
module.addOptionalArg("inertia", "The inertia tensor of the rigid body.", list, None)
module.addOptionalArg("initial_velocity", "The initial velocity vector.", list, None)
keyword = "discreterigidbody"


class DiscreteRigidBody(RigidBody):
    """
    Discrete Rigid Body entity for explicit dynamics.

    It reads the total accumulated displacement (and rotation) of its Reference Point (RP)
    from the model's NodeField "U" entries, then kinematically updates all surface node
    coordinates and displacement field variables accordingly.
    """

    def __init__(self, name, model, *args, **kwargs):
        self.name = name
        self.model = model

        model.rigidBodies[self.name] = self

        kwargs = CaseInsensitiveDict(kwargs)

        self.surfaceNodes = list(model.nodeSets[kwargs["nSet"]])
        rpNodeSet = model.nodeSets[kwargs["referencePoint"]]

        if len(rpNodeSet) > 1:
            raise ValueError("Reference point set must contain exactly one node!")

        self.rpNode = list(rpNodeSet)[0]
        self.domainSize = model.domainSize

        # Initialize default explicit velocities to avoid hasattr/getattr on RP node
        self.rpNode.current_velocity = np.zeros(self.domainSize)
        if self.domainSize == 3:
            self.rpNode.current_angular_velocity = np.zeros(3)

        self.surface_mesh = None
        self._query_engine = None

        # Precompute initial relative positions of surface nodes w.r.t. the RP
        self.initialRelativePositions = np.array([n.coordinates - self.rpNode.coordinates for n in self.surfaceNodes])

        self.mass = kwargs.get("mass")
        self.inertia = kwargs.get("inertia")
        self.initial_velocity = kwargs.get("initial_velocity")

        # Abstract the PointMass element
        self.point_mass_element = None
        if self.mass is not None:
            from edelweissfe.elements.pointmass import PointMass

            el_num = max(model.elements.keys()) + 1 if model.elements else 1
            self.point_mass_element = PointMass(
                el_num, [self.rpNode], model, self.mass, self.inertia, self.initial_velocity
            )
            model.elements[el_num] = self.point_mass_element

    def _getFieldU(self, fieldName, node):
        """Safely retrieve the accumulated displacement/rotation for a single node.
        Returns a zero vector if the field entry "U" has not been written yet."""
        if fieldName not in node.fields:
            return np.zeros(getFieldSize(fieldName, self.domainSize))
        node_field = self.model.nodeFields.get(fieldName)
        if node_field is None or "U" not in node_field:
            return np.zeros(getFieldSize(fieldName, self.domainSize))
        return node_field.subset(node)["U"][0].copy()

    def getCurrentKinematics(self):
        """
        Returns the current reference point displacement, rotation matrix, and initial coordinates.

        Returns
        -------
        u_rp : numpy.ndarray
            The current accumulated displacement vector of the reference point.
        R : numpy.ndarray
            The 3x3 rotation matrix representing the current orientation of the rigid body.
        rp_initial : numpy.ndarray
            The initial global coordinates of the reference point.
        """
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

    @performancetiming.timeit("rigid body update kinematics")
    def updateKinematics(self, timeStep=None):
        """
        Update surface node coordinates and displacement fields based on the current Reference Point state.

        This method applies the current accumulated displacement and rotation from the
        reference point to all surface nodes belonging to the discrete rigid body.

        Parameters
        ----------
        timeStep : TimeStep, optional
            The current time step information.
        """
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

    @performancetiming.timeit("rigid body query surface")
    def querySurface(self, coords, proximity_factor=None):
        """
        Query the signed distance and outward normals of the surface mesh for the given global coordinates.

        If `proximity_factor` is provided, a broadphase Axis-Aligned Bounding Box (AABB)
        check is performed internally to efficiently skip distant points.

        Parameters
        ----------
        coords : numpy.ndarray
            An array of shape (N, 3) containing the global query coordinates.
        proximity_factor : float, optional
            A distance padding added to the rigid body's AABB. Points outside
            the inflated box are culled from the expensive VTK surface query.

        Returns
        -------
        dists : numpy.ndarray
            Array of shape (N,) containing the signed distances. Negative values
            indicate penetration. Distances for points outside the proximity bounding
            box will be evaluated as `np.inf`.
        normals : numpy.ndarray
            Array of shape (N, 3) containing the outward normal vectors on the surface.
        """
        if self._query_engine is None:
            from edelweissmeshfree.utils.discretesurfacequery import (
                DiscreteSurfaceQuery,
            )

            if self.surface_mesh is None:
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
            coords_to_query, translation=u_rp, rotation_matrix=R, rotation_center=rp_initial
        )

        # Reassemble full arrays
        dists = np.full(n_points, np.inf)
        dists[active_indices] = active_dists

        normals = np.zeros((n_points, 3))
        normals[active_indices] = active_normals

        return dists, normals

    def getAABB(self):
        """
        Returns the current Axis-Aligned Bounding Box (AABB) of the rigid body surface.

        Returns
        -------
        aabb_min : numpy.ndarray
            The (x, y, z) minimum coordinate bounds of the current surface nodes.
        aabb_max : numpy.ndarray
            The (x, y, z) maximum coordinate bounds of the current surface nodes.
        """
        coords = np.array([n.coordinates for n in self.surfaceNodes])
        return np.min(coords, axis=0), np.max(coords, axis=0)

    def _getRotationMatrix3D(self, theta):
        """
        Construct a 3D rotation matrix from a rotation vector using Rodrigues' formula.

        Parameters
        ----------
        theta : numpy.ndarray
            A 3D rotation vector where the direction indicates the axis of rotation
            and the magnitude indicates the angle in radians.

        Returns
        -------
        R : numpy.ndarray
            A 3x3 rotation matrix.
        """
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
        start_label: int = None,
    ):
        """
        Creates a Discrete Rigid Body directly from a mesh file.

        Encapsulates node creation, surface element generation, and reference point
        (RP) kinematics setup automatically. Also computes basic mass and inertia
        properties based on the mesh volume if a density is supplied.

        Parameters
        ----------
        name : str
            The identifier name for the discrete rigid body.
        model : MPMModel
            The current model instance.
        filename : str
            The file path to the surface mesh (e.g., STL, OBJ).
        translation : numpy.ndarray, optional
            A 3D vector to translate the mesh globally upon initialization.
        density : float, optional
            The mass density of the rigid body. Used to compute mass and inertia.
        mass : float, optional
            The total mass of the rigid body. Overrides density-based computation.
        inertia : list, optional
            The 3D diagonal moments of inertia. Overrides density-based computation.
        initial_velocity : list, optional
            The initial velocity vector [vx, vy, vz].
        rp_coordinate : numpy.ndarray, optional
            The explicit global coordinates for the Reference Point. If None, it
            defaults to the center of mass of the surface mesh.
        start_label : int, optional
            The starting index label for newly generated nodes and elements.

        Returns
        -------
        instance : DiscreteRigidBody
            The fully initialized discrete rigid body entity.
        """
        import pyvista as pv
        from edelweissfe.elements.discreterigid import DiscreteRigidElement
        from edelweissfe.points.node import Node
        from edelweissfe.sets.elementset import ElementSet
        from edelweissfe.sets.nodeset import NodeSet

        filename_lower = filename.lower()
        if filename_lower.endswith(".exo") or filename_lower.endswith(".nc"):
            import netCDF4

            nc = netCDF4.Dataset(filename, "r")
            try:
                x = nc.variables["coordx"][:]
                y = nc.variables["coordy"][:]
                z = nc.variables["coordz"][:] if "coordz" in nc.variables else np.zeros_like(x)
                points = np.column_stack((x, y, z))

                if translation is not None:
                    points += np.array(translation)

                # Assume rigid body surface mesh is entirely in connect1
                if "connect1" not in nc.variables:
                    raise ValueError("No connect1 variable found in NetCDF/Exo file.")

                conn = nc.variables["connect1"][:] - 1  # 0-indexed
            finally:
                nc.close()

            num_elems, num_nodes_per_elem = conn.shape

            faces = []
            element_types = []
            pv_faces = []

            for i in range(num_elems):
                face = conn[i]
                faces.append(face)

                pv_faces.append(num_nodes_per_elem)
                pv_faces.extend(face)

                if num_nodes_per_elem == 3:
                    element_types.append("tria3")
                elif num_nodes_per_elem == 4:
                    element_types.append("quad4")
                else:
                    raise ValueError(f"Unsupported number of nodes {num_nodes_per_elem} for surface mesh.")

            surf = pv.PolyData(points, np.array(pv_faces))
            surf.compute_normals(cell_normals=True, point_normals=False, inplace=True)
            mesh = surf  # for volume fallback

        else:
            mesh = pv.read(filename)
            if isinstance(mesh, pv.MultiBlock):
                mesh = mesh.combine()

            points = mesh.points.copy()
            if translation is not None:
                points += np.array(translation)

            mesh.points = points
            surf = mesh.extract_surface()
            surf.compute_normals(cell_normals=True, point_normals=False, inplace=True)

            # Extract faces and map to EdelweissFE element types using actual VTK cell types
            cells = surf.cells
            faces = []
            element_types = []

            i = 0
            cell_idx = 0
            while i < len(cells):
                n = cells[i]
                faces.append(cells[i + 1 : i + 1 + n])

                vtk_type = surf.GetCellType(cell_idx)

                # Map VTK cell types to EdelweissFE element types
                # 5 = VTK_TRIANGLE, 9 = VTK_QUAD, 7 = VTK_POLYGON
                if vtk_type == 5:
                    element_types.append("tria3")
                elif vtk_type == 9:
                    element_types.append("quad4")
                elif vtk_type == 7:
                    if n == 3:
                        element_types.append("tria3")
                    elif n == 4:
                        element_types.append("quad4")
                    else:
                        raise ValueError(f"Unsupported VTK_POLYGON with {n} nodes for Discrete Rigid Body.")
                else:
                    # Fallback based on node count for strange/custom VTK cell types
                    if n == 3:
                        element_types.append("tria3")
                    elif n == 4:
                        element_types.append("quad4")
                    else:
                        raise ValueError(f"Unsupported VTK cell type {vtk_type} with {n} nodes.")

                i += 1 + n
                cell_idx += 1

        # Handle density -> mass & inertia
        if density is not None:
            volume = surf.volume
            if volume == 0.0:
                volume = getattr(mesh, "volume", 0.0)
            mass = volume * density
            # Simple fallback for inertia if none provided
            if inertia is None:
                inertia = [mass, mass, mass]

        # Generate Node Entities
        rigid_nodes = []
        node_id = start_label if start_label is not None else (max(model.nodes.keys()) + 1 if model.nodes else 1)
        for i, pt in enumerate(points):
            n = Node(node_id, pt.copy())
            model.nodes[n.label] = n
            rigid_nodes.append(n)
            node_id += 1

        nset_name = f"{name}_surface_nodes"
        model.nodeSets[nset_name] = NodeSet(nset_name, rigid_nodes)

        # Generate Element Entities
        rigid_elements = []
        el_id = start_label if start_label is not None else (max(model.elements.keys()) + 1 if model.elements else 1)
        for i, face in enumerate(faces):
            el_type = element_types[i]
            el_nodes = [rigid_nodes[idx] for idx in face]
            el = DiscreteRigidElement(el_id, el_nodes, model, el_type)

            model.elements[el.elNumber] = el
            rigid_elements.append(el)
            el_id += 1

        eset_name = f"{name}_surface"
        model.elementSets[eset_name] = ElementSet(eset_name, rigid_elements)

        # Reference Point
        if rp_coordinate is None:
            rp_coordinate = surf.center_of_mass()

        rp = Node(node_id, np.array(rp_coordinate))
        model.nodes[rp.label] = rp

        rp_nset_name = f"{name}_rp"
        model.nodeSets[rp_nset_name] = NodeSet(rp_nset_name, [rp])

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
            initial_velocity=initial_velocity,
        )

        # Store pyvista surface internally so query engine can reuse it without disk reads
        instance.surface_mesh = surf

        model.rigidBodies[name] = instance
        return instance
