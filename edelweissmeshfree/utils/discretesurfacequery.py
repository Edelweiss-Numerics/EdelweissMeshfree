import numpy as np
import pyvista as pv
import vtk


class DiscreteSurfaceQuery:
    def __init__(self, filename: str = None, mesh: pv.PolyData = None, initial_offset: np.ndarray = None):
        """
        Initializes the query engine by loading an Exodus mesh or using a provided PyVista mesh.
        Uses static VTK locators and evaluators to avoid memory leaks.

        Parameters
        ----------
        filename : str, optional
            Path to the Exodus mesh file for the rigid body.
        mesh : pv.PolyData, optional
            A direct PyVista mesh object to use. If provided, `filename` is ignored.
        initial_offset : np.ndarray, optional
            A translation vector applied to the mesh points before building
            the VTK locators.
        """
        if mesh is not None:
            self.mesh = mesh
        elif filename is not None:
            self.mesh = pv.read(filename)
            if isinstance(self.mesh, pv.MultiBlock):
                self.mesh = self.mesh.combine()
        else:
            raise ValueError("Must provide either 'filename' or 'mesh'")

        # Extract the outer surface to ensure we have a PolyData object
        self.mesh = self.mesh.extract_surface()

        # Apply the initial offset so that the contact surface sits at its
        # physical starting position (consistent with the visualization nodes).
        if initial_offset is not None:
            self.mesh.points = self.mesh.points + np.asarray(initial_offset, dtype=np.float64)

        # Ensure outward normals are computed for each cell
        self.mesh.compute_normals(cell_normals=True, point_normals=False, inplace=True)
        self.mesh_cell_normals = np.array(self.mesh.cell_normals)

        # Initialize implicit distance evaluator once to avoid memory leaks
        self.implicit_dist = vtk.vtkImplicitPolyDataDistance()
        self.implicit_dist.SetInput(self.mesh)

        # Initialize cell locator once to avoid memory leaks
        self.locator = vtk.vtkStaticCellLocator()
        self.locator.SetDataSet(self.mesh)
        self.locator.BuildLocator()

    def query(self, coords: np.ndarray, rigid_body_translation: np.ndarray = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Vectorized query to compute signed distance and closest face normals for an array of points.
        Uses cached VTK instances to prevent memory leaks.

        Parameters
        ----------
        coords : np.ndarray
            Array of shape (N, 3) containing the query coordinates.
        rigid_body_translation : np.ndarray, optional
            A translation vector to apply to the rigid body. Internally this inversely translates the query coords.

        Returns
        -------
        dists : np.ndarray
            Array of shape (N,) containing the signed distance (negative = inside/penetration).
        normals : np.ndarray
            Array of shape (N, 3) containing the outward normal vectors.
        """
        if rigid_body_translation is not None:
            local_coords = coords - rigid_body_translation
        else:
            local_coords = coords

        n_points = local_coords.shape[0]
        dists = np.empty(n_points, dtype=np.float64)
        closest_cells = np.empty(n_points, dtype=np.int64)

        # Pre-allocate reusable VTK out arguments for speed
        closest_point = [0.0, 0.0, 0.0]
        sub_id = vtk.reference(0)
        dist2 = vtk.reference(0.0)

        for i in range(n_points):
            pt = local_coords[i]
            dists[i] = self.implicit_dist.EvaluateFunction(pt)

            cell_id = vtk.reference(0)
            self.locator.FindClosestPoint(pt, closest_point, cell_id, sub_id, dist2)
            closest_cells[i] = int(cell_id)

        normals = self.mesh_cell_normals[closest_cells]

        return dists, normals
