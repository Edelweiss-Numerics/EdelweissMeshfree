import numpy as np
import pyvista as pv
from edelweissmeshfree.utils.discretesurfacequery import DiscreteSurfaceQuery

def run_test():
    print("Initializing DiscreteSurfaceQuery with rigid_body.exo...")
    query_engine = DiscreteSurfaceQuery("rigid_body.exo")
    
    print(f"Rigid body mesh loaded with {query_engine.mesh.n_points} points and {query_engine.mesh.n_cells} faces.")
    
    # Generate some random dummy points, e.g. 100 points around the rigid body
    # The block was at [0,0,0] to [10,10,10] with centers around [5,5,5].
    # But let's just make 10 random points
    coords = np.random.rand(10, 3) * 10
    
    print("Executing vectorized query on 10 dummy particle coordinates...")
    dists, normals = query_engine.query(coords)
    
    print("Query successful!")
    print(f"Sample dummy particle 0 coordinate: {coords[0]}")
    print(f"Sample dummy particle 0 distance to rigid body: {dists[0]}")
    print(f"Sample dummy particle 0 rigid body surface normal: {normals[0]}")
    
    print(f"Min distance: {np.min(dists)}, Max distance: {np.max(dists)}")

if __name__ == "__main__":
    run_test()
