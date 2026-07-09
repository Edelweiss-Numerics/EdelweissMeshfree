import pyvista as pv

mesh = pv.read("rigid_body.exo").extract_surface()
mesh.compute_normals(cell_normals=True, point_normals=False, inplace=True)
import vtk

implicit_dist = vtk.vtkImplicitPolyDataDistance()
implicit_dist.SetInput(mesh)
pt2 = [0.0, 10.07, 0.0]
closest_grad = [0.0, 0.0, 0.0]
implicit_dist.EvaluateGradient(pt2, closest_grad)
print(f"Gradient (outward normal): {closest_grad}")
