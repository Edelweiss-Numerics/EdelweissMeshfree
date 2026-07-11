import csv
import glob
import os
import sys

import numpy as np
import pyvista as pv


def animate_case(case_file):
    if not os.path.exists(case_file):
        print(f"Error: Specified case file '{case_file}' does not exist.")
        sys.exit(1)

    print(f"Reading Ensight Gold case file: {case_file}...")
    reader = pv.get_reader(case_file)
    reader.set_active_time_set(1)
    time_values = reader.time_values
    print(f"Found {len(time_values)} time steps (t = {time_values[0]:.4f} to {time_values[-1]:.4f} s).")

    # ------------------------------------------------------------------ helpers
    def read_step(t_val):
        """Read one time step and return (rigid_surface, particles) blocks."""
        reader.set_active_time_value(t_val)
        mesh = reader.read()
        return mesh["rigid_body_surface"], mesh["PSET_mesh_particles_all"], mesh

    def warp(block, vector_name):
        """Return a new mesh with points shifted by the named point-data vector."""
        if vector_name in block.point_data:
            pts = block.points + block.point_data[vector_name]
        else:
            pts = block.points
        out = block.copy()
        out.points = pts
        return out

    # ------------------------------------------------------------------ initial geometry
    pv.set_plot_theme("document")
    off_screen = "HEADLESS_TEST" in os.environ
    plotter = pv.Plotter(shape=(1, 2), notebook=False, off_screen=off_screen, window_size=[1600, 800])

    plotter.subplot(0, 0)

    rigid0, particles0, mesh0 = read_step(time_values[0])
    warped_rigid0 = warp(rigid0, "vertex_displacements")
    warped_particles0 = warp(particles0, "vertex_displacements")

    if "displacement" in particles0.cell_data:
        vel_mag0 = np.linalg.norm(particles0.cell_data["displacement"], axis=1)
        warped_particles0.cell_data["displacement_magnitude"] = vel_mag0
    else:
        warped_particles0.cell_data["displacement_magnitude"] = np.zeros(warped_particles0.n_cells)

    rigid_actor = plotter.add_mesh(
        warped_rigid0,
        color="#a0a0a0",
        opacity=0.5,
        show_edges=True,
        edge_color="#404040",
        line_width=1,
        ambient=0.3,
        diffuse=0.7,
        specular=0.2,
    )

    # Scan max displacement magnitude across all steps for optimal clim scaling
    max_disp = 0.1
    for t in time_values:
        _, p_temp, _ = read_step(t)
        if "vertex_displacements" in p_temp.point_data:
            mags = np.linalg.norm(p_temp.point_data["vertex_displacements"], axis=1)
            if len(mags) > 0:
                max_disp = max(max_disp, np.max(mags))
    print(f"Dynamic colormap range: [0.0, {max_disp:.4f}] m")

    particle_actor = plotter.add_mesh(
        warped_particles0,
        scalars="displacement_magnitude",
        cmap="turbo",
        clim=[0.0, max_disp],
        show_edges=False,
        scalar_bar_args={
            "title": "Displacement Magnitude (m)",
            "vertical": True,
            "position_x": 0.88,
            "position_y": 0.15,
            "height": 0.7,
            "width": 0.08,
        },
        ambient=0.4,
        diffuse=0.6,
    )

    plotter.add_text("Explicit Contact Dynamics (Quasistatic)", font_size=18, color="darkblue", position="upper_edge")

    plotter.subplot(0, 1)
    chart = pv.Chart2D()
    chart.title = "Forces vs Time"
    chart.x_label = "Time (s)"
    chart.y_label = "Force (N)"
    chart.background_color = (1.0, 1.0, 1.0, 0.5)
    plotter.add_chart(chart)

    t_plot = []
    fc_plot = []
    fb_plot = []
    fct_plot = []
    fbt_plot = []

    # ------------------------------------------------------------------ camera / overlays
    plotter.subplot(0, 0)
    plotter.camera_position = [
        (35.0, 15.0, 35.0),
        (0.0, 2.5, 0.0),
        (0.0, 1.0, 0.0),
    ]
    plotter.add_axes()
    plotter.show_grid()

    text_actor = plotter.add_text(
        f"Time: {time_values[0]:.4f} s",
        position="upper_left",
        font_size=14,
        color="black",
        shadow=True,
    )

    print("Saving animation to quasistatic_animation.mp4...")
    plotter.open_movie("quasistatic_animation.mp4", framerate=15)

    if not off_screen:
        plotter.show(interactive_update=True)

    # ------------------------------------------------------------------ animation loop
    for i, time_val in enumerate(time_values):
        rigid, particles, mesh = read_step(time_val)

        # Update rigid body exactly as recorded in the case file
        if "vertex_displacements" in rigid.point_data:
            wr = warp(rigid, "vertex_displacements")
            rigid_actor.mapper.dataset.copy_from(wr)

        # Update particle block exactly as recorded in the case file
        if "vertex_displacements" in particles.point_data:
            particle_actor.mapper.dataset.points = particles.points + particles.point_data["vertex_displacements"]

        if "displacement" in particles.cell_data:
            vel_mag = np.linalg.norm(particles.cell_data["displacement"], axis=1)
            particle_actor.mapper.dataset.cell_data["displacement_magnitude"] = vel_mag

        # Extract forces
        fn = 0.0
        ft = 0.0
        fb = 0.0
        fbt = 0.0

        if "NSET_rigid_body_rp" in mesh.keys():
            rp = mesh["NSET_rigid_body_rp"]
            if "normal_force" in rp.point_data:
                fn = np.linalg.norm(rp.point_data["normal_force"][0])
            if "friction_force" in rp.point_data:
                ft = np.linalg.norm(rp.point_data["friction_force"][0])

        if "NSET_bottom_nodes" in mesh.keys():
            bottom = mesh["NSET_bottom_nodes"]
            if "bottom_normal_force" in bottom.point_data:
                fb = np.sum(bottom.point_data["bottom_normal_force"])
            if "bottom_tangential_force" in bottom.point_data:
                fbt = np.sum(bottom.point_data["bottom_tangential_force"])

        t_plot.append(time_val)
        fc_plot.append(fn)
        fb_plot.append(fb)
        fct_plot.append(ft)
        fbt_plot.append(fbt)

        chart.clear()
        chart.line(t_plot, fc_plot, color="r", width=2, label="Contact Force (Normal)")
        chart.line(t_plot, fb_plot, color="b", width=2, label="Bottom BC Reaction (Normal)")
        chart.line(t_plot, fct_plot, color="g", width=2, label="Contact Friction (Tangential)", style="--")
        chart.line(t_plot, fbt_plot, color="orange", width=2, label="Bottom BC Reaction (Tangential)", style="--")

        max_f = max(max(fc_plot) if fc_plot else 0, max(fb_plot) if fb_plot else 0, max(fct_plot) if fct_plot else 0)
        if max_f > 0:
            chart.y_axis.range = [0, max_f * 1.2]

        text_actor.set_text(0, f"Time: {time_val:.4f} s  (Step {i + 1}/{len(time_values)})")

        plotter.render()
        plotter.write_frame()
        if not off_screen:
            import time

            time.sleep(0.05)

    print("Animation completed.")
    if not off_screen:
        plotter.show(interactive=True)
    plotter.close()

    # Export to CSV for validation
    csv_file = "validation_forces.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["time", "contact_normal_force", "contact_friction_force", "bottom_normal_force", "bottom_tangential_force"]
        )
        for i in range(len(t_plot)):
            writer.writerow([t_plot[i], fc_plot[i], fct_plot[i], fb_plot[i], fbt_plot[i]])
    print(f"Exported validation data to {csv_file}")


if __name__ == "__main__":
    case_files = sorted(glob.glob("quasistatic_sim_out_*.case"))
    if not case_files:
        print("Error: No case files found.")
        sys.exit(1)

    latest_case = case_files[-1]
    animate_case(latest_case)
