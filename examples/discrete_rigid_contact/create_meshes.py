import os
import sys


def main():
    cubit_dir = os.environ.get("CUBIT_BIN_DIR")
    if cubit_dir:
        if cubit_dir not in sys.path:
            sys.path.append(cubit_dir)
    else:
        fallback = "/home/matthias/Downloads/Coreform-Cubit-2026.6/bin"
        if os.path.exists(fallback):
            if fallback not in sys.path:
                sys.path.append(fallback)
            cubit_dir = fallback
        else:
            cubit_dir = "Not Set"

    try:
        import cubit
    except ImportError:
        print(
            f"Error: Could not import cubit. Please ensure Coreform Cubit is installed "
            f"and set the CUBIT_BIN_DIR environment variable to its bin directory.\n"
            f"Current directory path: {cubit_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    cubit.init(["cubit", "-nographics", "-nojournal"])

    # Make sure we are in the script's directory for saving files
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    print("Generating particles (solid block)...")
    cubit.cmd("reset")
    cubit.cmd("brick x 10 y 10 z 10")
    # Block is at [ -5, 5 ] in X, Y, Z
    cubit.cmd("volume 1 size 1")
    cubit.cmd("mesh volume 1")
    cubit.cmd("block 1 volume 1")
    cubit.cmd("export genesis 'particles.exo' overwrite")
    print("Exported particles.exo")

    print("Generating rigid body (cylindrical surface mesh)...")
    cubit.cmd("reset")
    cubit.cmd("cylinder radius 5 height 20")
    # Cylinder is originally centered at origin. Move it so it doesn't overlap with the block's space.
    # We move it in Y direction. The block extends from Y=-5 to 5.
    # Cylinder radius is 5. So if we place cylinder at Y=15, its surface is at Y=10.
    # So there is a gap of 5 between them.
    cubit.cmd("move Volume 1 y 15")

    # Mesh the surfaces
    cubit.cmd("surface all size 1")
    cubit.cmd("mesh surface all")

    # Create block from all surfaces. This creates 2D shell elements in the exodus file.
    cubit.cmd("block 1 surface all")
    cubit.cmd("export genesis 'rigid_body.exo' overwrite")
    print("Exported rigid_body.exo")


if __name__ == "__main__":
    main()
