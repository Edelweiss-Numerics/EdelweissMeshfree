Code organization
=================

The code is organized around major simulation responsibilities in the
``edelweissmeshfree`` package:

* ``models``: central model objects and simulation state containers.
* ``cells`` / ``cellelements``: background-grid entities and their wrappers.
* ``materialpoints`` / ``particles``: moving point entities and wrappers.
* ``mpmmanagers`` / ``particlemanagers``: update and search orchestration for
  material points and particles.
* ``stepactions`` / ``constraints`` / ``solvers``: analysis procedures,
  boundary actions, and nonlinear solution strategies.
* ``generators``: helper modules to construct cells, points, and sets.
* ``meshfree``: meshfree approximation, kernel, and integration utilities.
* ``fields`` / ``fieldoutput`` / ``outputmanagers``: field storage and result
  export.
* ``numerics``: low-level numerical infrastructure such as dof management.
* ``config`` and ``sets``: registration helpers and grouping containers used
  by model assembly.

This structure is designed so that new formulations can usually be added by
extending one focused subpackage while reusing the existing solver, output, and
model infrastructure.
