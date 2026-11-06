# -*- coding: utf-8 -*-
#  ---------------------------------------------------------------------
#
#  _____    _      _              _         __  __ ____  __  __
# | ____|__| | ___| |_      _____(_)___ ___|  \/  |  _ \|  \/  |
# |  _| / _` |/ _ \ \ \ /\ / / _ \ / __/ __| |\/| | |_) | |\/| |
# | |__| (_| |  __/ |\ V  V /  __/ \__ \__ \ |  | |  __/| |  | |
# |_____\__,_|\___|_| \_/\_/ \___|_|___/___/_|  |_|_|   |_|  |_|
#
#
#  Unit of Strength of Materials and Structural Analysis
#  University of Innsbruck,
#  2023 - today
#
#  Matthias Neuner matthias.neuner@uibk.ac.at
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
"""No Python file of the examples and of the package compiles with a SyntaxWarning. In particular, no ``return``,
``break`` or ``continue`` leaves a ``finally`` block (PEP 765): it discards the exception in flight, so an example that
fails would compare whatever state its model was left in with the gold, and a missing Marmot module would never become
a skip."""

import pathlib
import warnings

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_no_python_file_compiles_with_a_syntax_warning():
    warned = []
    for directory in ("examples", "edelweissmeshfree"):
        for path in sorted((ROOT / directory).rglob("*.py")):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", SyntaxWarning)
                compile(path.read_text(), str(path), "exec")
            warned += ["{:}:{:}: {:}".format(path.relative_to(ROOT), w.lineno, w.message) for w in caught]

    assert not warned, "\n".join(warned)
