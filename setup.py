# -*- coding: utf-8 -*-
#  ---------------------------------------------------------------------
#
#  _____    _      _              _
# | ____|__| | ___| |_      _____(_)___ ___
# |  _| / _` |/ _ \ \ \ /\ / / _ \ / __/ __|
# | |__| (_| |  __/ |\ V  V /  __/ \__ \__ \
# |_____\__,_|\___|_| \_/\_/_\___|_|___/___/
# |  \/  | ___  ___| |__  / _|_ __ ___  ___
# | |\/| |/ _ \/ __| '_ \| |_| '__/ _ \/ _ \
# | |  | |  __/\__ \ | | |  _| | |  __/  __/
# |_|  |_|\___||___/_| |_|_| |_|  \___|\___|
#
#
#  Unit of Strength of Materials and Structural Analysis
#  University of Innsbruck,
#
#  Research Group for Computational Mechanics of Materials
#  Institute of Structural Engineering, BOKU University, Vienna
#
#  2023 - today
#
#  Matthias Neuner |  matthias.neuner@boku.ac.at
#  Thomas Mader    |  thomas.mader@bokut.ac.at
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
import os
import sys
from os.path import expanduser, join

import numpy
from Cython.Build import build_ext, cythonize
from setuptools import setup
from setuptools.extension import Extension

directives = {
    "boundscheck": False,
    "wraparound": False,
    "nonecheck": False,
    "initializedcheck": False,
}

default_install_prefix = sys.prefix
print("*" * 80)
print("EdelweissMeshfree setup")
print("System prefix: " + sys.prefix)
print("*" * 80)

marmot_dir = expanduser(os.environ.get("MARMOT_INSTALL_DIR", default_install_prefix))
print("Marmot install directory (overwrite via environment var. MARMOT_INSTALL_DIR):")
print(marmot_dir)
print("*" * 80)

extensions = list()


def MarmotExtension(pyxpath, *args, **kwargs):
    """A custom extension that links against Marmot."""

    return Extension(
        "*",
        sources=[
            pyxpath,
        ],
        include_dirs=[join(marmot_dir, "include"), join(marmot_dir, "include", "eigen3"), numpy.get_include()],
        libraries=["Marmot"],
        library_dirs=[join(marmot_dir, "lib")],
        runtime_library_dirs=[join(marmot_dir, "lib")],
        language="c++",
        *args,
        **kwargs,
    )


extensions += [
    MarmotExtension("edelweissmeshfree/cells/marmotcell/marmotcell.pyx"),
]

extensions += [
    MarmotExtension("edelweissmeshfree/cells/marmotcell/lagrangianmarmotcell.pyx"),
]

extensions += [
    MarmotExtension("edelweissmeshfree/cells/marmotcell/bsplinemarmotcell.pyx"),
]

extensions += [
    Extension(
        "*",
        sources=[
            "edelweissmeshfree/mpmmanagers/utils.pyx",
        ],
        include_dirs=[join(marmot_dir, "include"), numpy.get_include()],
        runtime_library_dirs=[join(marmot_dir, "lib")],
        language="c++",
    )
]

extensions += [
    Extension(
        "*",
        sources=[
            "edelweissmeshfree/fieldoutput/mpresultcollector.pyx",
        ],
        include_dirs=[join(marmot_dir, "include"), numpy.get_include()],
        runtime_library_dirs=[join(marmot_dir, "lib")],
        language="c++",
    )
]

extensions += [
    MarmotExtension("edelweissmeshfree/materialpoints/marmotmaterialpoint/mp.pyx"),
]

extensions += [
    Extension(
        "*",
        sources=[
            "edelweissmeshfree/solvers/base/parallelization.pyx",
        ],
        include_dirs=[join(marmot_dir, "include"), join(marmot_dir, "include", "eigen3"), numpy.get_include()],
        language="c++",
        extra_compile_args=[
            "-fopenmp",
            "-Wno-maybe-uninitialized",
        ],
        extra_link_args=["-fopenmp"],
    )
]

extensions += [
    MarmotExtension("edelweissmeshfree/cellelements/marmotcellelement/marmotcellelement.pyx"),
]

extensions += [
    MarmotExtension("edelweissmeshfree/cellelements/marmotcellelement/lagrangianmarmotcellelement.pyx"),
]


extensions += [MarmotExtension("edelweissmeshfree/meshfree/kernelfunctions/marmot/marmotmeshfreekernelfunction.pyx")]
extensions += [MarmotExtension("edelweissmeshfree/meshfree/approximations/marmot/marmotmeshfreeapproximation.pyx")]
extensions += [
    MarmotExtension("edelweissmeshfree/particles/marmot/marmotparticlewrapper.pyx"),
]

setup(
    cmdclass={"build_ext": build_ext},
    ext_modules=cythonize(extensions, compiler_directives=directives, annotate=True, language_level=3),
),
