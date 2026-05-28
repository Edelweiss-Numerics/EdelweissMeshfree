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
"""Field output management for MPM and particle-based simulations."""

from typing import Callable

import numpy as np
from edelweissfe.journal.journal import Journal
from edelweissfe.utils.fieldoutput import FieldOutputController, _FieldOutputBase

from edelweissmeshfree.fieldoutput.mpresultcollector import MaterialPointResultCollector
from edelweissmeshfree.models.mpmmodel import MPMModel
from edelweissmeshfree.sets.materialpointset import MaterialPointSet
from edelweissmeshfree.sets.particleset import ParticleSet


class MaterialPointFieldOutput(_FieldOutputBase):
    """
    A FieldOutput for material points.

    Parameters
    ----------
    name
        The name of this FieldOutput.
    mpSet
        The :class:`MaterialPointSet on which this FieldOutput operates.
    resultName
        The name of the result entry in the :class:`ElementBase.
    model
        The :class:`MPMModel tree instance.
    journal
        The :class:`Journal instance for logging.
    saveHistory
        If the history of the results should be saved.
    f_x
        The function to apply to the results.
    export
        Whether to export the results.
    fExport_x
        The function to apply to the exported results.
    reshape_to_dimensions
        If the result should be reshaped to certain dimensions (e.g. for tensor results).
    """

    def __init__(
        self,
        name: str,
        mpSet: MaterialPointSet,
        resultName: str,
        model: MPMModel,
        journal: Journal,
        saveHistory: bool = False,
        f_x=Callable,
        export: str = None,
        fExport_x=Callable,
        reshape_to_dimensions: int = None,
    ):
        """Initialize a material point field output definition.

        Parameters
        ----------
        name
            The unique name of the field output.
        mpSet
            The material point set associated with the output.
        resultName
            The name of the requested material point result.
        model
            The model that provides the field data.
        journal
            The journal used for logging.
        saveHistory
            Whether the output history should be stored.
        f_x
            Optional transformation applied to the raw result values.
        export
            Whether the field output should be exported.
        fExport_x
            Optional transformation applied before export.
        reshape_to_dimensions
            Optional target shape for the collected result values.
        """
        self.associatedSet = mpSet
        self.resultName = resultName

        self.mpResultCollector = MaterialPointResultCollector(list(self.associatedSet), self.resultName)

        super().__init__(name, model, journal, saveHistory, f_x, export, fExport_x, reshape_to_dimensions)

    def updateResults(self, model: MPMModel):
        """Update the field output.
        Will use the current solution and reaction vector if result is a nodal result.

        Parameters
        ----------
        model
            The model tree.
        """

        result = self.mpResultCollector.getCurrentResults()

        super()._applyResultsPipleline(result)


class ParticleFieldOutput(_FieldOutputBase):
    """
    A FieldOutput for material points.

    Parameters
    ----------
    name
        The name of this FieldOutput.
    pSet
        The :class:`ParticleSet on which this FieldOutput operates.
    resultName
        The name of the result entry in the :class:`ElementBase.
    model
        The :class:`MPMModel tree instance.
    journal
        The :class:`Journal instance for logging.
    saveHistory
        If the history of the results should be saved.
    f_x
        The function to apply to the results.
    export
        Whether to export the results.
    fExport_x
        The function to apply to the exported results.
    reshape_to_dimensions
        If the result should be reshaped to certain dimensions (e.g. for tensor results).
    """

    def __init__(
        self,
        name: str,
        pSet: ParticleSet,
        resultName: str,
        model: MPMModel,
        journal: Journal,
        saveHistory: bool = False,
        f_x=Callable[[np.ndarray], np.ndarray],
        export: str = None,
        fExport_x=Callable[[np.ndarray], np.ndarray],
        reshape_to_dimensions: int = None,
    ):
        """Initialize a particle field output definition.

        Parameters
        ----------
        name
            The unique name of the field output.
        pSet
            The particle set associated with the output.
        resultName
            The name of the requested particle result.
        model
            The model that provides the field data.
        journal
            The journal used for logging.
        saveHistory
            Whether the output history should be stored.
        f_x
            Optional transformation applied to the raw result values.
        export
            Whether the field output should be exported.
        fExport_x
            Optional transformation applied before export.
        reshape_to_dimensions
            Optional target shape for the collected result values.
        """
        self.associatedSet = pSet
        self.resultName = resultName

        self.pResultCollector = MaterialPointResultCollector(list(self.associatedSet), self.resultName)

        super().__init__(name, model, journal, saveHistory, f_x, export, fExport_x, reshape_to_dimensions)

    def updateResults(self, model: MPMModel):
        """Update the field output.
        Will use the current solution and reaction vector if result is a nodal result.

        Parameters
        ----------
        model
            The model tree.
        """

        result = self.pResultCollector.getCurrentResults()

        super()._applyResultsPipleline(result)


class MPMFieldOutputController(FieldOutputController):
    """
    The central module for managing field outputs, which can be used by output managers.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the MPM field output controller.

        Parameters
        ----------
        *args
            Positional arguments forwarded to the base field output controller.
        **kwargs
            Keyword arguments forwarded to the base field output controller.
        """
        super().__init__(*args, **kwargs)

    def addPerMaterialPointFieldOutput(
        self,
        name: str,
        materialPointSet: MaterialPointSet,
        result: str = None,
        saveHistory: bool = False,
        f_x=None,
        export: str = None,
        fExport_x=None,
        reshape_to_dimensions: int = None,
    ):
        """
        Parameters
        ----------
        name
            The name of this FieldOutput.
        nodeField
            The :class:`NodeField, on which this FieldOutput should operate.
        resultName
            The name of the result entry in the :class:`NodeField
        saveHistory
            If the history of the results should be saved.
        f_x
            The function to apply to the results.
        export
            Whether to export the results.
        fExport_x
            The function to apply to the exported results.
        reshape_to_dimensions
            If the result should be reshaped to certain dimensions (e.g. for tensor results).
        """
        if name in self.fieldOutputs:
            raise Exception("FieldOutput {:} already exists!".format(name))

        if not result:
            result = name

        self.fieldOutputs[name] = MaterialPointFieldOutput(
            name,
            materialPointSet,
            result,
            self.model,
            self.journal,
            saveHistory,
            f_x,
            export,
            fExport_x,
            reshape_to_dimensions,
        )

    def addPerParticleFieldOutput(
        self,
        name: str,
        particleSet: ParticleSet,
        result: str = None,
        saveHistory: bool = False,
        f_x=None,
        export: str = None,
        fExport_x=None,
        reshape_to_dimensions=None,
    ):
        """Add a per-particle field output definition to the controller.

        Parameters
        ----------
        name
            The unique name of the field output.
        particleSet
            The particle set associated with the output.
        result
            The name of the requested particle result.
        saveHistory
            Whether the output history should be stored.
        f_x
            Optional transformation applied to the raw result values.
        export
            Whether the field output should be exported.
        fExport_x
            Optional transformation applied before export.
        reshape_to_dimensions
            Optional target shape for the collected result values.
        """
        return self.addPerMaterialPointFieldOutput(
            name, particleSet, result, saveHistory, f_x, export, fExport_x, reshape_to_dimensions
        )
