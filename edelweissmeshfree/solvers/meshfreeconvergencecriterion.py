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

"""The convergence criterion of EdelweissMeshfree's (deprecated) implicit solvers, as a parameter set of EdelweissFE's
legacy criterion (see :mod:`edelweissfe.solvers.base.convergencecriteria`), for EdelweissFE's implicit solvers
``NIST``, ``NISTParallel`` and ``NISTPArcLength``, which solve an
:class:`~edelweissmeshfree.models.mpmmodel.MPMModel` as they solve a finite element model.

A solver built from Python takes the tolerances per field from its ``jobInfo`` and the remaining parameters as options:

.. code-block:: python

    jobInfo = jobInfoWithMeshfreeTolerances()
    solver = NISTParallel(jobInfo, journal, linsolver="pardiso", **meshfreeCriterionOptions)
"""

from edelweissfe.config.configurator import loadConfiguration

#: The tolerances of the legacy criterion that EdelweissMeshfree's solvers apply to every field, by their names in the
#: ``jobInfo`` of an EdelweissFE solver.
meshfreeFieldTolerances = {
    "fluxResidualTolerance": 1e-4,
    "fluxResidualToleranceAlternative": 1e-3,
    "absoluteFluxResidualTolerance": 1e-14,
    "fieldCorrectionTolerance": 1e-14,
    "relativeFieldCorrectionTolerance": 1e-3,
}

#: The options of an EdelweissFE implicit solver that complete the parameter set: the alternative tolerance from
#: iteration 5 on, the zero flux and zero increment thresholds, no flux residual converged before the first correction,
#: and the residual growths counted between consecutive checked iterations of a field that is not converged. No
#: prediction, as the deprecated solvers make none unless they are given a predictor.
meshfreeCriterionOptions = {
    "iterationsForAlternativeTolerance": 5,
    "zeroFluxThreshold": 1e-9,
    "zeroIncrementThreshold": 1e-13,
    "minimumIterations": 1,
    "countFirstResidualAsGrowth": False,
    "countGrowthOfConvergedFields": False,
    "extrapolation": "off",
}


def jobInfoWithMeshfreeTolerances(**tolerances: float) -> dict:
    """The ``jobInfo`` of an EdelweissFE solver with EdelweissMeshfree's tolerances for every field. The scalar
    variables, which EdelweissMeshfree's solvers did not check, keep EdelweissFE's tolerances.

    Parameters
    ----------
    **tolerances
        Tolerances that replace those of :data:`meshfreeFieldTolerances` for every field, by their names in the
        ``jobInfo``, e.g. ``fluxResidualTolerance=5e-3``.

    Returns
    -------
    dict
        The job information.
    """

    unknown = set(tolerances) - set(meshfreeFieldTolerances)
    if unknown:
        raise ValueError(
            "no such tolerance: {:}; the tolerances are {:}".format(
                ", ".join(sorted(unknown)), ", ".join(meshfreeFieldTolerances)
            )
        )

    jobInfo = loadConfiguration({})

    # new dictionaries: the defaults that loadConfiguration hands out are EdelweissFE's own
    for name, tolerance in (meshfreeFieldTolerances | tolerances).items():
        jobInfo[name] = {
            field: (default if field == "scalar variables" else tolerance) for field, default in jobInfo[name].items()
        }

    return jobInfo
