"""Result collector for accumulating material point field output over time."""

import numpy as np


class MaterialPointResultCollector:
    """Pure Python fallback for the Cython MaterialPointResultCollector."""

    def __init__(self, materialPoints: list, result: str):
        """Initialize the material point result collector.

        Parameters
        ----------
        materialPoints
            The material points whose results should be collected.
        result
            The name of the material point result to gather.
        """
        self._materialPoints = materialPoints
        self._result = result
        results = [mp.getResultArray(result, getPersistentView=True) for mp in materialPoints]
        n_size = results[0].shape[0]
        self.resultsTable = np.empty([len(materialPoints), n_size])

    def update(self):
        """Update the cached result table from the current material point states.

        Parameters
        ----------
        None
            This method does not take additional parameters.
        """
        for i, mp in enumerate(self._materialPoints):
            self.resultsTable[i, :] = mp.getResultArray(self._result, getPersistentView=True)

    def getCurrentResults(self) -> np.ndarray:
        """Return the current material point results.

        Returns
        -------
        numpy.ndarray
            The table containing the current result values for all tracked material points.
        """
        self.update()
        return self.resultsTable
