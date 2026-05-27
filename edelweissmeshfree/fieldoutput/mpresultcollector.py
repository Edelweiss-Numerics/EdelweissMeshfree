import numpy as np


class MaterialPointResultCollector:
    """Pure Python fallback for the Cython MaterialPointResultCollector."""

    def __init__(self, materialPoints: list, result: str):
        self._materialPoints = materialPoints
        self._result = result
        results = [mp.getResultArray(result, getPersistentView=True) for mp in materialPoints]
        n_size = results[0].shape[0]
        self.resultsTable = np.empty([len(materialPoints), n_size])

    def update(self):
        for i, mp in enumerate(self._materialPoints):
            self.resultsTable[i, :] = mp.getResultArray(self._result, getPersistentView=True)

    def getCurrentResults(self) -> np.ndarray:
        self.update()
        return self.resultsTable
