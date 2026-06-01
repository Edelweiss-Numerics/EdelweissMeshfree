from edelweissmeshfree.cells.marmotcell.marmotcell import MarmotCellWrapper


# Mock for bsplinemarmotcell.pyx
class BSplineMarmotCellWrapper(MarmotCellWrapper):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("bsplinemarmotcell not available in this environment")
