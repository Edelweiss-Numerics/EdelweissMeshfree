# Mock for marmotcellelement.pyx
class MarmotCellElementWrapper(MarmotCellWrapper):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError('marmotcellelement not available in this environment')
