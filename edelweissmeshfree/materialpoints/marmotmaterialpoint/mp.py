# Mock for mp.pyx
class MarmotMaterialPointWrapper:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError('mp not available in this environment')
