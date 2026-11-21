from edelweissfe.utils.exceptions import ModuleNotBuilt


# Mock for marmotmeshfreekernelfunction.pyx
class MarmotMeshfreeKernelFunctionWrapper:
    def __init__(self, *args, **kwargs):
        raise ModuleNotBuilt("marmotmeshfreekernelfunction is not built in this environment")
