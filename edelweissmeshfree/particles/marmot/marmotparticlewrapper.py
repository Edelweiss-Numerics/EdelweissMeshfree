from edelweissfe.utils.exceptions import ModuleNotBuilt


class MarmotParticleWrapper:
    def __init__(self, *args, **kwargs):
        raise ModuleNotBuilt("Marmot is not built in this environment")
