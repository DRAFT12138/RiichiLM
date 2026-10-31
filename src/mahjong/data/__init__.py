"""Data providers; plain MJAI preparation does not load the training runtime."""

__all__ = ['SyntheticEvalSource', 'SyntheticTrainSource']


def __getattr__(name):
    if name in __all__:
        from . import synthetic
        value = getattr(synthetic, name)
        globals()[name] = value
        return value
    raise AttributeError(name)
