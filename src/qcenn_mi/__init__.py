"""QCeNN-MI research package; optional simulator imports are loaded on demand."""

__all__ = ["InversionResult", "solve_inverse_2x2"]
__version__ = "0.2.0rc1"


def __getattr__(name):
    if name in __all__:
        from . import solver

        value = getattr(solver, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
