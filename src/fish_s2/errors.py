"""Exceptions raised by :mod:`fish_s2`."""


class FishS2Error(RuntimeError):
    """Base class for SDK errors."""


class RuntimeNotFoundError(FishS2Error):
    """The native audio.cpp worker could not be found."""


class ModelLoadError(FishS2Error):
    """The native worker failed while loading the model."""


class InvalidRequestError(FishS2Error, ValueError):
    """A request or option is invalid."""


class BusyError(FishS2Error):
    """The model instance already has a request in flight."""


class GenerationError(FishS2Error):
    """Native inference rejected or failed a request."""


class WorkerCrashedError(FishS2Error):
    """The process that owns the model exited unexpectedly."""


class ProtocolError(FishS2Error):
    """The native worker produced an invalid response."""
