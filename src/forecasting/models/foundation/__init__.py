from .utils.lazy_error import _lazy_error

try :
    from ._chronos import Chronos
except ImportError as e:
    Chronos = _lazy_error("Chronos", str(e))

try :
    from ._moirai import Moirai
except ImportError as e:
    Moirai = _lazy_error("Moirai", str(e))

try:
    from ._tirex import TiRex
except ImportError as e:
    TiRex = _lazy_error("TiRex", str(e))

try:
    from ._timesfm import TimesFM
except ImportError as e:
    TimesFM = _lazy_error("TimesFM", str(e))