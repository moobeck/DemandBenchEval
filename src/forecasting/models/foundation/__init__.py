from .utils.lazy_error import _lazy_error

try:
    from .chronos import Chronos
except ImportError as e:
    Chronos = _lazy_error("Chronos", str(e))

try:
    from .tabpfn import TabPFN
except ImportError as e:
    TabPFN = _lazy_error("TabPFN", str(e))

try:
    from .moirai import Moirai
except ImportError as e:
    Moirai = _lazy_error("Moirai", str(e))
