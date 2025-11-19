def _lazy_error(name, error_msg):
    class LazyError:
        def __init__(self, name, msg):
            self._name = name
            self._msg = msg

        def __getattr__(self, item):
            raise ImportError(f"{self._name} could not be imported: {self._msg}")

        def __call__(self, *args, **kwargs):
            raise ImportError(f"{self._name} could not be imported: {self._msg}")

    return LazyError(name, error_msg)
