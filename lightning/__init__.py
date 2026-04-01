try:
    from .model import Model
except ModuleNotFoundError:  # pragma: no cover - allows logger/util imports without trainer deps
    Model = None
try:
    from .util import get_ckpt_callback, get_early_stop_callback
    from .util import get_logger
except ModuleNotFoundError:  # pragma: no cover - allows partial imports when trainer deps are absent
    get_ckpt_callback = None
    get_early_stop_callback = None
    get_logger = None
