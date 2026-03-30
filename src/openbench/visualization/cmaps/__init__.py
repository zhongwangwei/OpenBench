import sys

from .cmaps import Cmaps

sys.modules[__name__] = Cmaps()
