from .config import config
print(config)
from ._version import get_versions
from .utils.backend import arraylib as xp
print(xp)

v = get_versions()
__version__ = v.get("closest-tag", v["version"])
__git_version__ = v.get("full-revisionid")
del get_versions
