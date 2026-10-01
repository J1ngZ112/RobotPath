"""STP extension package.

Isaac Sim loads the UI extension when its runtime is available. Keeping the
package importable in ordinary Python allows offline baseline training and
evaluation without installing Isaac Sim into the training environment.
"""

from importlib.util import find_spec

if find_spec("isaacsim") is not None:
    from .extension import *
