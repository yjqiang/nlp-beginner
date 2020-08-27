from importlib import util


IS_CUPY_OK = util.find_spec('cupy') is not None
if IS_CUPY_OK:
    import cupy as np
else:
    import numpy as np
