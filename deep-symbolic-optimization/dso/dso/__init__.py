# Grab TF1 compat and turn off v2
import tensorflow._api.v2.compat.v1 as _tf
_tf.disable_v2_behavior()

# 2) Make all "import tensorflow" refer to this v1 compat module
import sys
sys.modules['tensorflow'] = _tf

from dso.core import DeepSymbolicOptimizer
from dso.task.regression.sklearn import DeepSymbolicRegressor