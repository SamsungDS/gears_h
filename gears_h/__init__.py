import os

import e3x
import jax


os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# tf.config.experimental.set_visible_devices([], "GPU")
e3x.Config.set_cartesian_order(False)
e3x.Config.set_normalization("orthonormal")
# jax.config.update("jax_enable_x64", True)
if "JAX_DEFAULT_MATMUL_PRECISION" not in os.environ:
    jax.config.update("jax_default_matmul_precision", "highest")