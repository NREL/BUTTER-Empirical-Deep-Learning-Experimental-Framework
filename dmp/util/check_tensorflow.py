import os

import tensorflow


def get_mkl_enabled_flag():
    mkl_enabled = False
    major_version = int(tensorflow.__version__.split(".")[0])
    minor_version = int(tensorflow.__version__.split(".")[1])
    onednn_enabled = 0
    if major_version >= 2:
        if minor_version < 5:
            from tensorflow.python import _pywrap_util_port  # type: ignore
        else:
            from tensorflow.python.util import _pywrap_util_port
            onednn_enabled = int(os.environ.get('TF_ENABLE_ONEDNN_OPTS', '0'))
        mkl_enabled = _pywrap_util_port.IsMklEnabled() or (onednn_enabled == 1)
    else:
        mkl_enabled = tensorflow.pywrap_tensorflow.IsMklEnabled()  # type: ignore
    return mkl_enabled


print("We are using Tensorflow version", tensorflow.__version__)
print("MKL enabled :", get_mkl_enabled_flag())
