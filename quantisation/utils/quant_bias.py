import numpy as np
def quant_bias(bias, bias_scale):
    bias_res = bias * bias_scale
    return np.int64(bias_res)
