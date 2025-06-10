import numpy as np
from yolov8n_quantisation.quantisation.utils.scale import *
from yolov8n_quantisation.quantisation.utils.clip import *


def rescale_shift(old_scale, new_scale, bit_size_for_koeff=8):
    # We can have overflow here for int32
    if old_scale > 0 and new_scale > 0:
        shift_val = np.int64(bit_size_for_koeff + np.floor(np.log2(old_scale/new_scale)))
        rescale_koeff = np.round((2 ** shift_val) * (new_scale / old_scale)).astype(np.int32)
        if rescale_koeff > (2 ** bit_size_for_koeff) - 1:
            shift_val -= 1
            rescale_koeff = np.round((2 ** shift_val) * (new_scale / old_scale)).astype(np.int32)
            if rescale_koeff > (2 ** bit_size_for_koeff) - 1:
                print('Problem with rescale coeff: {} > {} ({} and {})'.format(rescale_koeff, (2 ** bit_size_for_koeff) - 1, old_scale, new_scale))
                exit()
    else:
        rescale_koeff = 0
        shift_val = 0
    return rescale_koeff, shift_val


def rescale_coeff(new_scale, old_scale):
    # rescale_koeff, shift_val = rescale_shift(old_scale, new_scale)
    # return
    return new_scale / old_scale


def requantize(arr_q_input, old_scale, new_scale, bit_size, bit_size_for_koeff=8):
    bit_size_max_val = 2 ** (bit_size - 1) - 1
    # We can have overflow here for int32
    arr = arr_q_input.astype(np.int64)
    if np.unique(old_scale > 0) == True and new_scale > 0:
        shift_val = bit_size_for_koeff + np.floor(np.log2(old_scale/new_scale))
        rescale_koeff = np.round((2 ** shift_val) * (new_scale / old_scale)).astype(np.int64)
        if rescale_koeff.max() > (2 ** bit_size_for_koeff) - 1:
            # print('Problem with rescale coeff: {} > {} ({} and {})'.format(rescale_koeff, (2 ** bit_size_for_koeff) - 1, old_scale, new_scale))
            shift_val -= 1
            rescale_koeff = np.round((2 ** shift_val) * (new_scale / old_scale)).astype(np.int64)
            if rescale_koeff.max() > (2 ** bit_size_for_koeff) - 1:
                print('Problem with rescale coeff: {} > {} ({} and {})'.format(rescale_koeff, (2 ** bit_size_for_koeff) - 1, old_scale, new_scale))
                exit()
    else:
        arr[...] = 0
        rescale_koeff = 0
        shift_val = 0

    # print(rescale_koeff, shift_val)
    arr_q = (rescale_koeff * arr)
    arr_q = arr_q // (2 ** (shift_val - 1))
    arr_q = arr_q // 2 + arr_q % 2
    # arr_q = np.round((rescale_koeff * arr) / (2 ** shift_val))
    arr_q = np.clip(arr_q, -bit_size_max_val, bit_size_max_val)
    arr_q = arr_q.astype(np.int64)
    return arr_q, rescale_koeff, np.int64(shift_val)


def requant_matrix(matrix, a_conv, res_scale, k):
    orig_scale_conv = scale(a_conv, k)
    for batch in range(matrix.shape[0]):
        for channel in range(matrix.shape[1]):
            matrix[batch, channel, :, :] = matrix[batch, channel, :, :] * orig_scale_conv / res_scale[batch, channel, :, :]
    return clip(np.int64(np.round(matrix)), k), orig_scale_conv
