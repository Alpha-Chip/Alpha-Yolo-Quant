import torch
from yolov8n_quantisation.quantisation.utils.scale import *
from yolov8n_quantisation.quantisation.utils.clip import *


def get_pass(arr_scale):
    if type(arr_scale) != float:
        scale_pass = arr_scale.shape[1]
    else:
        scale_pass = arr_scale
    return scale_pass


def requantize(arr_q_input, old_scale, new_scale, bit_size, device, bit_size_for_koeff=8):
    bit_size_max_val = 2 ** (bit_size - 1) - 1
    arr = arr_q_input
    # We can have overflow here for int32
    old_scale_pass = get_pass(old_scale)
    new_scale_pass = get_pass(new_scale)
    if type(old_scale) == type(new_scale) == float:
        old_scale = torch.tensor(old_scale, dtype=torch.float32)
        new_scale = torch.tensor(new_scale, dtype=torch.float32)
    # if torch.unique(old_scale > 0) == True and new_scale > 0:
    if old_scale_pass > 0 and new_scale_pass > 0:
        shift_val = bit_size_for_koeff + torch.floor(torch.log2(old_scale/new_scale))
        rescale_koeff = torch.round((2 ** shift_val) * (new_scale / old_scale))
        if rescale_koeff.max() > (2 ** bit_size_for_koeff) - 1:
            # print('Problem with rescale coeff: {} > {} ({} and {})'.format(rescale_koeff, (2 ** bit_size_for_koeff) - 1, old_scale, new_scale))
            shift_val -= 1
            rescale_koeff = torch.round((2 ** shift_val) * (new_scale / old_scale))
            if rescale_koeff.max() > (2 ** bit_size_for_koeff) - 1:
                print('Problem with rescale coeff: {} > {} ({} and {})'.format(rescale_koeff, (2 ** bit_size_for_koeff) - 1, old_scale, new_scale))
                exit()
    else:
        arr[...] = 0
        rescale_koeff = 0
        shift_val = 0

    # print(rescale_koeff, shift_val)
    rescale_koeff = rescale_koeff.to(device)
    shift_val = shift_val.to(device)
    arr_q = (rescale_koeff * arr)
    arr_q = arr_q // (2 ** (shift_val - 1))
    arr_q = arr_q // 2 + arr_q % 2
    arr_q = torch.clip(arr_q, -bit_size_max_val, bit_size_max_val)
    return arr_q.type(torch.float32), rescale_koeff, shift_val
