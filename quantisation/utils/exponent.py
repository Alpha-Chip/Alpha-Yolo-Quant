import numpy as np


def exponent(x):
    return np.exp(x)


def get_scale_from_max_val(max_val, bit_size):
    bit_size_max_val = 2 ** (bit_size - 1) - 1
    scale = bit_size_max_val / max_val
    return scale


def quantize(arr, max_val, bit_size):
    bit_size_max_val = 2 ** (bit_size - 1) - 1
    scale = get_scale_from_max_val(max_val, bit_size)
    arr_q = np.round(arr * scale)
    arr_q = np.clip(arr_q, -bit_size_max_val, bit_size_max_val)
    return arr_q


def dequantize(arr_q, max_val, bit_size):
    arr = arr_q.astype(np.float32)
    scale = get_scale_from_max_val(max_val, bit_size)
    if scale > 0:
        arr /= scale
    else:
        arr[...] = 0
    return arr


def create_exponent_lookup_table(max_conv_value, bit_size_act):
    global EXPONENT_LOOKUP

    # First create lookup table
    max_val = 2 ** bit_size_act - 1
    EXPONENT_LOOKUP = dict()
    for i in range(-max_val, 1, 1):
        arr = np.array((i,))
        d = dequantize(arr, max_conv_value, bit_size_act)[0]
        # d = dequantize(arr, 6, bit_size_act)[0]
        float_val = np.array((exponent(d),))
        quant_val = quantize(float_val, 1, bit_size_act)[0]
        EXPONENT_LOOKUP[i] = quant_val
        # print(arr, float_val, quant_val)
    with open(f'utils/exponent_table_{bit_size_act}_bit.txt', 'w') as f_obj:
        f_obj.write(f'// EXPONENT TABLE FOR {bit_size_act} BIT\n\n')
        for key, value in EXPONENT_LOOKUP.items():
            f_obj.write(f'{key} = {value}\n')
    return EXPONENT_LOOKUP


# print(create_sigmoid_lookup_table(10))


def exponent_quant(x, lookup):
    # global SIGMOID_LOOKUP

    # lookup = create_sigmoid_lookup_table(max_conv_value, k)
    # with open(f'utils/sigmoid_table_{k}_bit.txt', 'w') as f_obj:
    #     f_obj.write(f'// SIGMOID TABLE FOR {k} BIT\n\n')
    #     for key, value in lookup.items():
    #         f_obj.write(f'{key} = {value}\n')
    ret = x
    mass_copy = x.copy()
    k = np.array(list(lookup.keys()))
    v = np.array(list(lookup.values()))
    sidx = k.argsort()
    k = k[sidx]
    v = v[sidx]
    idx = np.searchsorted(k, ret.ravel()).reshape(ret.shape)
    idx[idx == len(k)] = 0
    mask = k[idx] == ret
    ret = np.where(mask, v[idx], 0)

    return ret


# def silu(mass, k):
#     sigmoid_table = create_sigmoid_lookup_table(k)
#     sigmoid_mass = mass.copy()
#     for batch in range(mass.shape[0]):
#         for channel in range(mass.shape[1]):
#             for height in range(mass.shape[2]):
#                 for width in range(mass.shape[3]):
#                     sigmoid_mass[batch, channel, height, width] = sigmoid_table[sigmoid_mass[batch, channel, height, width]]
#
#     return mass * sigmoid_mass





# def silu(x):
#     return (x * (1 / (1 + (np.e**(-x)))))


# 539894

#  275885834

# 8040.78576915085    0.0009464820872245293
# print(silu(31))
# print(31 / 14.131720713981359, 1/(np.e**(-31) + 1))
