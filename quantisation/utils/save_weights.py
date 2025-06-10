from yolov8n_quantisation.quantisation.stage_0 import MAX_ACTIVATIONS_MODE
import gzip
import pickle
import numpy as np
import os
import torch


def save_in_file(dir_names, arr, file_name):
    pickle.dump(arr, gzip.open(f'{dir_names}/weights_pickle/{file_name}', 'wb+', compresslevel=3), protocol=4)


def save_batch(dir_names, arr, layer, file_name):
    if MAX_ACTIVATIONS_MODE.lower() == 'min_mae':
        try:
            os.mkdir(f'{dir_names}/batches/{layer}')
        except Exception as e:
            pass
        pickle.dump(arr, gzip.open(f'{dir_names}/batches/{layer}/{file_name}', 'wb+', compresslevel=3), protocol=4)
    else:
        pass


def load_from_file(dir_names, file_name):
    return pickle.load(gzip.open(f'{dir_names}/weights_pickle/{file_name}', 'rb'))


def save_bias_scales(dir_names, arr, file_name):
    pickle.dump(arr, gzip.open(f'{dir_names}/bias_scales/{file_name}', 'wb+', compresslevel=3), protocol=4)


def load_scale(dir_names, file_name):
    return pickle.load(gzip.open(f'{dir_names}/bias_scales/{file_name}', 'rb'))


def load_scales(dir_names):
    all_scales = {}
    files = os.listdir(os.path.join(dir_names, 'bias_scales'))
    for file in files:
        file_name = file.split('_scale')[0]
        all_scales[file_name] = torch.from_numpy(load_scale(dir_names, file)).type(torch.float32)
    return all_scales


def bit_converter(final_file_name, k, value, element):
    bin_prefix = bin(value).split('b')[0]
    bin_value = bin(value).split('b')[1]
    if element == 'bias':
        zeroes = '0' * (18 - len(bin_value))
        if 18 - len(bin_value) < 0:
            print(f'BIAS MORE THAN 18 BIT! {bin_value} {final_file_name}')
        if len(bin_prefix) == 2:
            bin_prefix = bin_prefix[0] + '18'
        else:
            bin_prefix = '18'
    elif element == 'rescale':
        zeroes = '0' * (k - len(bin_value))
        if k - len(bin_value) < 0:
            print(f'RESCALE MORE THAN {k} BIT! {bin_value} {final_file_name}')
        bin_prefix = str(k)
    else:
        zeroes = '0' * (k - len(bin_value) - 1)
        if (k - len(bin_value) - 1) < 0:
            print(f'MORE THAN {k} BIT! {bin_value} {final_file_name}')
        if len(bin_prefix) == 2:
            bin_prefix = bin_prefix[0] + str(k - 1)
        else:
            bin_prefix = str(k - 1)
    res = zeroes + bin_value
    return f"{bin_prefix}'b{res}"


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


def save_txt_weight(conv, bias, file_name, type, k, dir_names):
    final_file_name = f'{file_name}_type_{type}_bit_{k}_shape_{conv.shape}'
    with open(f'{dir_names}/quant_weights_yolov8n/{final_file_name}.txt', 'w') as f_obj:
        i = 0
        for batch in range(conv.shape[0]):
            f_obj.write(f'\n//   Batch: {batch}\n\n')
            for channel in range(conv.shape[1]):
                for height in range(conv.shape[2]):
                    for width in range(conv.shape[3]):
                        f_obj.write(f'weight[{i}] = {bit_converter(final_file_name, k, conv[batch, channel, height, width], "weight")}; // {conv[batch, channel, height, width]}\n')
                        i += 1
                f_obj.write('\n')
        f_obj.write('\n\n')
        i = 0
        for batch in range(bias.shape[0]):
            for channel in range(bias.shape[1]):
                for height in range(bias.shape[2]):
                    for width in range(bias.shape[3]):
                        f_obj.write(f'weight_bias[{i}] = {bit_converter(final_file_name, k, bias[batch, channel, height, width], "bias")}; // {bias[batch, channel, height, width]}\n')
                        i += 1


def save_txt_activations(arr, file_name, dir_names, type, k, silu=False):
    if silu == False:
        final_file_name = f'quant_activations/conv2d/{file_name}_type_{type}_bit_{k}_shape_{arr.shape}'
    else:
        final_file_name = f'quant_activations/silu/{file_name}_type_{type}_bit_{k}_shape_{arr.shape}'
    with open(f'{dir_names}/{final_file_name}.txt', 'w') as f_obj:
        i = 0
        for batch in range(arr.shape[0]):
            for channel in range(arr.shape[1]):
                f_obj.write(f'\n//   Channel: {channel}\n\n')
                for height in range(arr.shape[2]):
                    for width in range(arr.shape[3]):
                        f_obj.write(f'pixel[{i}] = {bit_converter(final_file_name, k, arr[batch, channel, height, width], "activ")}; // {arr[batch, channel, height, width]}\n')
                        i += 1
                f_obj.write('\n')


def save_txt_rescale_shift(conv, rescale, shift, file_name, dir_names, type, k, silu=False):
    # rescale, shift = rescale_shift(old_scale, new_scale, bit_size_for_koeff=bit_size_for_koeff)
    if silu == False:
        final_file_name = f'quant_activations/conv2d/{file_name}_type_{type}_bit_{k}_shape_{conv.shape}'
    else:
        final_file_name = f'quant_activations/silu/{file_name}_type_{type}_bit_{k}_shape_{conv.shape}'

    rescale_mass = []
    shift_mass = []
    try:
        for el_index in range(rescale.shape[1]):
            rescale_mass.append(bit_converter(final_file_name, k, rescale[0, el_index, 0, 0], "rescale"))
            shift_mass.append(bit_converter(final_file_name, k, shift[0, el_index, 0, 0], "rescale"))
    except:
        rescale = np.expand_dims(np.array([rescale]), (0, 2, 3))
        shift = np.expand_dims(np.array([shift]), (0, 2, 3))
        rescale_mass.append(bit_converter(final_file_name, k, rescale[0, 0, 0, 0], "rescale"))
        shift_mass.append(bit_converter(final_file_name, k, shift[0, 0, 0, 0], "rescale"))

    with open(f'{dir_names}/{final_file_name}.txt', 'a') as f_obj:
        f_obj.write('\n')
        for ind in range(len(rescale_mass)):
            f_obj.write(f'rescale[{ind}] = {rescale_mass[ind]}; // {rescale[0, ind, 0, 0]}\n')
        f_obj.write('\n')
        for ind in range(len(shift_mass)):
            f_obj.write(f'shift[{ind}] = {shift_mass[ind]}; // {shift[0, ind, 0, 0]}\n')

