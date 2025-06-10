import numpy as np


def pad(matrix, pad=0):
    if pad != 0:
        a = np.zeros((matrix.shape[0], matrix.shape[1], matrix.shape[2] + 2*pad, matrix.shape[2] + 2*pad))
        for batch in range(matrix.shape[0]):
            for channel in range(matrix.shape[1]):
                a[batch, channel, pad:pad+matrix.shape[2], pad:pad+matrix.shape[3]] += matrix[batch, channel]
    else:
        return matrix
    return a


def conv2d(filename, img, conv, bias, dir_names, padding=0,  stride=1):
    out_size_x = int(((img.shape[2] + padding * 2 - conv.shape[2]) / stride) + 1)
    out_size_y = int(((img.shape[3] + padding * 2 - conv.shape[2]) / stride) + 1)
    res = np.zeros((img.shape[0], conv.shape[0], out_size_x, out_size_y)) # batch_size входа, выход conv2d, размер входной маьот
    img = pad(img, padding)
    with open(f'{dir_names}/first_pixel/{filename}_fp.txt', 'w') as f_obj:
        for conv_batch in range(conv.shape[0]):
            submatr = np.zeros((img.shape[0], conv.shape[1], out_size_x, out_size_y))
            for batch in range(img.shape[0]):
                for channel in range(img.shape[1]):
                    count_strdie_h = 0
                    for height in range(0, img.shape[2], stride):
                        count_strdie_w = 0
                        for width in range(0, img.shape[3], stride):
                            if (width+conv.shape[2] <= img.shape[3]) and (height+conv.shape[2] <= img.shape[2]):
                                layer = img[batch, channel, height:height+conv.shape[3], width:width+conv.shape[2]]
                                weight = conv[conv_batch, channel, :, :]
                                # print(f'layer: {layer}')
                                # print(f'weight: {weight}')
                                # print(f'zxc: {width, width+conv.shape[2]}')
                                submatr[batch, channel, count_strdie_h, count_strdie_w] += np.sum(np.multiply(layer, weight))
                                if width == 0 and height == 0 and conv_batch == 0:
                                    f_obj.write(f'IMG {channel}:\n{str(layer)}\n')
                                    f_obj.write(f'CONV {channel}:\n{str(weight)}\n')
                                    f_obj.write(f'CUR RESULT_{channel}: {str(np.sum(np.multiply(layer, weight)))}\n\n')
                                # print(f'submatr: {submatr.shape}, res: {res.shape}, batch: {batch}, conv_batch: {conv_batch}, channel: {channel}')
                            count_strdie_w += 1
                        count_strdie_h += 1
                    res[batch, conv_batch] += submatr[batch, channel]
                    # print(f'RES: {res}')
                res[batch, conv_batch] += bias[0, conv_batch, 0, 0]
                f_obj.write(f'\nFIRST_PIXEL: {str(res[0, 0, 0, 0])}, BIAS: {bias[0, 0, 0, 0]}\n\n')
                break
            break


def add_silu(layer_name, res_silu, dir_names):
    with open(f'{dir_names}/first_pixel/{layer_name}_fp.txt', 'a') as f_obj:
        f_obj.write(f'\nSILU: {str(res_silu[0, 0, 0, 0])}\n')


def add_rescale_shift(layer_name, dir_names, arr, rescale, shift):
    arr_q = (rescale * arr)
    arr_q = arr_q // (2 ** (shift - 1))
    arr_q = arr_q // 2 + arr_q % 2
    arr_q = np.clip(arr_q, -127, 127)
    arr_q = arr_q.astype(np.int64)
    with open(f'{dir_names}/first_pixel/{layer_name}_fp.txt', 'a') as f_obj:
        f_obj.write(f'\nRESULT AFTER RESCALE: {arr_q[0, 0, 0, 0]}, RESCALE_COEFF: {rescale[0, 0, 0, 0]}, SHIFT: {shift[0, 0, 0, 0]}\n')
