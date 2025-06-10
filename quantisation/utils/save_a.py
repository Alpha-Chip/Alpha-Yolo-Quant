import numpy as np


def save_a(matr, layer):
    with open('a_save_activations.txt', 'w') as f_obj:
        pass
    with open('a_save_activations.txt', 'a') as f_obj:
        f_obj.write(f"{layer}: {abs(matr).max()}\n")


def save_max_a(maxim_a, matr, layer):
    # a = abs(matr).max()
    # if layer not in maxim_a.keys():
    #     maxim_a[layer] = {}
    #
    # for batch in range(matr.shape[0]):
    #     for channel in range(matr.shape[1]):
    #         if f'{batch}_{channel}' not in maxim_a[layer].keys():
    #             maxim_a[layer][f'{batch}_{channel}'] = []
    #         maxim_a[layer][f'{batch}_{channel}'].append(np.float64(abs(matr[batch, channel, :, :]).max()))

    a = abs(matr).max()
    if layer not in maxim_a.keys():
        maxim_a[layer] = [a]
    else:
        maxim_a[layer].append(a)
