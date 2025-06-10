import numpy as np
import torch
from collections import OrderedDict
# weights = load_from_file(f'{path}/weights.pickle')
#
#
#
# # привожу все веса в numpy
# for key, value in weights.items():
#     weights[key] = value.detach().cpu().numpy()
    # print(key)

def read_pt(filename):
    a = torch.load(filename)
    a = a.detach().cpu().numpy()
    return a

def unsq(mass):
    mass = np.expand_dims(mass, 1)
    mass = np.expand_dims(mass, 2)
    mass = np.expand_dims(mass, 3)
    return mass


def weights_activ(path):
    weights = torch.load(path)

    # привожу все веса в numpy
    for key, value in weights.items():
        if 'bias' in key.split('.')[-1]:
            weights[key] = unsq(value.detach().cpu().numpy())
        else:
            weights[key] = value.detach().cpu().numpy()
    return weights

    # conv_1 = weights['layer0.weight']
    # matrix_txt(conv_1.transpose(3, 0, 1, 2), 'CONV_1')
    # print(type(conv_1), conv_1.transpose(3, 0, 1, 2).shape)
    # conv_1_bias = unsq(weights['layer0.bias'])
    # conv_1_orig = read_pt(f'{path}/conv_1.pt')
    #
    # conv_2 = weights['layer4.weight']
    # conv_2_bias = unsq(weights['layer4.bias'])
    # conv_2_orig = read_pt(f'{path}/conv_2.pt')
    #
    # conv_3 = weights['layer8.weight']
    # conv_3_bias = unsq(weights['layer8.bias'])
    # conv_3_orig = read_pt(f'{path}/conv_3.pt')
    #
    # conv_4 = weights['layer11.weight']
    # conv_4_bias = unsq(weights['layer11.bias'])
    # conv_4_orig = read_pt(f'{path}/conv_4.pt')
    #
    # conv_5 = weights['layer15.weight']
    # conv_5_bias = unsq(weights['layer15.bias'])
    # conv_5_orig = read_pt(f'{path}/conv_5.pt')
    #
    # conv_6 = weights['layer18.weight']
    # conv_6_bias = unsq(weights['layer18.bias'])
    # conv_6_orig = read_pt(f'{path}/conv_6.pt')
    #
    # conv_7 = weights['layer22.weight']
    # conv_7_bias = unsq(weights['layer22.bias'])
    # conv_7_orig = read_pt(f'{path}/conv_7.pt')
    #
    # conv_8 = weights['layer25.weight']
    # conv_8_bias = unsq(weights['layer25.bias'])
    # conv_8_orig = read_pt(f'{path}/conv_8.pt')
    #
    # relu_1 = read_pt(f'{path}/relu_1.pt')
    # relu_2 = read_pt(f'{path}/relu_2.pt')
    # relu_3 = read_pt(f'{path}/relu_3.pt')
    # relu_4 = read_pt(f'{path}/relu_4.pt')
    # relu_5 = read_pt(f'{path}/relu_5.pt')
    # relu_6 = read_pt(f'{path}/relu_6.pt')
    # relu_7 = read_pt(f'{path}/relu_7.pt')
    # relu_8 = read_pt(f'{path}/relu_8.pt')
    # relu_9 = read_pt(f'{path}/relu_9.pt')
    # relu_10 = read_pt(f'{path}/relu_10.pt')
    #
    # fc_1 = weights['classif0.weight']
    # fc_1_bias = weights['classif0.bias']
    # fc_1_orig = read_pt(f'{path}/fc_1.pt')
    #
    # fc_2 = weights['classif3.weight']
    # fc_2_bias = weights['classif3.bias']
    # fc_2_orig = read_pt(f'{path}/fc_2.pt')
    #
    # fc_3 = weights['classif6.weight']
    # fc_3_bias = weights['classif6.bias']
    # fc_3_orig = read_pt(f'{path}/fc_3.pt')
    #
    #
    #
    #
    #
    #
    # return conv_1, conv_1_bias, conv_1_orig, \
    #     conv_2, conv_2_bias, conv_2_orig, \
    #     conv_3, conv_3_bias, conv_3_orig, \
    #     conv_4, conv_4_bias, conv_4_orig, \
    #     conv_5, conv_5_bias, conv_5_orig, \
    #     conv_6, conv_6_bias, conv_6_orig, \
    #     conv_7, conv_7_bias, conv_7_orig, \
    #     conv_8, conv_8_bias, conv_8_orig, \
    #     relu_1, relu_2, relu_3, \
    #     relu_4, relu_5, relu_6, \
    #     relu_7, relu_8, relu_9, relu_10, \
    #     fc_1, fc_1_bias, fc_1_orig, \
    #     fc_2, fc_2_bias, fc_2_orig, \
    #     fc_3, fc_3_bias, fc_3_orig

# weights_activ()
