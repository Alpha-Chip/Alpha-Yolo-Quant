import numpy as np
import torch
from yolov8n_quantisation.quantisation.utils.a import a
from yolov8n_quantisation.quantisation.utils.scale import scale
# np.random.seed(0)
# start_32 = np.array([[[
#     [-0.5417, -0.2370, -0.5756, -0.8272, -0.6499, -0.5756],
#     [0.9769, 0.9756, -0.8351, -0.9149, 0.1691, -0.5756],
#     [-0.6765, 0.9893, 0.0304, -0.7762, -0.2968, -0.5756],
#     [0.3367, 0.7293, 0.2635, 0.5142, 0.2971, -0.5756],
#     [0.7389, 0.9740, -0.6587, -0.9091, 0.5404, -0.5756],
#     [0.7389, 0.9740, -0.6587, -0.9091, 0.5404, -0.5756]],
#
#     [[-0.1415, -0.3345, 0.5183, 0.7751, 0.9597, 0.9597],
#      [-0.2032, 0.7381, 0.8946, -0.6361, -0.8385, 0.9597],
#      [0.9069, 0.5962, 0.8006, 0.8857, 0.7229, 0.9597],
#      [0.9205, 0.7757, 0.6663, -0.6767, -0.7036, 0.9597],
#      [-0.8681, -0.1987, 0.8713, -0.0591, -0.3664, 0.9597],
#      [-0.8681, -0.1987, 0.8713, -0.0591, -0.3664, 0.9597]
#     ]
# ],
#     [[
#         [-0.5417, -0.2370, -0.5756, -0.8272, -0.6499, -0.5756],
#         [0.9769, 0.9756, -0.8351, -0.9149, 0.1691, -0.5756],
#         [-0.6765, 0.9893, 0.0304, -0.7762, -0.2968, -0.5756],
#         [0.3367, 0.7293, 0.2635, 0.5142, 0.2971, -0.5756],
#         [0.7389, 0.9740, -0.6587, -0.9091, 0.5404, -0.5756],
#         [0.7389, 0.9740, -0.6587, -0.9091, 0.5404, -0.5756]],
#
#         [[-0.1415, -0.3345, 0.5183, 0.7751, 0.9597, 0.9597],
#          [-0.2032, 0.7381, 0.8946, -0.6361, -0.8385, 0.9597],
#          [0.9069, 0.5962, 0.8006, 0.8857, 0.7229, 0.9597],
#          [0.9205, 0.7757, 0.6663, -0.6767, -0.7036, 0.9597],
#          [-0.8681, -0.1987, 0.8713, -0.0591, -0.3664, 0.9597],
#          [-0.8681, -0.1987, 0.8713, -0.0591, -0.3664, 0.9597]
#          ]
#     ]
# ])
# print('ИСХОДНАЯ МАТРИЦА')
# print(start_32)
# print(start_32.shape)
# start_32 = start_32.flatten()
# print(start_32.shape)
#
# fc_1 = np.random.sample((3, 144))
# fc_1_bias = np.random.sample((3))
# print(fc_1, fc_1.shape, fc_1_bias.shape)


def new_clip(matrix, a):
    matrix[matrix > a] = a
    matrix[matrix < -a] = -a
    return matrix


# РЕАЛИЗАЦИЯ ДЛЯ ОДНОГО BATCH_SIZE
def quant_matrix(matrix, k, start=False):

    res_matrix = torch.zeros(matrix.shape)
    all_scales = torch.zeros((matrix.shape[0], 1))
    for batch in range(matrix.shape[0]):
        if start == True:
            a_max_channel = 1
        else:
            a_max_channel = a(matrix[batch, :, :, :])
        clip_batch_matrix = new_clip(matrix[batch], a_max_channel)
        scale_input_channel = scale(a_max_channel, k)
        all_scales[batch, :] = scale_input_channel
        res_matrix[batch] = torch.round(clip_batch_matrix * scale_input_channel).type(torch.float32)
    return res_matrix, all_scales
