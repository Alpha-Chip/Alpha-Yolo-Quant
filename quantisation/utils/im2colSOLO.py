import numpy as np


def pad(matrix, pad=0):
    if pad > 0:
        matrix = np.pad(matrix, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
    return matrix


def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
    # First figure out what the size of the output should be
    N, C, H, W = x_shape
    # print(x_shape)
    # assert (H  - field_height) % stride == 0
    # assert (W  - field_height) % stride == 0
    out_height = int((H - field_height) / stride + 1)
    out_width = int((W - field_width) / stride + 1)

    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

    return k.astype(int), i.astype(int), j.astype(int)


def im2col_indices(x, field_height, field_width, padding=1, stride=1):
    """ An implementation of im2col based on some fancy indexing """
    # Zero-pad the input

    k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding, stride)

    cols = x[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    return cols


def im2colzxc(img, conv, padding=0, stride=1):
    h_out = ((img.shape[2] + padding * 2 - conv.shape[2]) / stride) + 1
    w_out = ((img.shape[3] + padding * 2 - conv.shape[2]) / stride) + 1
    img = pad(img, padding)
    X_col = im2col_indices(img, conv.shape[2], conv.shape[3], padding, stride)
    W_col = conv.reshape(conv.shape[0], -1)

    out = np.dot(W_col, X_col)

    out = out.reshape(conv.shape[0], int(h_out), int(w_out), img.shape[0])
    out = out.transpose(3, 0, 1, 2)
    # for i in range(out.shape[1]):
    #     out[:, i] = out[:, i] + bias[i]
    return out


# start_32 = np.array([[
#     [[-0.5417, -0.2370, -0.5756],
#     [0.9769, 0.9756, -0.8351],
#     [-0.6765, 0.9893, 0.0304]],
#
#     [[-0.1415, -0.3345, 0.5183],
#      [-0.2032, 0.7381, 0.8946],
#      [0.9069, 0.5962, 0.8006]]
# ],
# [
#     [[-0.3456, -0.24530, -0.54556],
#     [0.9573, 0.34556, -0.861],
#     [-0.12365, 0.94593, 0.5604]],
#
#     [[-0.1215, -0.7645, 0.0983],
#      [-0.435032, 0.12381, 0.9046],
#      [0.6769, 0.9862, 0.0106]]
# ]
# ])
#
#
# conv_2d_start_0 = np.array([[
#     [[-1.2546, -3.4398],
#     [3.3244, -1.9576]],
#
#     [[2.3199, -4.4192],
#     [-3.1818, -0.6805]]
# ],
#     [
#     [[4.5071, -3.4401],
#     [-2.8766, 0.2476]],
#
#     [[0.9866, 3.6618],
#      [-3.1660, -2.0877]]
#     ]
# ])
#
# # bias_0 = 0
# # bias_1 = 0
# bias_0 = -3.9077
# bias_1 = -4.9077
#
#
# bias = np.array([[[[bias_0]]], [[[bias_1]]]])
# import time
# start = time.time()
# print(im2colzxc(start_32, conv_2d_start_0, bias, padding=1))
# end = time.time()
# print(end - start)
# print('---------------')
# from conv2d import *
# start = time.time()
# print(conv2d(start_32, conv_2d_start_0, bias, padding=1))
# end = time.time()
# print(end - start)
