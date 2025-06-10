import numpy as np
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
# ]])
# print('ИСХОДНАЯ МАТРИЦА')
# print(start_32)
# print(start_32.shape)
def pad(matrix, pad=0):
    if pad > 0:
        matrix = np.pad(matrix, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
    return matrix


def maxpooling(img, kernel=2, padding=0, stride=1):
    if stride != kernel:
        stride = stride
    out_size_x = int(((img.shape[2] + padding * 2 - kernel) / stride) + 1)
    out_size_y = int(((img.shape[3] + padding * 2 - kernel) / stride) + 1)
    if padding != 0:
        img = pad(img, padding)
    res = np.zeros((img.shape[0], img.shape[1], out_size_x, out_size_y))
    for batch in range(img.shape[0]):
        submatr = np.zeros((img.shape[0], img.shape[1], out_size_x, out_size_y))
        for channel in range(img.shape[1]):
            count_strdie_h = 0
            for height in range(0, img.shape[2] - kernel + 1, stride):
                count_strdie_w = 0
                for width in range(0, img.shape[3] - kernel + 1, stride):
                    layer = img[batch, channel, height:height+kernel, width:width+kernel]
                    if height < padding and width < padding:
                        submatr[batch, channel, count_strdie_h, count_strdie_w] += np.max(layer[padding - height:, padding - width:])
                        # print(layer, '\n')
                        # print(layer[padding - height:, padding - width:])
                        # print(height, width)
                        # print('-' * 100)
                    elif height < padding and width >= padding and width < out_size_y - padding:
                        submatr[batch, channel, count_strdie_h, count_strdie_w] += np.max(layer[padding - height:, :])
                        # print(layer, '\n')
                        # print(layer[padding - height:, :])
                        # print(height, width)
                        # print('-' * 100)
                    elif height >= padding and width < padding and height < out_size_x - padding:
                        submatr[batch, channel, count_strdie_h, count_strdie_w] += np.max(layer[:, padding - width:])
                        # print(layer, '\n')
                        # print(layer[:, padding - width:])
                        # print(height, width)
                        # print('-' * 100)
                    elif height >= out_size_x - padding and width >= out_size_y - padding:
                        submatr[batch, channel, count_strdie_h, count_strdie_w] += np.max(layer[:-(height - out_size_x + 2 * padding - 1), :-(width - out_size_y + 2 * padding - 1)])
                        # print(layer, '\n')
                        # print(layer[:-(height - out_size_x + 2 * padding - 1), :-(width - out_size_y + 2 * padding - 1)])
                        # print('-' * 100)
                    elif height >= out_size_x - padding and width < out_size_y - padding and width >= padding:
                        submatr[batch, channel, count_strdie_h, count_strdie_w] += np.max(layer[:-(height - out_size_x + 2 * padding - 1), :])
                        # print(layer, '\n')
                        # print(layer[:-(height - padding - 1), :])
                        # print('-' * 100)
                    elif height < out_size_x - padding and width >= out_size_y - padding and height >= padding:
                        submatr[batch, channel, count_strdie_h, count_strdie_w] += np.max(layer[:, :-(width - out_size_y + 2 * padding - 1)])
                        # print(layer, '\n')
                        # print(layer[:, :-(width - padding - 1)])
                        # print(height, width)
                        # print('-' * 100)
                    elif height <= padding and width >= out_size_y - padding:
                        # print(layer, '\n')
                        # print(layer[padding - height:, :-(width - out_size_y + 2 * padding - 1)])
                        submatr[batch, channel, count_strdie_h, count_strdie_w] += np.max(layer[padding - height:, :-(width - out_size_y + 2 * padding - 1)])
                        # print(layer, '\n')
                        # print(layer[padding - height:, :-(width - out_size_y + padding)])
                        # print(height, width)
                        # print('-' * 100)
                    elif height >= out_size_x - padding and width < padding:
                        submatr[batch, channel, count_strdie_h, count_strdie_w] += np.max(layer[:-(height - out_size_x + 2 * padding - 1), padding - width:])
                        # print(layer, '\n')
                        # print(layer[:-(height - padding - 1), padding - width:])
                        # print(height, width)
                        # print('-' * 100)
                    else:
                        submatr[batch, channel, count_strdie_h, count_strdie_w] += np.max(layer)


                    # submatr[batch, channel, count_strdie_h, count_strdie_w] += np.max(layer)
                    # print(layer)
                    # print(np.max(layer))
                    # print('-' * 10)
                    count_strdie_w += 1
                count_strdie_h += 1
        res[batch] += submatr[batch]
    res = np.int64(res)
    return res


def get_pools(img: np.array, pool_size: int, stride: int) -> np.array:
    # To store individual pools
    pools = []

    # Iterate over all row blocks (single block has `stride` rows)
    for i in np.arange(img.shape[2], step=stride):
        # Iterate over all column blocks (single block has `stride` columns)
        for j in np.arange(img.shape[3], step=stride):

            # Extract the current pool
            mat = img[:, :, i:i + pool_size, j:j + pool_size]

            # Make sure it's rectangular - has the shape identical to the pool size
            if mat.shape[2] == pool_size:
                # Append to the list of pools
                pools.append(mat)
    pools = np.array(pools)

    # Return all pools as a Numpy array
    return pools.reshape(pools.shape[0] * img.shape[0], img.shape[1], pool_size, pool_size)


def max_pooling(img, pools: np.array) -> np.array:
    # Total number of pools
    num_pools = pools.shape[0] / img.shape[0]
    # Shape of the matrix after pooling - Square root of the number of pools
    # Cast it to int, as Numpy will return it as float
    # For example -> np.sqrt(16) = 4.0 -> int(4.0) = 4
    tgt_shape = (int(np.sqrt(num_pools)), int(np.sqrt(num_pools)))
    # To store the max values
    pooled = []

    # Iterate over all pools
    for pool in pools:
        # Append the max value only
        pooled.append(np.max(np.max(pool, 1), 1))
    pooled = np.array(pooled).T


    # Reshape to target shape
    return pooled.reshape(img.shape[0], img.shape[1], tgt_shape[0], tgt_shape[0])


# def maxpooling(img, kernel=2):
#     M = img.shape[2]
#     N = img.shape[3]
#     K = kernel
#     L = kernel
#
#     MK = M // K
#     NL = N // L
#     print(MK, NL)
#     return img[:, :, :MK * K, :NL * L].reshape(img.shape[0], img.shape[1], MK, NL).max(axis=(1, 3))

# print(avgpooling(start_32))


