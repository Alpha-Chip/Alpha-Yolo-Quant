import numpy as np

from yolov8n_quantisation.quantisation.utils.clip import *


def requant_matrix(matrix, scale, k):
    matrix = matrix * scale
    return clip(np.int64(np.round(matrix)), k)


