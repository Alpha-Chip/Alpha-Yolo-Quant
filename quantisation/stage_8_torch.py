import torch
import numpy as np
import torchvision
import torch.nn as nn
from torchvision.transforms import transforms
from yolov8n_quantisation.quantisation.utils.scale import scale
from yolov8n_quantisation.quantisation.utils.result_txt import result_txt
from yolov8n_quantisation.quantisation.utils.conv2d_print_fp import conv2d
from yolov8n_quantisation.quantisation.utils.silu_torch import sigmoid_quant
from yolov8n_quantisation.quantisation.utils.silu import create_sigmoid_lookup_table
from yolov8n_quantisation.quantisation.utils.max_a import max_a
import random
from PIL import Image
from yolov8n_quantisation.quantisation.utils.txt_matrix_print import *
from yolov8n_quantisation.quantisation.utils.save_weights import load_scales
from yolov8n_quantisation.quantisation.utils.rescale_coeff_torch import requantize
from yolov8n_quantisation.quantisation.utils.save_a import *
from yolov8n_quantisation.quantisation.stage_0 import D, W, R, MAIN_DIR_NAME, BATCHNF_WEIGHTS, QUANT_WEIGHTS, K, detect_1_channels
import deeplake
from yolov8n_quantisation.quantisation.utils.coco import *
from tqdm import tqdm
from map_boxes import mean_average_precision_for_boxes
import pandas as pd
import warnings
from yolov8n_quantisation.quantisation.utils.quant_matrix_torch import quant_matrix
from yolov8n_quantisation.quantisation.utils.write_run_result import write_run_result


warnings.filterwarnings('ignore')
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
maxim_a = {}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('CUDA available ✅' if torch.cuda.is_available() else 'CUDA not available ❌')

# -----Conv-----
def convolution(in_channels, out_channels, kernel_size, padding, stride):
    in_channels = int(in_channels)
    out_channels = int(out_channels)
    conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, stride=stride, bias=True)
    # batchn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
    silu = nn.SiLU(inplace=True)
    # layers = [conv, batchn, silu]
    layers = [conv, silu]
    layers = nn.Sequential(*layers)
    return layers


# -----C2F-----
def split(x, out_channels):
    x = torch.split(x, int(out_channels / 2), dim=1)
    x1 = x[0]
    x = x[1]
    return x1, x


def bottleneck(in_channels, out_channels, kernel_size, padding, stride, shortcut=True):
    if shortcut:
        bottle_conv_1 = convolution(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        bottle_conv_2 = convolution(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        return bottle_conv_1 + bottle_conv_2
    else:
        pass


def sppf(in_channels, out_channels):
    sppf_conv_1 = convolution(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0, stride=1)
    maxp = nn.Sequential(nn.MaxPool2d(5, 1, padding=2))
    sppf_conv_2 = convolution(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0, stride=1)
    return sppf_conv_1 + maxp + maxp + maxp + sppf_conv_2


# Верхняя линия
def detect_0(in_channels):
    conv_1_0 = convolution(in_channels=in_channels, out_channels=64, kernel_size=3, padding=1, stride=1)
    conv_2_0 = convolution(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1)
    conv_3_0 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, padding=0, stride=1))
    first_line_conv = conv_1_0 + conv_2_0 + conv_3_0
    return first_line_conv


# Нижняя линия
def detect_1(in_channels):
    conv_1_1 = convolution(in_channels=in_channels, out_channels=detect_1_channels, kernel_size=3, padding=1, stride=1)
    conv_2_1 = convolution(in_channels=detect_1_channels, out_channels=detect_1_channels, kernel_size=3, padding=1, stride=1)
    conv_3_1 = nn.Sequential(nn.Conv2d(in_channels=detect_1_channels, out_channels=80, kernel_size=1, padding=0, stride=1))
    second_line_conv = conv_1_1 + conv_2_1 + conv_3_1
    return second_line_conv


def up_down_parsing(x):
    x_up = x.clone()
    x_down = x.clone()
    return x_up, x_down


def make_anchors(feats, strides, grid_cell_offset=0.5):
    """Generate anchors from features."""
    anchor_points, stride_tensor = [], []
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        _, _, h, w = feats[i].shape
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # shift x
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # shift y
        sy, sx = torch.meshgrid(sy, sx)
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points).transpose(0, 1), torch.cat(stride_tensor).transpose(0, 1)


def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)  # xywh bbox
    return torch.cat((x1y1, x2y2), dim)  # xyxy bbox


def xywh2xyxy(x):
    """
    Convert bounding box coordinates from (x, y, width, height) format to (x1, y1, x2, y2) format where (x1, y1) is the
    top-left corner and (x2, y2) is the bottom-right corner.

    Args:
        x (np.ndarray | torch.Tensor): The input bounding box coordinates in (x, y, width, height) format.

    Returns:
        y (np.ndarray | torch.Tensor): The bounding box coordinates in (x1, y1, x2, y2) format.
    """
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = torch.empty_like(x) if isinstance(x, torch.Tensor) else np.empty_like(x)  # faster than clone/copy
    dw = x[..., 2] / 2  # half-width
    dh = x[..., 3] / 2  # half-height
    y[..., 0] = x[..., 0] - dw  # top left x
    y[..., 1] = x[..., 1] - dh  # top left y
    y[..., 2] = x[..., 0] + dw  # bottom right x
    y[..., 3] = x[..., 1] + dh  # bottom right y
    return y


def coord(prediction):
    conf_thres = 0.00000001 # Threshold for classes 0.00000001    0.25
    nc = 80 # num classes
    mi = 4 + nc  # mask start index
    xc = prediction[:, 4:mi].amax(1) > conf_thres  # candidates
    nm = prediction.shape[1] - nc - 4
    max_nms = 30000
    agnostic = False
    max_wh = 7680
    iou_thres = 0.45
    max_det = 300
    bs = prediction.shape[0]  # batch size

    prediction = prediction.transpose(-1, -2)
    prediction[..., :4] = xywh2xyxy(prediction[..., :4])
    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[:, 2:4] < min_wh) | (x[:, 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        box, cls, mask = x.split((4, nc, nm), 1)

        conf, j = cls.max(1, keepdim=True)
        x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres] # Зачем это надо? (кв скобки)

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        if n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes


        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes

        scores = x[:, 4]  # scores

        boxes = x[:, :4] + c  # boxes (offset by class)
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS

        i = i[:max_det]  # limit detections
        output[xi] = x[i]
        return output


def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None, padding=True, xywh=False):
    """
    Rescales bounding boxes (in the format of xyxy by default) from the shape of the image they were originally
    specified in (img1_shape) to the shape of a different image (img0_shape).

    Args:
        img1_shape (tuple): The shape of the image that the bounding boxes are for, in the format of (height, width).
        boxes (torch.Tensor): the bounding boxes of the objects in the image, in the format of (x1, y1, x2, y2)
        img0_shape (tuple): the shape of the target image, in the format of (height, width).
        ratio_pad (tuple): a tuple of (ratio, pad) for scaling the boxes. If not provided, the ratio and pad will be
            calculated based on the size difference between the two images.
        padding (bool): If True, assuming the boxes is based on image augmented by yolo style. If False then do regular
            rescaling.
        xywh (bool): The box format is xywh or not, default=False.

    Returns:
        boxes (torch.Tensor): The scaled bounding boxes, in the format of (x1, y1, x2, y2)
    """
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[1], img1_shape[1] / img0_shape[2])  # gain  = old / new
        pad = (
            round((img1_shape[1] - img0_shape[2] * gain) / 2 - 0.1),
            round((img1_shape[0] - img0_shape[1] * gain) / 2 - 0.1),
        )  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]


    if padding:
        boxes[..., 0] -= pad[0]  # x padding
        boxes[..., 1] -= pad[1]  # y padding
        if not xywh:
            boxes[..., 2] -= pad[0]  # x padding
            boxes[..., 3] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    return clip_boxes(boxes, img0_shape)



def clip_boxes(boxes, shape):

    """
    Takes a list of bounding boxes and a shape (height, width) and clips the bounding boxes to the shape.

    Args:
        boxes (torch.Tensor): the bounding boxes to clip
        shape (tuple): the shape of the image

    Returns:
        (torch.Tensor | numpy.ndarray): Clipped boxes
    """
    if isinstance(boxes, torch.Tensor):  # faster individually (WARNING: inplace .clamp_() Apple MPS bug)
        boxes[..., 0] = boxes[..., 0].clamp(0, shape[2])  # x1
        boxes[..., 1] = boxes[..., 1].clamp(0, shape[1])  # y1
        boxes[..., 2] = boxes[..., 2].clamp(0, shape[2])  # x2
        boxes[..., 3] = boxes[..., 3].clamp(0, shape[1])  # y2
    else:
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[2])  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[1])  # y1, y2
    return boxes


def convert_res(data):
    box = data[:, :4]
    cls = data[:, -2:]
    return box, cls


all_scales = load_scales(MAIN_DIR_NAME)
max_a_dict = max_a(f'{MAIN_DIR_NAME}/results/max_a.txt')
lookup = create_sigmoid_lookup_table(7, K)


def silu(x, scale_x, a_input):
    orig_scale_conv = scale(7, K)
    res_conv_copy = x.clone()

    res, rescale, shift = requantize(x, scale_x, orig_scale_conv, K, device)
    res_silu = sigmoid_quant(res, lookup, device)
    res_silu *= res_conv_copy
    res_silu = torch.round(res_silu)

    scale_silu = scale(1, K) * scale_x

    scale_silu_input = scale(a_input, K)
    res_silu, rescale, shift = requantize(res_silu, scale_silu, scale_silu_input, K, device)
    return res_silu, scale_silu_input


class Yolov8(nn.Module):
    def __init__(self):
        super(Yolov8, self).__init__()

        features = {}  # Слои нейронки

        d = D
        w = W
        r = R

        # -----Conv_P1-----
        conv0 = convolution(in_channels=3, out_channels=64 * w, kernel_size=3, padding=1, stride=2)
        features['conv0'] = conv0
        self.conv0 = features['conv0']

        # -----Conv_P2-----
        conv1 = convolution(in_channels=64 * w, out_channels=128 * w, kernel_size=3, padding=1, stride=2)
        features['conv1'] = conv1
        self.conv1 = features['conv1']

        # -----C2F_2-----
        c2f_conv_0 = convolution(in_channels=128 * w, out_channels=128 * w, kernel_size=1, padding=0, stride=1)
        features['c2f_conv_0'] = c2f_conv_0
        n_2 = int(round(3 * d))
        for i in range(n_2):
            c2f_bottle_1 = bottleneck(in_channels=64 * w, out_channels=64 * w, kernel_size=3, padding=1, stride=1)
            features[f'cf2_bottle_{i}'] = c2f_bottle_1
        c2f_conv_1 = convolution(in_channels=192 * w, out_channels=128 * w, kernel_size=1, padding=0, stride=1)
        features['c2f_conv_1'] = c2f_conv_1
        self.cf2_conv_0 = features['c2f_conv_0']
        self.cf2_bottle_0 = features['cf2_bottle_0']
        self.cf2_conv_1 = features['c2f_conv_1']

        # -----Conv_P3-----
        conv3 = convolution(in_channels=128 * w, out_channels=256 * w, kernel_size=3, padding=1, stride=2)
        features['conv3'] = conv3
        self.conv3 = features['conv3']

        # -----C2F_4-----
        c2f_conv_2 = convolution(in_channels=256 * w, out_channels=256 * w, kernel_size=1, padding=0, stride=1)
        features['c2f_conv_2'] = c2f_conv_2
        n_4 = int(round(6 * d))
        for i in range(n_2 + 1, n_4 + n_2 + 1):
            c2f_bottle_2 = bottleneck(in_channels=128 * w, out_channels=128 * w, kernel_size=3, padding=1, stride=1)
            features[f'cf2_bottle_{i}'] = c2f_bottle_2
        c2f_conv_3 = convolution(in_channels=512 * w, out_channels=256 * w, kernel_size=1, padding=0, stride=1)
        features['c2f_conv_3'] = c2f_conv_3
        self.cf2_conv_2 = features['c2f_conv_2']
        self.cf2_bottle_2 = features['cf2_bottle_2']
        self.cf2_bottle_3 = features['cf2_bottle_3']
        self.cf2_conv_3 = features['c2f_conv_3']

        # -----Conv_P4-----
        conv5 = convolution(in_channels=256 * w, out_channels=512 * w, kernel_size=3, padding=1, stride=2)
        features['conv5'] = conv5
        self.conv5 = features['conv5']

        # -----C2F_6-----
        c2f_conv_4 = convolution(in_channels=512 * w, out_channels=512 * w, kernel_size=1, padding=0, stride=1)
        features['c2f_conv_4'] = c2f_conv_4
        n_6 = int(round(6 * d))
        for i in range(n_4 + n_2 + 1, n_6 + n_4 + n_2 + 1):
            c2f_bottle_3 = bottleneck(in_channels=256 * w, out_channels=256 * w, kernel_size=3, padding=1, stride=1)
            features[f'cf2_bottle_{i}'] = c2f_bottle_3
        c2f_conv_5 = convolution(in_channels=1024 * w, out_channels=512 * w, kernel_size=1, padding=0, stride=1)
        features['c2f_conv_5'] = c2f_conv_5
        self.cf2_conv_4 = features['c2f_conv_4']
        self.cf2_bottle_4 = features['cf2_bottle_4']
        self.cf2_bottle_5 = features['cf2_bottle_5']
        self.cf2_conv_5 = features['c2f_conv_5']

        # -----Conv_P5-----
        conv7 = convolution(in_channels=512 * w, out_channels=512 * w * r, kernel_size=3, padding=1, stride=2)
        features['conv7'] = conv7
        self.conv7 = features['conv7']

        # -----C2F_8-----
        c2f_conv_6 = convolution(in_channels=512 * w * r, out_channels=512 * w * r, kernel_size=1, padding=0, stride=1)
        features['c2f_conv_6'] = c2f_conv_6
        n_8 = int(round(3 * d))
        for i in range(n_8):
            c2f_bottle_4 = bottleneck(in_channels=256 * w * r, out_channels=256 * w * r, kernel_size=3, padding=1,
                                      stride=1)
            features[f'cf2_bottle_{n_8 + n_6 + n_4 + n_2 + i}'] = c2f_bottle_4
        c2f_conv_7 = convolution(in_channels=768 * w * r, out_channels=512 * w * r, kernel_size=1, padding=0, stride=1)
        features['c2f_conv_7'] = c2f_conv_7
        self.cf2_conv_6 = features['c2f_conv_6']
        self.cf2_bottle_6 = features['cf2_bottle_6']
        self.cf2_conv_7 = features['c2f_conv_7']

        # -----SPPF-----
        sppf_conv_1 = convolution(in_channels=512 * w * r, out_channels=256 * w * r, kernel_size=1, padding=0, stride=1)
        features['sppf_conv_1'] = sppf_conv_1
        maxp_0 = nn.Sequential(nn.MaxPool2d(5, 1, padding=2))
        features['maxp_0'] = maxp_0
        maxp_1 = nn.Sequential(nn.MaxPool2d(5, 1, padding=2))
        features['maxp_1'] = maxp_1
        maxp_2 = nn.Sequential(nn.MaxPool2d(5, 1, padding=2))
        features['maxp_2'] = maxp_2
        sppf_conv_2 = convolution(in_channels=1024 * w * r, out_channels=512 * w * r, kernel_size=1, padding=0,
                                  stride=1)
        features['sppf_conv_2'] = sppf_conv_2
        self.sppf_conv_1 = features['sppf_conv_1']
        self.maxp_0 = features['maxp_0']
        self.maxp_1 = features['maxp_1']
        self.maxp_2 = features['maxp_2']
        self.sppf_conv_2 = features['sppf_conv_2']

        # -----UPSAMPLE_10-----
        ups_10 = nn.Sequential(nn.Upsample(None, 2, 'nearest'))
        features['ups_10'] = ups_10
        self.ups_10 = features['ups_10']

        # -----C2F_12-----
        c2f_conv_8 = convolution(in_channels=512 * w * (1 + r), out_channels=512 * w, kernel_size=1, padding=0,
                                 stride=1)  # 128?
        features['c2f_conv_8'] = c2f_conv_8
        n_12 = int(round(3 * d))
        for i in range(n_12):
            features[f'cf2_bottle_{n_12 + n_8 + n_6 + n_4 + n_2 + i}'] = bottleneck(in_channels=256 * w,
                                                                                    out_channels=256 * w, kernel_size=3,
                                                                                    padding=1, stride=1)
        c2f_conv_9 = convolution(in_channels=256 * w * (1 + r), out_channels=512 * w, kernel_size=1, padding=0,
                                 stride=1)
        features['c2f_conv_9'] = c2f_conv_9
        self.cf2_conv_8 = features['c2f_conv_8']
        self.cf2_bottle_7 = features['cf2_bottle_7']
        self.cf2_conv_9 = features['c2f_conv_9']

        # -----UPSAMPLE_13-----
        # ups_13 = nn.Sequential(nn.ConvTranspose2d(int(512*w), int(256*w), 2, 2, 0, bias=True))
        ups_13 = nn.Sequential(nn.Upsample(None, 2, 'nearest'))
        features['ups_13'] = ups_13
        self.ups_13 = features['ups_13']

        # -----C2F_15-----
        c2f_conv_10 = convolution(in_channels=256 * w * (1 + r), out_channels=256 * w, kernel_size=1, padding=0,
                                  stride=1)
        features['c2f_conv_10'] = c2f_conv_10
        n_15 = int(round(3 * d))
        for i in range(n_15):
            features[f'cf2_bottle_{n_15 + n_12 + n_8 + n_6 + n_4 + n_2 + i}'] = bottleneck(in_channels=128 * w,
                                                                                           out_channels=128 * w,
                                                                                           kernel_size=3, padding=1,
                                                                                           stride=1)
        c2f_conv_11 = convolution(in_channels=128 * w * (1 + r), out_channels=256 * w, kernel_size=1, padding=0,
                                  stride=1)
        features['c2f_conv_11'] = c2f_conv_11
        self.cf2_conv_10 = features['c2f_conv_10']
        self.cf2_bottle_8 = features['cf2_bottle_8']
        self.cf2_conv_11 = features['c2f_conv_11']

        # -----Conv_P3-----
        conv8 = convolution(in_channels=256 * w, out_channels=256 * w, kernel_size=3, padding=1, stride=2)
        features['conv8'] = conv8
        self.conv8 = features['conv8']

        # -----C2F_18-----
        c2f_conv_12 = convolution(in_channels=768 * w, out_channels=512 * w, kernel_size=1, padding=0, stride=1)
        features['c2f_conv_12'] = c2f_conv_12
        n_18 = int(round(3 * d))
        for i in range(n_18):
            features[f'cf2_bottle_{n_18 + n_15 + n_12 + n_8 + n_6 + n_4 + n_2 + i}'] = bottleneck(in_channels=256 * w,
                                                                                                  out_channels=256 * w,
                                                                                                  kernel_size=3,
                                                                                                  padding=1, stride=1)
        c2f_conv_13 = convolution(in_channels=768 * w, out_channels=512 * w, kernel_size=1, padding=0, stride=1)
        features['c2f_conv_13'] = c2f_conv_13
        self.cf2_conv_12 = features['c2f_conv_12']
        self.cf2_bottle_9 = features['cf2_bottle_9']
        self.cf2_conv_13 = features['c2f_conv_13']

        # -----Conv_19-----
        conv9 = convolution(in_channels=512 * w, out_channels=512 * w, kernel_size=3, padding=1, stride=2)
        features['conv9'] = conv9
        self.conv9 = features['conv9']

        # -----C2F_21-----
        c2f_conv_14 = convolution(in_channels=512 * w * (1 + r), out_channels=1024 * w, kernel_size=1, padding=0,
                                  stride=1)
        features['c2f_conv_14'] = c2f_conv_14
        n_19 = int(round(3 * d))
        for i in range(n_19):
            features[f'cf2_bottle_{n_19 + n_18 + n_15 + n_12 + n_8 + n_6 + n_4 + n_2 + i}'] = bottleneck(
                in_channels=512 * w, out_channels=512 * w, kernel_size=3, padding=1, stride=1)
        c2f_conv_15 = convolution(in_channels=512 * w * (1 + r), out_channels=1024 * w, kernel_size=1, padding=0,
                                  stride=1)
        features['c2f_conv_15'] = c2f_conv_15
        self.cf2_conv_14 = features['c2f_conv_14']
        self.cf2_bottle_10 = features['cf2_bottle_10']
        self.cf2_conv_15 = features['c2f_conv_15']

        # -----DETECT_5-----
        up_line = detect_0(256 * w)
        down_line = detect_1(256 * w)
        features['detect_5'] = [up_line, down_line]

        # -----DETECT_6-----
        up_line = detect_0(512 * w)
        down_line = detect_1(512 * w)
        features['detect_6'] = [up_line, down_line]

        # -----DETECT_X-----
        up_line = detect_0(512 * w * r)
        down_line = detect_1(512 * w * r)
        features['detect_x'] = [up_line, down_line]

        # -----Detect_UP-----
        self.detect_5_up = features['detect_5'][0]
        self.detect_5_down = features['detect_5'][1]

        self.detect_6_up = features['detect_6'][0]
        self.detect_6_down = features['detect_6'][1]

        self.detect_x_up = features['detect_x'][0]
        self.detect_x_down = features['detect_x'][1]


        # -----DFL-----
        dfl = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1, padding=0, stride=1, bias=False).requires_grad_(
            False)
        features['dfl'] = dfl
        self.dfl = features['dfl']


    def forward(self, x):
        orig_img = x.clone()
        x, _ = quant_matrix(x, K)
        scale_x = all_scales['Conv_P1']
        x = x.to(device)

        # -----Conv_P1-----
        x = self.conv0[0](x)  # Conv
        x, _ = silu(x, scale_x, max_a_dict['conv_p2'])
        # x = self.conv0[1](x)  # SiLU

        # -----Conv_P2-----
        x = self.conv1[0](x)
        scale_x = all_scales['Conv_P2']
        x, _ = silu(x, scale_x, max_a_dict['conv_0_c2f'])
        # x = self.conv1[1](x)

        # -----C2F_2-----
        x = self.cf2_conv_0[0](x)
        scale_x = all_scales['C2F_2_conv_0']
        x, scale_res_conv_0_с2f = silu(x, scale_x, max_a_dict['conv_b_0_c2f'])
        # x = self.cf2_conv_0[1](x)

        x1, x = split(x, x.shape[1])
        x2 = x.clone()
        # --Bottleneck--
        x_bottle_0 = x.clone()
        x = self.cf2_bottle_0[0](x)
        scale_x = all_scales['C2F_2_bottle_0']
        x, _ = silu(x, scale_x, max_a_dict['conv_b_1_c2f'])
        # x = self.cf2_bottle_0[1](x)
        x = self.cf2_bottle_0[2](x)
        scale_x = all_scales['C2F_2_bottle_1']
        x, scale_x = silu(x, scale_x, max_a_dict['conv_b_2_c2f'])
        # x = self.cf2_bottle_0[3](x)
        x, _, _ = requantize(x, scale_x, scale_res_conv_0_с2f, K, device)
        x += x_bottle_0
        # --Bottleneck--
        x = torch.cat((x1, x2, x), dim=1)
        x = self.cf2_conv_1[0](x)
        scale_x = all_scales['C2F_2_conv_1']
        x, _ = silu(x, scale_x, max_a_dict['conv_p3'])
        # x = self.cf2_conv_1[1](x)

        # -----Conv_P3-----
        x = self.conv3[0](x)
        scale_x = all_scales['Conv_P3']
        x, _ = silu(x, scale_x, max_a_dict['conv_2_c2f'])
        # x = self.conv3[1](x)

        # -----C2F_4-----
        x = self.cf2_conv_2[0](x)
        scale_x = all_scales['C2F_4_conv_0']
        x, scale_res_conv_2_с2f = silu(x, scale_x, max_a_dict['conv_b1_c2f'])
        # x = self.cf2_conv_2[1](x)
        x1, x = split(x, x.shape[1])
        x2 = x.clone()
        # --Bottleneck--
        x_bottle_0 = x.clone()
        x = self.cf2_bottle_2[0](x)
        # x = self.cf2_bottle_2[1](x)
        scale_x = all_scales['C2F_4_bottle_0']
        x, _ = silu(x, scale_x, max_a_dict['conv_b2_c2f'])
        x = self.cf2_bottle_2[2](x)
        scale_x = all_scales['C2F_4_bottle_1']
        x, scale_x = silu(x, scale_x, max_a_dict['conv_b3_c2f'])
        x, _, _ = requantize(x, scale_x, scale_res_conv_2_с2f, K, device)
        # x = self.cf2_bottle_2[3](x)
        x += x_bottle_0
        x3 = x.clone()
        x_bottle_1 = x.clone()
        x = self.cf2_bottle_3[0](x)
        scale_x = all_scales['C2F_4_bottle_2']
        x, _ = silu(x, scale_x, max_a_dict['conv_b4_c2f'])
        # x = self.cf2_bottle_3[1](x)
        x = self.cf2_bottle_3[2](x)
        scale_x = all_scales['C2F_4_bottle_3']
        x, scale_x = silu(x, scale_x, max_a_dict['conv_b5_c2f'])
        x, _, _ = requantize(x, scale_x, scale_res_conv_2_с2f, K, device)
        # x = self.cf2_bottle_3[3](x)
        x += x_bottle_1
        # --Bottleneck--
        x = torch.cat((x1, x2, x3, x), dim=1)
        x = self.cf2_conv_3[0](x)
        scale_x = all_scales['C2F_4_conv_1']
        x, scale_res_c2f_4 = silu(x, scale_x, max_a_dict['conv_5'])
        # x = self.cf2_conv_3[1](x)
        x_result_1 = x.clone()  # ------------------------------------->
        scale_result_1 = scale_res_c2f_4

        # -----Conv_P4-----
        x = self.conv5[0](x)
        scale_x = all_scales['Conv_P4']
        x, _ = silu(x, scale_x, max_a_dict['cf2_conv_4'])
        # x = self.conv5[1](x)

        # -----C2F_6-----
        x = self.cf2_conv_4[0](x)
        scale_x = all_scales['C2F_6_conv_0']
        x, scale_res_conv_4_с2f = silu(x, scale_x, max_a_dict['cf2_bconv_4'])
        # x = self.cf2_conv_4[1](x)
        x1, x = split(x, x.shape[1])
        x2 = x.clone()
        # --Bottleneck--
        x_bottle_0 = x.clone()
        x = self.cf2_bottle_4[0](x)
        scale_x = all_scales['C2F_6_bottle_0']
        x, _ = silu(x, scale_x, max_a_dict['cf2_bconv1_4'])
        # x = self.cf2_bottle_4[1](x)
        x = self.cf2_bottle_4[2](x)
        scale_x = all_scales['C2F_6_bottle_1']
        x, scale_x = silu(x, scale_x, max_a_dict['cf2_bconv_5'])
        # x = self.cf2_bottle_4[3](x)
        x, _, _ = requantize(x, scale_x, scale_res_conv_4_с2f, K, device)
        x += x_bottle_0
        x3 = x.clone()
        x_bottle_1 = x.clone()
        x = self.cf2_bottle_5[0](x)
        scale_x = all_scales['C2F_6_bottle_2']
        x, _ = silu(x, scale_x, max_a_dict['cf2_bconv1_5'])
        # x = self.cf2_bottle_5[1](x)
        x = self.cf2_bottle_5[2](x)
        # x = self.cf2_bottle_5[3](x)
        scale_x = all_scales['C2F_6_bottle_3']
        x, scale_x = silu(x, scale_x, max_a_dict['cf2_6_conv_last'])
        x, _, _ = requantize(x, scale_x, scale_res_conv_4_с2f, K, device)
        x += x_bottle_1
        # --Bottleneck--
        x = torch.cat((x1, x2, x3, x), dim=1)
        x = self.cf2_conv_5[0](x)
        scale_x = all_scales['C2F_6_conv_1']
        x, scale_res_c2f_6 = silu(x, scale_x, max_a_dict['conv7'])
        # x = self.cf2_conv_5[1](x)
        # matrix_txt_name(x, 'cf2_conv_6', 'matrix_batchf.txt')
        x_result_2 = x.clone()  # ------------------------------------->
        scale_result_2 = scale_res_c2f_6

        # -----Conv_P5-----
        x = self.conv7[0](x)
        scale_x = all_scales['Conv_P5']
        x, _ = silu(x, scale_x, max_a_dict['cf2_conv_6'])
        # x = self.conv7[1](x)

        # -----C2F_8-----
        x = self.cf2_conv_6[0](x)
        scale_x = all_scales['C2F_8_conv_0']
        x, scale_res_conv_6_с2f = silu(x, scale_x, max_a_dict['cf2_bottle_6'])
        # x = self.cf2_conv_6[1](x)
        x1, x = split(x, x.shape[1])
        x2 = x.clone()
        # --Bottleneck--
        x_bottle_0 = x.clone()
        x = self.cf2_bottle_6[0](x)
        # x = self.cf2_bottle_6[1](x)
        scale_x = all_scales['C2F_8_bottle_0']
        x, _ = silu(x, scale_x, max_a_dict['cf2_bottle_61'])
        x = self.cf2_bottle_6[2](x)
        scale_x = all_scales['C2F_8_bottle_1']
        x, scale_x = silu(x, scale_x, max_a_dict['cf2_conv_7'])
        x, _, _ = requantize(x, scale_x, scale_res_conv_6_с2f, K, device)
        # x = self.cf2_bottle_6[3](x)
        x += x_bottle_0
        # --Bottleneck--
        x = torch.cat((x1, x2, x), dim=1)
        x = self.cf2_conv_7[0](x)
        scale_x = all_scales['C2F_8_conv_1']
        x, _ = silu(x, scale_x, max_a_dict['sppf_conv_1'])
        # x = self.cf2_conv_7[1](x)

        # -----SPPF-----
        x = self.sppf_conv_1[0](x)
        # print(x) - good
        scale_x = all_scales['SPPF_conv_0']
        x, _ = silu(x, scale_x, max_a_dict['sppf_conv_2'])
        # x = self.sppf_conv_1[1](x)
        x1 = x.clone()
        x = self.maxp_0(x)
        # result_txt(x)
        x2 = x.clone()
        x = self.maxp_1(x)
        x3 = x.clone()
        x = self.maxp_2(x)
        x_1 = torch.cat((x1, x2, x3, x), dim=1)
        # result_txt(x_1)
        x = self.sppf_conv_2[0](x_1)
        # print(np.expand_dims(self.sppf_conv_2[0].bias.numpy(), (0, 2, 3)).shape)
        # conv2d('aaaaaa', x_1.numpy(), self.sppf_conv_2[0].weight.numpy(), np.expand_dims(self.sppf_conv_2[0].bias.numpy(), (0, 2, 3)), MAIN_DIR_NAME, 0, 1)
        scale_x = all_scales['SPPF_conv_1']
        x, scale_res_sppf_conv_1 = silu(x, scale_x, max_a_dict['cf2_conv_8'])
        # x = self.sppf_conv_2[1](x)  # ------------------------------------->
        x_result_3 = x.clone()  # ------------------------------------->
        scale_result_3 = scale_res_sppf_conv_1

        # -----UPSAMPLE_10-----
        x_result_3 = self.ups_10(x_result_3)

        # -----CONCAT_2X3-----
        x_result_3, _, _ = requantize(x_result_3, scale_result_3, scale_result_2, K, device)
        x_result_3 = torch.cat((x_result_3, x_result_2), dim=1)
        # print(x_result_3)
        # result_txt(x_result_3)

        # -----C2F_12-----
        x_result_3 = self.cf2_conv_8[0](x_result_3)
        scale_x = all_scales['C2F_12_conv_0']
        x_result_3, scale_res_conv_8_с2f = silu(x_result_3, scale_x, max_a_dict['cf2_conv_80'])
        # x_result_3 = self.cf2_conv_8[1](x_result_3)
        x1, x_result_3 = split(x_result_3, x_result_3.shape[1])
        x2 = x_result_3.clone()
        # --Bottleneck--
        x_result_3 = self.cf2_bottle_7[0](x_result_3)
        scale_x = all_scales['C2F_12_bottle_0']
        x_result_3, _ = silu(x_result_3, scale_x, max_a_dict['cf2_conv_81'])
        # x = self.cf2_bottle_7[1](x)
        x_result_3 = self.cf2_bottle_7[2](x_result_3)
        scale_x = all_scales['C2F_12_bottle_1']
        x_result_3, scale_x = silu(x_result_3, scale_x, max_a_dict['cf2_conv_9'])
        x_result_3, _, _ = requantize(x_result_3, scale_x, scale_res_conv_8_с2f, K, device)
        # x = self.cf2_bottle_7[3](x)
        # --Bottleneck--
        x_result_3 = torch.cat((x1, x2, x_result_3), dim=1)
        x_result_3 = self.cf2_conv_9[0](x_result_3)
        scale_x = all_scales['C2F_12_conv_1']
        x_result_3, scale_res_c2f_12 = silu(x_result_3, scale_x, max_a_dict['cf2_conv_10'])
        # x = self.cf2_conv_9[1](x)
        x_result_4 = x_result_3.clone()  # ------------------------------------->
        scale_result_4 = scale_res_c2f_12

        # -----UPSAMPLE_13-----
        x_result_3 = self.ups_13(x_result_3)
        scale_result_3 = scale_res_c2f_12

        # -----CONCAT_1X3-----
        x_result_3, _, _ = requantize(x_result_3, scale_result_3, scale_result_1, K, device)
        x_result_3 = torch.cat((x_result_3, x_result_1), dim=1)

        # -----C2F_15-----
        x_result_3 = self.cf2_conv_10[0](x_result_3)
        scale_x = all_scales['C2F_15_conv_0']
        x_result_3, scale_res_conv_10_с2f = silu(x_result_3, scale_x, max_a_dict['cf2_bottle_8'])
        # x_result_3 = self.cf2_conv_10[1](x_result_3)
        x1, x_result_3 = split(x_result_3, x_result_3.shape[1])
        x2 = x_result_3.clone()
        # --Bottleneck--
        x_result_3 = self.cf2_bottle_8[0](x_result_3)
        scale_x = all_scales['C2F_15_bottle_0']
        x_result_3, _ = silu(x_result_3, scale_x, max_a_dict['cf2_bottle_81'])
        # x_result_3 = self.cf2_bottle_8[1](x_result_3)
        x_result_3 = self.cf2_bottle_8[2](x_result_3)
        scale_x = all_scales['C2F_15_bottle_1']
        x_result_3, scale_x = silu(x_result_3, scale_x, max_a_dict['cf2_conv_11'])
        # x_result_3 = self.cf2_bottle_8[3](x_result_3)
        # --Bottleneck--
        x_result_3, _, _ = requantize(x_result_3, scale_x, scale_res_conv_10_с2f, K, device)
        x_result_3 = torch.cat((x1, x2, x_result_3), dim=1)
        x_result_3 = self.cf2_conv_11[0](x_result_3)
        scale_x = all_scales['C2F_15_conv_1']
        x_result_3, scale_res_c2f_15 = silu(x_result_3, scale_x, max_a_dict['conv8'])
        # x_result_3 = self.cf2_conv_11[1](x_result_3)
        x_result_5 = x_result_3.clone()  # ------------------------------------->
        scale_result_5 = scale_res_c2f_15

        # -----Conv_P3-----
        x_result_3 = self.conv8[0](x_result_3)
        scale_x = all_scales['Conv_16']
        x_result_3, scale_result_3 = silu(x_result_3, scale_x, max_a_dict['cf2_conv_12'])
        # x_result_3 = self.conv8[1](x_result_3)

        # -----CONCAT_3X4-----
        x_result_4, _, _ = requantize(x_result_4, scale_result_4, scale_result_3, K, device)
        x_result_3 = torch.cat((x_result_3, x_result_4), dim=1)

        # -----C2F_18-----
        x_result_3 = self.cf2_conv_12[0](x_result_3)
        scale_x = all_scales['C2F_18_conv_0']
        x_result_3, scale_res_conv_12_с2f = silu(x_result_3, scale_x, max_a_dict['cf2_bottle_9'])
        # x_result_3 = self.cf2_conv_12[1](x_result_3)
        x1, x_result_3 = split(x_result_3, x_result_3.shape[1])
        x2 = x_result_3.clone()
        # --Bottleneck--
        x_result_3 = self.cf2_bottle_9[0](x_result_3)
        scale_x = all_scales['C2F_18_bottle_0']
        x_result_3, _ = silu(x_result_3, scale_x, max_a_dict['cf2_bottle_90'])
        # x_result_3 = self.cf2_bottle_9[1](x_result_3)
        x_result_3 = self.cf2_bottle_9[2](x_result_3)
        scale_x = all_scales['C2F_18_bottle_1']
        x_result_3, scale_x = silu(x_result_3, scale_x, max_a_dict['cf2_conv_13'])
        # x_result_3 = self.cf2_bottle_9[3](x_result_3)
        # --Bottleneck--
        x_result_3, _, _ = requantize(x_result_3, scale_x, scale_res_conv_12_с2f, K, device)
        x_result_3 = torch.cat((x1, x2, x_result_3), dim=1)
        x_result_3 = self.cf2_conv_13[0](x_result_3)
        scale_x = all_scales['C2F_18_conv_1']
        x_result_3, scale_res_c2f_18 = silu(x_result_3, scale_x, max_a_dict['conv9'])
        # x_result_3 = self.cf2_conv_13[1](x_result_3)
        x_result_6 = x_result_3.clone()  # ------------------------------------->
        scale_result_6 = scale_res_c2f_18
        # matrix_txt_name(x_result_3, 'C2F_18', 'matrix_batchf.txt')

        # -----Conv_19-----
        x_result_3 = self.conv9[0](x_result_3)
        scale_x = all_scales['Conv_19']
        x_result_3, scale_result_3 = silu(x_result_3, scale_x, max_a_dict['cf2_conv_14'])
        # x_result_3 = self.conv9[1](x_result_3)

        # -----CONCAT_Xx3-----
        x, _, _ = requantize(x, scale_res_sppf_conv_1, scale_result_3, K, device)
        x = torch.cat((x_result_3, x), dim=1)

        # -----C2F_21-----
        x = self.cf2_conv_14[0](x)
        scale_x = all_scales['C2F_21_conv_0']
        x, scale_res_conv_14_с2f = silu(x, scale_x, max_a_dict['cf2_bottle_10'])
        # x = self.cf2_conv_14[1](x)
        x1, x = split(x, x.shape[1])
        x2 = x.clone()
        # --Bottleneck--
        x = self.cf2_bottle_10[0](x)
        scale_x = all_scales['C2F_21_bottle_0']
        x, _ = silu(x, scale_x, max_a_dict['cf2_bottle_101'])
        # x = self.cf2_bottle_10[1](x)
        x = self.cf2_bottle_10[2](x)
        scale_x = all_scales['C2F_21_bottle_1']
        x, scale_x = silu(x, scale_x, max_a_dict['cf2_conv_15'])
        # x = self.cf2_bottle_10[3](x)
        # --Bottleneck--
        x, _, _ = requantize(x, scale_x, scale_res_conv_14_с2f, K, device)
        x = torch.cat((x1, x2, x), dim=1)
        x = self.cf2_conv_15[0](x)
        scale_x = all_scales['C2F_21_conv_1']
        x, scale_res_c2f_21 = silu(x, scale_x, max_a_dict['x_down_0'])
        # x = self.cf2_conv_15[1](x)

        # -----Detect_5-----
        x_result_5_up, x_result_5_down = up_down_parsing(x_result_5)
        x_result_5_up = self.detect_5_up[0](x_result_5_up)
        scale_x_5 = all_scales['x_result_5_up_0']
        x_result_5_up, _ = silu(x_result_5_up, scale_x_5, max_a_dict['x_result_5_up_1'])
        # x_result_5_up = self.detect_5_up[1](x_result_5_up)
        x_result_5_up = self.detect_5_up[2](x_result_5_up)
        scale_x_5 = all_scales['x_result_5_up_1']
        x_result_5_up, _ = silu(x_result_5_up, scale_x_5, max_a_dict['x_result_5_up_2'])
        # x_result_5_up = self.detect_5_up[3](x_result_5_up)
        x_result_5_up = self.detect_5_up[4](x_result_5_up)
        scale_result_5_up = all_scales['x_result_5_up_2']

        x_result_5_down = self.detect_5_down[0](x_result_5_down)
        scale_x_5 = all_scales['x_result_5_down_0']
        x_result_5_down, _ = silu(x_result_5_down, scale_x_5, max_a_dict['x_result_5_down_1'])
        # x_result_5_down = self.detect_5_down[1](x_result_5_down)
        x_result_5_down = self.detect_5_down[2](x_result_5_down)
        scale_x_5 = all_scales['x_result_5_down_1']
        x_result_5_down, _ = silu(x_result_5_down, scale_x_5, max_a_dict['x_result_5_down_2'])
        # x_result_5_down = self.detect_5_down[3](x_result_5_down)
        x_result_5_down = self.detect_5_down[4](x_result_5_down)
        scale_result_5_down = all_scales['x_result_5_down_2']

        # -----Detect_6-----
        x_result_6_up, x_result_6_down = up_down_parsing(x_result_6)
        x_result_6_up = self.detect_6_up[0](x_result_6_up)
        scale_x_6 = all_scales['x_result_6_up_0']
        x_result_6_up, _ = silu(x_result_6_up, scale_x_6, max_a_dict['x_result_6_up_1'])
        # x_result_6_up = self.detect_6_up[1](x_result_6_up)
        x_result_6_up = self.detect_6_up[2](x_result_6_up)
        scale_x_6 = all_scales['x_result_6_up_1']
        x_result_6_up, _ = silu(x_result_6_up, scale_x_6, max_a_dict['x_result_6_up_2'])
        # x_result_6_up = self.detect_6_up[3](x_result_6_up)
        x_result_6_up = self.detect_6_up[4](x_result_6_up)
        scale_result_6_up = all_scales['x_result_6_up_2']


        x_result_6_down = self.detect_6_down[0](x_result_6_down)
        scale_x_6 = all_scales['x_result_6_down_0']
        x_result_6_down, _ = silu(x_result_6_down, scale_x_6, max_a_dict['x_result_6_down_1'])
        # x_result_6_down = self.detect_6_down[1](x_result_6_down)
        x_result_6_down = self.detect_6_down[2](x_result_6_down)
        scale_x_6 = all_scales['x_result_6_down_1']
        x_result_6_down, _ = silu(x_result_6_down, scale_x_6, max_a_dict['x_result_6_down_2'])
        # x_result_6_down = self.detect_6_down[3](x_result_6_down)
        x_result_6_down = self.detect_6_down[4](x_result_6_down)
        scale_result_6_down = all_scales['x_result_6_down_2']


        # -----Detect_x-----
        x_up, x_down = up_down_parsing(x)
        x_up = self.detect_x_up[0](x_up)
        scale_x_detect = all_scales['x_up_0']
        x_up, _ = silu(x_up, scale_x_detect, max_a_dict['x_up_1'])
        # x_up = self.detect_x_up[1](x_up)
        x_up = self.detect_x_up[2](x_up)
        scale_x_detect = all_scales['x_up_1']
        x_up, _ = silu(x_up, scale_x_detect, max_a_dict['x_up_2'])
        # x_up = self.detect_x_up[3](x_up)
        x_up = self.detect_x_up[4](x_up)
        scale_x_up = all_scales['x_up_2']

        x_down = self.detect_x_down[0](x_down)
        scale_x_detect = all_scales['x_down_0']
        x_down, _ = silu(x_down, scale_x_detect, max_a_dict['x_down_1'])
        # x_down = self.detect_x_down[1](x_down)
        x_down = self.detect_x_down[2](x_down)
        scale_x_detect = all_scales['x_down_1']
        x_down, _ = silu(x_down, scale_x_detect, max_a_dict['x_down_2'])
        # x_down = self.detect_x_down[3](x_down)
        x_down = self.detect_x_down[4](x_down)
        scale_x_down = all_scales['x_down_2']

        scale_result_5_up = scale_result_5_up.to(device)
        scale_result_6_up = scale_result_6_up.to(device)
        scale_x_up = scale_x_up.to(device)
        scale_result_5_down = scale_result_5_down.to(device)
        scale_result_6_down = scale_result_6_down.to(device)
        scale_x_down = scale_x_down.to(device)


        x_result_5_up = x_result_5_up / scale_result_5_up
        x_result_6_up = x_result_6_up / scale_result_6_up
        x_up = x_up / scale_x_up

        x_result_5_down = x_result_5_down / scale_result_5_down
        x_result_6_down = x_result_6_down / scale_result_6_down
        x_down = x_down / scale_x_down

        up = [x_result_5_up, x_result_6_up, x_up]

        stride = torch.Tensor([8., 16., 32.])
        box = torch.cat((x_result_5_up.view(1, 64, -1), x_result_6_up.view(1, 64, -1), x_up.view(1, 64, -1)), 2)


        b, c, a = box.shape
        box = box.view(b, 4, 16, a).transpose(2, 1).softmax(1)
        anchor, strides = make_anchors(up, stride)
        # conv2d('dfldfldfl_torch', box.numpy(), self.dfl.weight.numpy(), np.zeros(box.shape), MAIN_DIR_NAME, 0, 1)
        dfl = self.dfl(box).view(b, 4, a)
        # result_txt(dfl, 1)

        dbox = dist2bbox(dfl, anchor.unsqueeze(0), xywh=True, dim=1) * strides


        cls = torch.cat((x_result_5_down.view(1, 80, -1), x_result_6_down.view(1, 80, -1), x_down.view(1, 80, -1)), 2)
        cls = cls.sigmoid()

        dbox_cls = torch.cat((dbox, cls), 1)

        preds = coord(dbox_cls)
        if preds != None:
            for i, pred in enumerate(preds):
                img_batch = orig_img[i]
                pred[:, :4] = scale_boxes(orig_img.shape[2:], pred[:, :4], img_batch.shape)
            boxes, classes = convert_res(pred)
        else:
            boxes, classes = None, None

        return boxes, classes



model = Yolov8()
model = model.to(device)

model.load_state_dict(torch.load(f'{MAIN_DIR_NAME}/results/{QUANT_WEIGHTS}'))


# -----Тест одной фотографии-----
# test_transform = transforms.Compose([
#     transforms.Resize(640),
#     transforms.ToTensor()
#     ])
# img = Image.open('utils/cats_2_640.jpg')
# img = test_transform(img)
# img = img.unsqueeze(0).float()
# model.eval()
# with torch.no_grad():
#     print(model(img), 'result after load weights')


val_dataset = deeplake.load("hub://activeloop/coco-val")
tform = transforms.Compose([
    transforms.ToPILImage(), # Must convert to PIL image for subsequent operations to run
    # transforms.Resize((640, 640)),
    transforms.ToTensor(), # Must convert to pytorch tensor for subsequent operations to run
    transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0)==1 else x)
])

batch_transform = transforms.Compose([
    transforms.Resize((640, 640))
])

print('START QUANT VALIDATION 🚀')
trainloader = val_dataset.pytorch(num_workers=0, batch_size=1, transform = {'images': tform, 'boxes': None, 'categories': None}, shuffle=False)
model.eval()
ann = pd.DataFrame({'ImageID': [], 'LabelName': [], 'XMin': [], 'XMax': [], 'YMin': [], 'YMax': []})
det = pd.DataFrame({'ImageID': [], 'LabelName': [], 'Conf': [], 'XMin': [], 'XMax': [], 'YMin': [], 'YMax': []})
ann_mass = []
det_mass = []
no_pred = []
with torch.no_grad():
    for ind, batch in enumerate(tqdm(trainloader)):
        img = batch_transform(batch['images'])
        img = img.to(device)
        boxes, classes = model(img)
        ann_mass.append((batch['images'], str(ind), batch['boxes'], batch['categories']))
        if isinstance(boxes, torch.Tensor):
            det_mass.append((str(ind), boxes, classes))
        else:
            no_pred.append(str(ind))

        # ann = map_from_torch_ann_np(ann, batch['images'], str(ind), batch['boxes'], batch['categories'])
        # det = map_from_torch_np(det, str(ind), no_pred, boxes.numpy(), classes.numpy(), ann=0)
        # print(det.tail())


for el_ann in ann_mass:
    ann = map_from_torch_ann_np(ann, el_ann[0], el_ann[1], el_ann[2], el_ann[3])
for el_det in det_mass:
    det = map_from_torch_np(det, el_det[0], no_pred, el_det[1].cpu().numpy(), el_det[2].cpu().numpy(), ann=0)


det.to_csv(f'{MAIN_DIR_NAME}/results/det_QUANT_{K}_channel.csv', index=False)

ann = ann[['ImageID', 'LabelName', 'XMin', 'XMax', 'YMin', 'YMax']].values
det = det[['ImageID', 'LabelName', 'Conf', 'XMin', 'XMax', 'YMin', 'YMax']].values
result_ap = []
for iou_threshold in np.arange(0.5, 1, 0.05):
    mean_ap, average_precisions = mean_average_precision_for_boxes(ann, det, round(iou_threshold, 2))
    print(round(iou_threshold, 2), mean_ap)
    result_ap.append(mean_ap)
print(result_ap)
print(sum(result_ap) / len(result_ap))
write_run_result(sum(result_ap) / len(result_ap), 7)
