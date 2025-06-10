import torch
import numpy as np
import torchvision
from torchvision.transforms import transforms
import torch.nn as nn
from collections import OrderedDict
from ultralytics import YOLO
import os
from yolov8n_quantisation.quantisation.stage_0 import D, W, R, ORIG_WEIGHTS, MAIN_DIR_NAME, MODEL_NAME, detect_1_channels
from yolov8n_quantisation.quantisation.utils.create_dirs import dirs


# try:
#     if os.path.isdir('results') == False:
#         os.mkdir('results')
#     print('DIR "results" SUCCESSFULLY CREATED ✅')
# except Exception as e:
#     print(f'DIRS ERROR ❌: {e}')
#     pass
dirs(MAIN_DIR_NAME)


test_transform = transforms.Compose([
    transforms.Resize(640),
    transforms.ToTensor(),
    ])


features = {} # Слои нейронки


# -----Conv-----
def convolution(in_channels, out_channels, kernel_size, padding, stride):
    in_channels = int(in_channels)
    out_channels = int(out_channels)
    conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, stride=stride, bias=False)
    batchn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
    silu = nn.SiLU(inplace=True)
    layers = [conv, batchn, silu]
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
    return torch.cat(anchor_points), torch.cat(stride_tensor)


def up_down_parsing(x):
    x_up = x.clone()
    x_down = x.clone()
    return x_up, x_down


def make_anchors(feats, strides, grid_cell_offset=0.5):
    """Generate anchors from features."""
    print(feats, strides, 'FEATS STRIDES')
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
    print(prediction, 'data')
    conf_thres = 0.25 # Threshold for classes  0.000000001   0.25
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
    # print(prediction, 'pred')
    prediction[..., :4] = xywh2xyxy(prediction[..., :4])
    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    print(output, 'putput')
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[:, 2:4] < min_wh) | (x[:, 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence
        # print(x, 'x')

        box, cls, mask = x.split((4, nc, nm), 1)

        conf, j = cls.max(1, keepdim=True)
        x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres] # Зачем это надо? (кв скобки)
        # print(x, x.shape, 'x')

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        if n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes

        print(x, x.shape, 'x')

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        print(c.shape, 'c')

        scores = x[:, 4]  # scores
        print(scores, scores.shape, 'scores')

        boxes = x[:, :4] + c  # boxes (offset by class)
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS

        i = i[:max_det]  # limit detections
        print(boxes, i, 'boxes i')
        output[xi] = x[i]
        print(output, 'output')
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
        print(pad, 'pad')
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
    print(boxes, 'BOXES')
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
    print(boxes, 'boxes pad')
    return boxes


def convert_res(data):
    box = data[:, :4]
    cls = data[:, -2:]
    return box, cls


class Yolov8(nn.Module):
    def __init__(self, features):
        super(Yolov8, self).__init__()

        #-----Conv_P1-----
        self.conv0 = features['conv0']

        # -----Conv_P2-----
        self.conv1 = features['conv1']

        # -----C2F_2-----
        self.cf2_conv_0 = features['c2f_conv_0']
        self.cf2_conv_1 = features['c2f_conv_1']
        self.cf2_bottle_0 = features['cf2_bottle_0']


        # -----Conv_P3-----
        self.conv3 = features['conv3']

        # -----C2F_4-----
        self.cf2_conv_2 = features['c2f_conv_2']
        self.cf2_conv_3 = features['c2f_conv_3']
        self.cf2_bottle_2 = features['cf2_bottle_2']
        self.cf2_bottle_3 = features['cf2_bottle_3']

        # -----Conv_P4-----
        self.conv5 = features['conv5']

        # -----C2F_6-----
        self.cf2_conv_4 = features['c2f_conv_4']
        self.cf2_conv_5 = features['c2f_conv_5']
        self.cf2_bottle_4 = features['cf2_bottle_4']
        self.cf2_bottle_5 = features['cf2_bottle_5']


        # -----Conv_P5-----
        self.conv7 = features['conv7']

        # -----C2F_8-----
        self.cf2_conv_6 = features['c2f_conv_6']
        self.cf2_conv_7 = features['c2f_conv_7']
        self.cf2_bottle_6 = features['cf2_bottle_6']


        # -----SPPF-----
        self.sppf_conv_1 = features['sppf_conv_1']
        self.maxp_0 = features['maxp_0']
        self.maxp_1 = features['maxp_1']
        self.maxp_2 = features['maxp_2']
        self.sppf_conv_2 = features['sppf_conv_2']

        # -----UPSAMPLE_10-----
        self.ups_10 = features['ups_10']

        # -----C2F_12-----
        self.cf2_conv_8 = features['c2f_conv_8']
        self.cf2_conv_9 = features['c2f_conv_9']
        self.cf2_bottle_7 = features['cf2_bottle_7']


        # -----UPSAMPLE_13-----
        self.ups_13 = features['ups_13']

        # -----C2F_15-----
        self.cf2_conv_10 = features['c2f_conv_10']
        self.cf2_conv_11 = features['c2f_conv_11']
        self.cf2_bottle_8 = features['cf2_bottle_8']


        # -----Conv_P3-----
        self.conv8 = features['conv8']

        # -----C2F_18-----
        self.cf2_conv_12 = features['c2f_conv_12']
        self.cf2_conv_13 = features['c2f_conv_13']
        self.cf2_bottle_9 = features['cf2_bottle_9']


        # -----Conv_19-----
        self.conv9 = features['conv9']

        # -----C2F_21-----
        self.cf2_conv_14 = features['c2f_conv_14']
        self.cf2_conv_15 = features['c2f_conv_15']
        self.cf2_bottle_10 = features['cf2_bottle_10']



        # -----Detect_UP-----
        self.detect_5_up = features['detect_5'][0]
        self.detect_6_up = features['detect_6'][0]
        self.detect_x_up = features['detect_x'][0]


        # -----Detect_DOWN-----
        self.detect_5_down = features['detect_5'][1]
        self.detect_6_down = features['detect_6'][1]
        self.detect_x_down = features['detect_x'][1]


        # -----DFL-----
        self.dfl = features['dfl']


    def forward(self, x):
        orig_img = x.clone()
        # orig_img = torch.transpose(orig_img, 0, 2)
        # orig_img = x.clone()
        # -----Conv_P1-----
        # print(x, x.shape, 'orig')
        x = self.conv0(x)
        # print(x, 'conv_0')

        # -----Conv_P2-----
        x = self.conv1(x)
        # print(x, 'conv_1')

        # -----C2F_2-----
        x = self.cf2_conv_0(x)
        x1, x = split(x, x.shape[1])
        x2 = x.clone()
            # --Bottleneck--
        x_bottle_0 = x.clone()
        x = self.cf2_bottle_0(x)
        x += x_bottle_0
            # --Bottleneck--
        x = torch.cat((x1, x2, x), dim=1)
        x = self.cf2_conv_1(x)
        # print(x, x.shape, 'after c2f')

        # -----Conv_P3-----
        x = self.conv3(x)
        # print(x, x.shape, 'conv_3')

        # -----C2F_4-----
        x = self.cf2_conv_2(x)
        x1, x = split(x, x.shape[1])
        x2 = x.clone()
            # --Bottleneck--
        x_bottle_0 = x.clone()
        x = self.cf2_bottle_2(x)
        x += x_bottle_0
        x3 = x.clone()
        x_bottle_1 = x.clone()
        x = self.cf2_bottle_3(x)
        x += x_bottle_1
            # --Bottleneck--
        x = torch.cat((x1, x2, x3, x), dim=1)
        x = self.cf2_conv_3(x)
        x_result_1 = x.clone() #------------------------------------->

        # -----Conv_P4-----
        x = self.conv5(x)

        # -----C2F_6-----
        x = self.cf2_conv_4(x)
        x1, x = split(x, x.shape[1])
        x2 = x.clone()
            # --Bottleneck--
        x_bottle_0 = x.clone()
        x = self.cf2_bottle_4(x)
        x += x_bottle_0
        x3 = x.clone()
        x_bottle_1 = x.clone()
        x = self.cf2_bottle_5(x)
        x += x_bottle_1
            # --Bottleneck--
        x = torch.cat((x1, x2, x3, x), dim=1)
        x = self.cf2_conv_5(x)
        x_result_2 = x.clone()  # ------------------------------------->

        # -----Conv_P5-----
        x = self.conv7(x)


        # -----C2F_8-----
        x = self.cf2_conv_6(x)
        x1, x = split(x, x.shape[1])
        x2 = x.clone()
            # --Bottleneck--
        x_bottle_0 = x.clone()
        x = self.cf2_bottle_6(x)
        x += x_bottle_0
            # --Bottleneck--
        x = torch.cat((x1, x2, x), dim=1)
        x = self.cf2_conv_7(x)

        # -----SPPF-----
        x = self.sppf_conv_1(x)
        x1 = x.clone()
        x = self.maxp_0(x)
        x2 = x.clone()
        x = self.maxp_1(x)
        x3 = x.clone()
        x = self.maxp_2(x)
        # print(x1.shape, x2.shape, x3.shape, x.shape)
        x = torch.cat((x1, x2, x3, x), dim=1)
        x = self.sppf_conv_2(x) # ------------------------------------->
        x_result_3 = x.clone()  # ------------------------------------->

        # -----UPSAMPLE_10-----
        x_result_3 = self.ups_10(x_result_3)

        # -----CONCAT_2X3-----
        x_result_3 = torch.cat((x_result_3, x_result_2), dim=1)

        # -----C2F_12-----
        x_result_3 = self.cf2_conv_8(x_result_3)
        x1, x_result_3 = split(x_result_3, x_result_3.shape[1])
        x2 = x_result_3.clone()
            # --Bottleneck--
        x_result_3 = self.cf2_bottle_7(x_result_3)
            # --Bottleneck--
        x_result_3 = torch.cat((x1, x2, x_result_3), dim=1)
        x_result_3 = self.cf2_conv_9(x_result_3)
        x_result_4 = x_result_3.clone()  # ------------------------------------->


        # -----UPSAMPLE_13-----
        x_result_3 = self.ups_13(x_result_3)

        # -----CONCAT_1X3-----
        x_result_3 = torch.cat((x_result_3, x_result_1), dim=1)

        # -----C2F_15-----
        x_result_3 = self.cf2_conv_10(x_result_3)
        x1, x_result_3 = split(x_result_3, x_result_3.shape[1])
        x2 = x_result_3.clone()
            # --Bottleneck--
        x_result_3 = self.cf2_bottle_8(x_result_3)
            # --Bottleneck--
        x_result_3 = torch.cat((x1, x2, x_result_3), dim=1)
        x_result_3 = self.cf2_conv_11(x_result_3)
        x_result_5 = x_result_3.clone()  # ------------------------------------->

        # -----Conv_P3-----
        x_result_3 = self.conv8(x_result_3)

        # -----CONCAT_3X4-----
        x_result_3 = torch.cat((x_result_3, x_result_4), dim=1)

        # -----C2F_18-----
        x_result_3 = self.cf2_conv_12(x_result_3)
        x1, x_result_3 = split(x_result_3, x_result_3.shape[1])
        x2 = x_result_3.clone()
            # --Bottleneck--
        x_result_3 = self.cf2_bottle_9(x_result_3)
            # --Bottleneck--
        x_result_3 = torch.cat((x1, x2, x_result_3), dim=1)
        x_result_3 = self.cf2_conv_13(x_result_3)
        x_result_6 = x_result_3.clone()  # ------------------------------------->

        # -----Conv_19-----
        x_result_3 = self.conv9(x_result_3)

        # -----CONCAT_Xx3-----
        x = torch.cat((x_result_3, x), dim=1)

        # -----C2F_21-----
        x = self.cf2_conv_14(x)
        x1, x = split(x, x.shape[1])
        x2 = x.clone()
            # --Bottleneck--
        x = self.cf2_bottle_10(x)
            # --Bottleneck--
        x = torch.cat((x1, x2, x), dim=1)
        x = self.cf2_conv_15(x)

        print(x_result_5.shape, x_result_6.shape, x.shape, 'RESULTS SHAPES')


        # -----Detect_5-----
        x_result_5_up, x_result_5_down = up_down_parsing(x_result_5)
        x_result_5_up = self.detect_5_up(x_result_5_up)
        x_result_5_down = self.detect_5_down(x_result_5_down)

        # -----Detect_6-----
        x_result_6_up, x_result_6_down = up_down_parsing(x_result_6)
        x_result_6_up = self.detect_6_up(x_result_6_up)
        x_result_6_down = self.detect_6_down(x_result_6_down)

        # -----Detect_x-----
        x_up, x_down = up_down_parsing(x)
        x_up = self.detect_x_up(x_up)
        x_down = self.detect_x_down(x_down)


        # print(x_result_5_up, x_result_6_up, x_up, 'RESULTS SHAPES *UP* bbox')
        up = [x_result_5_up, x_result_6_up, x_up]
        down = [x_result_5_down, x_result_6_down, x_down]


        up_down = []
        for i in range(3):
            up_down.append(torch.cat((up[i], down[i]), 1))
        print(up_down, 'up_down')

        stride = torch.Tensor([8., 16., 32.])
        # print(x_result_5_down, x_result_6_down, x_down, 'RESULTS SHAPES *DOWN* cls')
        box = torch.cat((x_result_5_up.view(1, 64, -1), x_result_6_up.view(1, 64, -1), x_up.view(1, 64, -1)), 2)
        # print(bbox.view(1, 4, 16, a))
        # x = torch.arange(16, dtype=torch.float)
        # print(x)
        # print(box, box.shape)
        b, c, a = box.shape
        # print(len(up), stride.shape)
        anchor, strides = make_anchors(up, stride)
        # print(anchor, strides)
        dfl = self.dfl(box.view(b, 4, 16, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        dbox = dist2bbox(dfl, anchor.unsqueeze(0), xywh=True, dim=1) * strides
        # print(dbox, dbox.shape, 'dbox')


        cls = torch.cat((x_result_5_down.view(1, 80, -1), x_result_6_down.view(1, 80, -1), x_down.view(1, 80, -1)), 2)
        cls = cls.sigmoid()
        # print(cls, cls.shape, 'cls res')

        dbox_cls = torch.cat((dbox, cls), 1)

        print()
        preds = coord(dbox_cls)
        if preds != None:
            for i, pred in enumerate(preds):
                img_batch = orig_img[i]
                pred[:, :4] = scale_boxes(orig_img.shape[2:], pred[:, :4], img_batch.shape)
            boxes, classes = convert_res(pred)
        else:
            boxes, classes = None, None

        # return x_down


d = D
w = W
r = R

# -----Conv_P1-----
conv0 = convolution(in_channels=3, out_channels=64*w, kernel_size=3, padding=1, stride=2)
features['conv0'] = conv0

# -----Conv_P2-----
conv1 = convolution(in_channels=64*w, out_channels=128*w, kernel_size=3, padding=1, stride=2)
features['conv1'] = conv1

# -----C2F_2-----
c2f_conv_0 = convolution(in_channels=128*w, out_channels=128*w, kernel_size=1, padding=0, stride=1)
features['c2f_conv_0'] = c2f_conv_0
n_2 = int(round(3 * d))
for i in range(n_2):
    c2f_bottle_1 = bottleneck(in_channels=64*w, out_channels=64*w, kernel_size=3, padding=1, stride=1)
    features[f'cf2_bottle_{i}'] = c2f_bottle_1
c2f_conv_1 = convolution(in_channels=192*w, out_channels=128*w, kernel_size=1, padding=0, stride=1)
features['c2f_conv_1'] = c2f_conv_1

# -----Conv_P3-----
conv3 = convolution(in_channels=128*w, out_channels=256*w, kernel_size=3, padding=1, stride=2)
features['conv3'] = conv3

# -----C2F_4-----
c2f_conv_2 = convolution(in_channels=256*w, out_channels=256*w, kernel_size=1, padding=0, stride=1)
features['c2f_conv_2'] = c2f_conv_2
n_4 = int(round(6 * d))
for i in range(n_2+1, n_4+n_2+1):
    c2f_bottle_2 = bottleneck(in_channels=128*w, out_channels=128*w, kernel_size=3, padding=1, stride=1)
    features[f'cf2_bottle_{i}'] = c2f_bottle_2
c2f_conv_3 = convolution(in_channels=512*w, out_channels=256*w, kernel_size=1, padding=0, stride=1)
features['c2f_conv_3'] = c2f_conv_3

# -----Conv_P4-----
conv5 = convolution(in_channels=256*w, out_channels=512*w, kernel_size=3, padding=1, stride=2)
features['conv5'] = conv5

# -----C2F_6-----
c2f_conv_4 = convolution(in_channels=512*w, out_channels=512*w, kernel_size=1, padding=0, stride=1)
features['c2f_conv_4'] = c2f_conv_4
n_6 = int(round(6 * d))
for i in range(n_4+n_2+1, n_6+n_4+n_2+1):
    c2f_bottle_3 = bottleneck(in_channels=256*w, out_channels=256*w, kernel_size=3, padding=1, stride=1)
    features[f'cf2_bottle_{i}'] = c2f_bottle_3
c2f_conv_5 = convolution(in_channels=1024*w, out_channels=512*w, kernel_size=1, padding=0, stride=1)
features['c2f_conv_5'] = c2f_conv_5

# -----Conv_P5-----
conv7 = convolution(in_channels=512*w, out_channels=512*w*r, kernel_size=3, padding=1, stride=2)
features['conv7'] = conv7

# -----C2F_8-----
c2f_conv_6 = convolution(in_channels=512*w*r, out_channels=512*w*r, kernel_size=1, padding=0, stride=1)
features['c2f_conv_6'] = c2f_conv_6
n_8 = int(round(3 * d))
for i in range(n_8):
    c2f_bottle_4 = bottleneck(in_channels=256*w*r, out_channels=256*w*r, kernel_size=3, padding=1, stride=1)
    features[f'cf2_bottle_{n_8+n_6+n_4+n_2+i}'] = c2f_bottle_4
c2f_conv_7 = convolution(in_channels=768*w*r, out_channels=512*w*r, kernel_size=1, padding=0, stride=1)
features['c2f_conv_7'] = c2f_conv_7

# -----SPPF-----
sppf_conv_1 = convolution(in_channels=512*w*r, out_channels=256*w*r, kernel_size=1, padding=0, stride=1)
features['sppf_conv_1'] = sppf_conv_1
maxp_0 = nn.Sequential(nn.MaxPool2d(5, 1, padding=2))
features['maxp_0'] = maxp_0
maxp_1 = nn.Sequential(nn.MaxPool2d(5, 1, padding=2))
features['maxp_1'] = maxp_1
maxp_2 = nn.Sequential(nn.MaxPool2d(5, 1, padding=2))
features['maxp_2'] = maxp_2
sppf_conv_2 = convolution(in_channels=1024*w*r, out_channels=512*w*r, kernel_size=1, padding=0, stride=1)
features['sppf_conv_2'] = sppf_conv_2

# -----UPSAMPLE_10-----
ups_10 = nn.Sequential(nn.Upsample(None, 2, 'nearest'))
features['ups_10'] = ups_10

# -----C2F_12-----
c2f_conv_8 = convolution(in_channels=512*w*(1 + r), out_channels=512*w, kernel_size=1, padding=0, stride=1) # out_channels=128?
features['c2f_conv_8'] = c2f_conv_8
n_12 = int(round(3 * d))
for i in range(n_12):
    features[f'cf2_bottle_{n_12+n_8+n_6+n_4+n_2+i}'] = bottleneck(in_channels=256*w, out_channels=256*w, kernel_size=3, padding=1, stride=1)
c2f_conv_9 = convolution(in_channels=256*w*(1 + r), out_channels=512*w, kernel_size=1, padding=0, stride=1)
features['c2f_conv_9'] = c2f_conv_9

# -----UPSAMPLE_13-----
# ups_13 = nn.Sequential(nn.ConvTranspose2d(int(512*w), int(256*w), 2, 2, 0, bias=True))
ups_13 = nn.Sequential(nn.Upsample(None, 2, 'nearest'))
features['ups_13'] = ups_13

# -----C2F_15-----
c2f_conv_10 = convolution(in_channels=256*w*(1 + r), out_channels=256*w, kernel_size=1, padding=0, stride=1)
features['c2f_conv_10'] = c2f_conv_10
n_15 = int(round(3 * d))
for i in range(n_15):
    features[f'cf2_bottle_{n_15+n_12+n_8+n_6+n_4+n_2+i}'] = bottleneck(in_channels=128*w, out_channels=128*w, kernel_size=3, padding=1, stride=1)
c2f_conv_11 = convolution(in_channels=128*w*(1 + r), out_channels=256*w, kernel_size=1, padding=0, stride=1)
features['c2f_conv_11'] = c2f_conv_11

# -----Conv_P3-----
conv8 = convolution(in_channels=256*w, out_channels=256*w, kernel_size=3, padding=1, stride=2)
features['conv8'] = conv8

# -----C2F_18-----
c2f_conv_12 = convolution(in_channels=768*w, out_channels=512*w, kernel_size=1, padding=0, stride=1)
features['c2f_conv_12'] = c2f_conv_12
n_18 = int(round(3 * d))
for i in range(n_18):
    features[f'cf2_bottle_{n_18+n_15+n_12+n_8+n_6+n_4+n_2+i}'] = bottleneck(in_channels=256*w, out_channels=256*w, kernel_size=3, padding=1, stride=1)
c2f_conv_13 = convolution(in_channels=768*w, out_channels=512*w, kernel_size=1, padding=0, stride=1)
features['c2f_conv_13'] = c2f_conv_13

# -----Conv_19-----
conv9 = convolution(in_channels=512*w, out_channels=512*w, kernel_size=3, padding=1, stride=2)
features['conv9'] = conv9

# -----C2F_21-----
c2f_conv_14 = convolution(in_channels=512*w*(1 + r), out_channels=1024*w, kernel_size=1, padding=0, stride=1)
features['c2f_conv_14'] = c2f_conv_14
n_19 = int(round(3 * d))
for i in range(n_19):
    features[f'cf2_bottle_{n_19+n_18+n_15+n_12+n_8+n_6+n_4+n_2+i}'] = bottleneck(in_channels=512*w, out_channels=512*w, kernel_size=3, padding=1, stride=1)
c2f_conv_15 = convolution(in_channels=512*w*(1 + r), out_channels=1024*w, kernel_size=1, padding=0, stride=1)
features['c2f_conv_15'] = c2f_conv_15


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

# -----DFL-----
dfl = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1, padding=0, stride=1, bias=False).requires_grad_(False)
features['dfl'] = dfl
a = torch.arange(16, dtype=torch.float)


model = Yolov8(features)  # Кастомная модель
weights = model.state_dict()

model_orig = YOLO(f'input_data/{MODEL_NAME}')  # Оригинальная модель

weights_orig = model_orig.state_dict()

my_names = list(weights.keys())
orig_values = list(weights_orig.values())
new_orig_dict = {}
for i, (key, value) in enumerate(weights_orig.items()):
    new_orig_dict[my_names[i]] = value


state_dict = OrderedDict(new_orig_dict)
model.load_state_dict(state_dict)
torch.save(model.state_dict(), f'{MAIN_DIR_NAME}/results/{ORIG_WEIGHTS}')
print('ORIG WEIGHTS SUCCESSFULLY SAVED!')
