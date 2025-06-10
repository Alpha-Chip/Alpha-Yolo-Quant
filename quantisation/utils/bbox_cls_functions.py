import numpy as np


def make_anchors(feats, strides, grid_cell_offset=0.5):
    """Generate anchors from features."""
    anchor_points, stride_tensor = [], []
    dtype = feats[0].dtype
    for i, stride in enumerate(strides):
        _, _, h, w = feats[i].shape
        sx = np.arange(stop=w, dtype=dtype) + grid_cell_offset  # shift x
        sy = np.arange(stop=h, dtype=dtype) + grid_cell_offset  # shift y
        sy, sx = np.meshgrid(sy, sx)
        anchor_points.append(np.stack((sy, sx), -1).reshape(-1, 2))
        stride_tensor.append(np.full((h * w, 1), stride, dtype=dtype))
    return np.concatenate(anchor_points).transpose(), np.concatenate(stride_tensor).transpose()


def softmax(X, theta = 1.0, axis = None):
    y = np.atleast_2d(X)
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)
    y = y * float(theta)
    y = y - np.expand_dims(np.max(y, axis = axis), axis)
    y = np.exp(y)
    ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)
    p = y / ax_sum
    if len(X.shape) == 1:
        p = p.flatten()
    return p


def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """Transform distance(ltrb) to box(xywh or xyxy)."""

    lt, rb = np.split(distance, 2, axis=1)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return np.concatenate((c_xy, wh), dim)  # xywh bbox
    return np.concatenate((x1y1, x2y2), dim)  # xyxy bbox


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def nms(dets, scores, thresh):
    '''
    dets is a numpy array : num_dets, 4
    scores ia  nump array : num_dets,
    '''
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1] # get boxes with more ious first

    keep = []
    while order.size > 0:
        i = order[0] # pick maxmum iou box
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1) # maximum width
        h = np.maximum(0.0, yy2 - yy1 + 1) # maxiumum height
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def nms_quant(dets, scores, thresh):
    '''
    dets is a numpy array : num_dets, 4
    scores ia  nump array : num_dets,
    '''
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    areas = (x2 - x1 + 412) * (y2 - y1 + 412)
    order = scores.argsort()[::-1] # get boxes with more ious first

    keep = []
    while order.size > 0:
        i = order[0] # pick maxmum iou box
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 412) # maximum width
        h = np.maximum(0.0, yy2 - yy1 + 412) # maxiumum height
        inter = w * h

        inter *= 2.22

        inds = np.where(inter <= areas[i] + areas[order[1:]] - inter)[0]
        order = order[inds + 1]

        # ovr = inter / (areas[i] + areas[order[1:]] - inter)
        # inds = np.where(ovr <= thresh)[0]
        # order = order[inds + 1]

    return keep


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
    y = np.empty_like(x)  # faster than clone/copy
    dw = x[..., 2] / 2  # half-width
    dh = x[..., 3] / 2  # half-height
    y[..., 0] = x[..., 0] - dw  # top left x
    y[..., 1] = x[..., 1] - dh  # top left y
    y[..., 2] = x[..., 0] + dw  # bottom right x
    y[..., 3] = x[..., 1] + dh  # bottom right y
    return y


def coord(prediction):
    conf_thres = 0.25 # Threshold for classes 0.25   0.000000000001
    nc = 80 # num classes
    mi = 4 + nc  # mask start index
    xc = prediction[:, 4:mi].max(axis=1) > conf_thres  # candidates
    nm = prediction.shape[1] - nc - 4
    max_nms = 30000
    agnostic = False
    max_wh = 7680
    iou_thres = 0.45
    max_det = 300
    bs = prediction.shape[0]  # batch size

    prediction = prediction.transpose(0, 2, 1)
    prediction[..., :4] = xywh2xyxy(prediction[..., :4])
    output = [np.zeros((0, 6 + nm))] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[:, 2:4] < min_wh) | (x[:, 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        if nm != 0:
            box, cls, mask = np.array_split(x, [4, nc, nm], 1)
        else:
            box, cls = np.split(x, np.array([4]), 1)
            mask = np.zeros((box.shape[0], 0))
        # print(box.shape, cls.shape, mask.shape, 'zxc')

        conf = cls.max(1, keepdims=True)
        j = cls.argmax(1, keepdims=True)
        x = np.concatenate((box, conf, j, mask), 1)[conf.reshape(-1) > conf_thres] # Зачем это надо? (кв скобки)

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        if n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort[::-1][:max_nms]]  # sort by confidence and remove excess boxes


        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes

        scores = x[:, 4]  # scores

        boxes = x[:, :4] + c  # boxes (offset by class)
        i = nms(boxes, scores, iou_thres)  # NMS

        i = i[:max_det]  # limit detections
        output[xi] = x[i]
        return output


def coord_quant(prediction):
    conf_thres = 8192  # Threshold for classes 0.00000001    0.25      int8:  0     32   scale 127      32767   8192
    nc = 80  # num classes
    mi = 4 + nc  # mask start index
    xc = prediction[:, 4:mi].max(axis=1) > conf_thres  # candidates
    nm = prediction.shape[1] - nc - 4
    max_nms = 30000
    agnostic = False
    max_wh = 7680
    iou_thres = 0.45
    max_det = 300
    bs = prediction.shape[0]  # batch size

    prediction = prediction.transpose(0, 2, 1)
    prediction[..., :4] = xywh2xyxy(prediction[..., :4])
    output = [np.zeros((0, 6 + nm))] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[:, 2:4] < min_wh) | (x[:, 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        if nm != 0:
            box, cls, mask = np.array_split(x, [4, nc, nm], 1)
        else:
            box, cls = np.split(x, np.array([4]), 1)
            mask = np.zeros((box.shape[0], 0))
        # print(box.shape, cls.shape, mask.shape, 'zxc')

        conf = cls.max(1, keepdims=True)
        j = cls.argmax(1, keepdims=True)
        x = np.concatenate((box, conf, j, mask), 1)[conf.reshape(-1) > conf_thres] # Зачем это надо? (кв скобки)

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        if n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort[::-1][:max_nms]]  # sort by confidence and remove excess boxes


        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes

        scores = x[:, 4]  # scores

        boxes = x[:, :4] + c  # boxes (offset by class)
        # i = nms(boxes, scores, iou_thres)  # NMS
        i = nms_quant(boxes, scores, iou_thres)  # NMS   NEED TIME FIX
        i = np.int64(i)

        i = i[:max_det]  # limit detections
        output[xi] = x[i]
        print(output, 'OUTPUT')
        output[0][:, :4] = np.divide(output[0][:, :4], 412.1635)
        output[0][:, 4] = np.divide(output[0][:, 4], 32767.0)
        return output


def clip_boxes(boxes, shape):

    """
    Takes a list of bounding boxes and a shape (height, width) and clips the bounding boxes to the shape.

    Args:
        boxes (torch.Tensor): the bounding boxes to clip
        shape (tuple): the shape of the image

    Returns:
        (torch.Tensor | numpy.ndarray): Clipped boxes
    """

    boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[2])  # x1, x2
    boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[1])  # y1, y2
    return boxes


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
        print(img1_shape, boxes, img0_shape, 'ZXCZXCZXC')
        gain = min(img1_shape[0] / img0_shape[1], img1_shape[1] / img0_shape[2])  # gain  = old / new
        pad = (
            round((img1_shape[1] - img0_shape[2] * gain) / 2 - 0.1),
            round((img1_shape[0] - img0_shape[1] * gain) / 2 - 0.1),
        )  # wh padding
        print('GAIN PAD', gain, pad)
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
    print(boxes)
    return clip_boxes(boxes, img0_shape)


def convert_res(data):
    box = data[:, :4]
    cls = data[:, -2:]
    return box, cls
