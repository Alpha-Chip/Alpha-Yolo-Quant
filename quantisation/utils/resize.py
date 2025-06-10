import torch.nn.functional as F
import math


def scale_img(img, same_shape=False, gs=32):
    """Scales and pads an image tensor of shape img(bs,3,y,x) based on given ratio and grid size gs, optionally
    retaining the original shape.
    """
    h, w = img.shape[2:]
    rw = 640 / w
    rh = 640 / h
    s = (int(h * rh), int(w * rw))  # new size
    img = F.interpolate(img, size=s, mode="bilinear", align_corners=False)  # resize
    if not same_shape:  # pad/crop img
        h = math.ceil(h * rh / gs) * gs
        w = math.ceil(w * rw / gs) * gs
    return F.pad(img, [0, w - s[1], 0, h - s[0]], value=0.447)  # value = imagenet mean
