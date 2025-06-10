import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import numpy as np


# Создание словарика
# with open('coco.txt', 'r+') as f_obj:
#     line = f_obj.readlines()
#     for name in line:
#         name = name.split(':')
#         name[1] = name[1].strip()
#         new_line = f'"{name[0]}": "{name[1]}"'
#         f_obj.write(f'{new_line},\n')


coco_dataset = {
    "0": "person",
    "1": "bicycle",
    "2": "car",
    "3": "motorcycle",
    "4": "airplane",
    "5": "bus",
    "6": "train",
    "7": "truck",
    "8": "boat",
    "9": "traffic light",
    "10": "fire hydrant",
    "11": "stop sign",
    "12": "parking meter",
    "13": "bench",
    "14": "bird",
    "15": "cat",
    "16": "dog",
    "17": "horse",
    "18": "sheep",
    "19": "cow",
    "20": "elephant",
    "21": "bear",
    "22": "zebra",
    "23": "giraffe",
    "24": "backpack",
    "25": "umbrella",
    "26": "handbag",
    "27": "tie",
    "28": "suitcase",
    "29": "frisbee",
    "30": "skis",
    "31": "snowboard",
    "32": "sports ball",
    "33": "kite",
    "34": "baseball bat",
    "35": "baseball glove",
    "36": "skateboard",
    "37": "surfboard",
    "38": "tennis racket",
    "39": "bottle",
    "40": "wine glass",
    "41": "cup",
    "42": "fork",
    "43": "knife",
    "44": "spoon",
    "45": "bowl",
    "46": "banana",
    "47": "apple",
    "48": "sandwich",
    "49": "orange",
    "50": "broccoli",
    "51": "carrot",
    "52": "hot dog",
    "53": "pizza",
    "54": "donut",
    "55": "cake",
    "56": "chair",
    "57": "couch",
    "58": "potted plant",
    "59": "bed",
    "60": "dining table",
    "61": "toilet",
    "62": "tv",
    "63": "laptop",
    "64": "mouse",
    "65": "remote",
    "66": "keyboard",
    "67": "cell phone",
    "68": "microwave",
    "69": "oven",
    "70": "toaster",
    "71": "sink",
    "72": "refrigerator",
    "73": "book",
    "74": "clock",
    "75": "vase",
    "76": "scissors",
    "77": "teddy bear",
    "78": "hair drier",
    "79": "toothbrush"
}


def label(cat):
    return [coco_dataset[str(int(el))] for el in cat[0]]


def plot_res(orig_img, boxes, classes):
    fig, ax = plt.subplots(1)
    ax.imshow(orig_img[0].permute(1, 2, 0).numpy())
    boxes = boxes.numpy()
    classes = classes.numpy()
    for i in range(boxes.shape[0]):
        w = boxes[i][2] - boxes[i][0]
        h = boxes[i][3] - boxes[i][1]
        c = coco_dataset[str(int(classes[i][1]))]
        proba = round(float(classes[i][0]), 2)
        rect = patches.Rectangle((boxes[i][0], boxes[i][1]), w, h, linewidth=1.5, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(boxes[i][0], boxes[i][1], f'{c}: {proba}%', fontsize=10, color='w')
    plt.show()


def plot_res_box_classes(orig_img, boxes, classes):
    fig, ax = plt.subplots(1)
    ax.imshow(orig_img[0].permute(1, 2, 0).numpy())
    boxes = boxes.numpy()
    classes = classes.numpy()
    for i in range(boxes.shape[1]):
        w = abs(boxes[0][i][2])
        h = abs(boxes[0][i][3])

        c = coco_dataset[str(int(classes[0][i]))]
        # print(c, boxes[0][i][0], boxes[0][i][1], boxes[0][i][2], boxes[0][i][3], 'norm:', boxes[0][i][0] / 640, boxes[0][i][1] / 426)
        rect = patches.Rectangle((boxes[0][i][0], boxes[0][i][1]), w, h, linewidth=1.5, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(boxes[0][i][0], boxes[0][i][1], f'{c}', fontsize=10, color='w')
    plt.show()


def plot_res_np(orig_img, boxes, classes):
    fig, ax = plt.subplots(1)
    ax.imshow(orig_img.transpose(1, 2, 0))
    for i in range(boxes.shape[0]):
        w = boxes[i][2] - boxes[i][0]
        h = boxes[i][3] - boxes[i][1]
        c = coco_dataset[str(int(classes[i][1]))]
        proba = round(float(classes[i][0]), 2)
        rect = patches.Rectangle((boxes[i][0], boxes[i][1]), w, h, linewidth=1.5, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(boxes[i][0], boxes[i][1], f'{c}: {proba * 100}%', fontsize=10, color='w')
    plt.show()


def map_from_torch(df, ind, no_pred, boxes, classes, ann=1):
    if boxes == None:
        no_pred.append(ind)
        # df.loc[len(df)] = [int(ind), '0', 1, 1, 1, 1, 1]
        return df
    boxes = boxes.cpu().numpy().copy()
    classes = classes.cpu().numpy().copy()
    cls = []
    w = 640
    h = 640
    boxes[:, 0] = boxes[:, 0] / w
    boxes[:, 2] = boxes[:, 2] / w
    boxes[:, 1] = boxes[:, 1] / h
    boxes[:, 3] = boxes[:, 3] / h
    boxes_df = pd.DataFrame(boxes, columns=['XMin', 'YMin', 'XMax', 'YMax'])
    for i in range(boxes.shape[0]):
        c = coco_dataset[str(int(classes[i][1]))]
        cls.append(c)
    boxes_df['ImageID'] = str(int(ind))
    boxes_df['LabelName'] = cls
    if ann == 0:
        boxes_df['Conf'] = classes[:, 0]
    df = pd.concat([df, boxes_df], ignore_index=True)
    return df


def map_from_torch_ann(df, orig_img, ind, boxes, classes, ann=1):
    boxes = boxes[0].numpy().copy()
    classes = classes.numpy().copy()
    cls = []
    w = orig_img.shape[3]
    h = orig_img.shape[2]
    boxes[:, 2] = (boxes[:, 0].copy() + boxes[:, 2]) / w #w
    boxes[:, 3] = (boxes[:, 1].copy() + boxes[:, 3]) / h #h
    boxes[:, 0] = boxes[:, 0] / w #xmin
    boxes[:, 1] = boxes[:, 1] / h #ymin
    boxes_df = pd.DataFrame(boxes, columns=['XMin', 'YMin', 'XMax', 'YMax'])
    for i in range(boxes.shape[0]):
        c = coco_dataset[str(int(classes[0][i]))]
        cls.append(c)
    boxes_df['ImageID'] = str(int(ind))
    boxes_df['LabelName'] = cls
    if ann == 0:
        boxes_df['Conf'] = classes[:, 0]
    df = pd.concat([df, boxes_df], ignore_index=True)
    return df


def map_from_torch_np(df, ind, no_pred, boxes, classes, ann=1):
    if isinstance(boxes, np.ndarray):
        boxes = boxes.copy()
        classes = classes.copy()
        cls = []
        w = 640
        h = 640
        boxes[:, 0] = boxes[:, 0] / w
        boxes[:, 2] = boxes[:, 2] / w
        boxes[:, 1] = boxes[:, 1] / h
        boxes[:, 3] = boxes[:, 3] / h
        boxes_df = pd.DataFrame(boxes, columns=['XMin', 'YMin', 'XMax', 'YMax'])
        for i in range(boxes.shape[0]):
            c = coco_dataset[str(int(classes[i][1]))]
            cls.append(c)
        boxes_df['ImageID'] = str(int(ind))
        boxes_df['LabelName'] = cls
        if ann == 0:
            boxes_df['Conf'] = classes[:, 0]
        df = pd.concat([df, boxes_df], ignore_index=True)
        return df
    else:
        no_pred.append(ind)
        return df


def map_from_torch_ann_np(df, orig_img, ind, boxes, classes, ann=1):
    boxes = boxes[0].numpy().copy()
    classes = classes.numpy().copy()
    cls = []
    w = orig_img.shape[3]
    h = orig_img.shape[2]
    boxes[:, 2] = (boxes[:, 0].copy() + boxes[:, 2]) / w #w
    boxes[:, 3] = (boxes[:, 1].copy() + boxes[:, 3]) / h #h
    boxes[:, 0] = boxes[:, 0] / w #xmin
    boxes[:, 1] = boxes[:, 1] / h #ymin
    boxes_df = pd.DataFrame(boxes, columns=['XMin', 'YMin', 'XMax', 'YMax'])
    for i in range(boxes.shape[0]):
        c = coco_dataset[str(int(classes[0][i]))]
        cls.append(c)
    boxes_df['ImageID'] = str(int(ind))
    boxes_df['LabelName'] = cls
    if ann == 0:
        boxes_df['Conf'] = classes[:, 0]
    df = pd.concat([df, boxes_df], ignore_index=True)
    return df