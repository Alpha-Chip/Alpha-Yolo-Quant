from ultralytics import YOLO
import torch
import deeplake
from torchvision.transforms import transforms
from tqdm import tqdm
from map_boxes import mean_average_precision_for_boxes
from yolov8n_quantisation.quantisation.utils.coco import *
from yolov8n_quantisation.quantisation.stage_0 import MODEL_NAME, MAIN_DIR_NAME
import warnings


model = YOLO(f'input_data/{MODEL_NAME}')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('CUDA available ✅' if torch.cuda.is_available() else 'CUDA not available ❌')
model = model.to(device)

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

print('START VALIDATION 🚀')
trainloader = val_dataset.pytorch(num_workers=0, batch_size=1, transform = {'images': tform, 'boxes': None, 'categories': None}, shuffle=False)
ann = pd.DataFrame({'ImageID': [], 'LabelName': [], 'XMin': [], 'XMax': [], 'YMin': [], 'YMax': []})
det = pd.DataFrame({'ImageID': [], 'LabelName': [], 'Conf': [], 'XMin': [], 'XMax': [], 'YMin': [], 'YMax': []})
no_pred = []

for ind, batch in enumerate(tqdm(trainloader)):
    img = batch_transform(batch['images'])
    img = img.to(device)
    results = model.predict(img, imgsz=640, conf=0.00000001)
    for el in results:
        boxes = el.boxes.xyxy
        cls = torch.unsqueeze(el.boxes.cls, 1)
        conf = torch.unsqueeze(el.boxes.conf, 1)
        classes = torch.concat((conf, cls), 1)
        ann = map_from_torch_ann(ann, batch['images'], str(ind), batch['boxes'], batch['categories'])
        det = map_from_torch(det, str(ind), no_pred, boxes, classes, ann=0)


ann.to_csv(f'{MAIN_DIR_NAME}/results/ann_orig.csv', index=False)
det.to_csv(f'{MAIN_DIR_NAME}/results/det_orig.csv', index=False)

ann = ann[['ImageID', 'LabelName', 'XMin', 'XMax', 'YMin', 'YMax']].values
det = det[['ImageID', 'LabelName', 'Conf', 'XMin', 'XMax', 'YMin', 'YMax']].values
result_ap = []
for iou_threshold in np.arange(0.5, 1, 0.05):
    mean_ap, average_precisions = mean_average_precision_for_boxes(ann, det, round(iou_threshold, 2))
    print(round(iou_threshold, 2), mean_ap)
    result_ap.append(mean_ap)
print(result_ap)
print(f'mAP .50-.95: {sum(result_ap) / len(result_ap)}')
