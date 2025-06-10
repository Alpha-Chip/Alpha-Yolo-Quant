from yolov8n_quantisation.quantisation.utils.silu import *
from yolov8n_quantisation.quantisation.utils.scale import scale
from yolov8n_quantisation.quantisation.utils.rescale_coeff import requantize
from yolov8n_quantisation.quantisation.utils.quant_matrix import quant_matrix
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
from torchvision.transforms import transforms
import matplotlib.patches as patches


# --------------------------------------------------------------------
# k = 8
# max_value = 6
# orig_array = np.arange(-max_value, max_value, 0.1)
# orig_array_copy = np.copy(orig_array)
# sigmoid_orig = [sigmoid(i) for i in orig_array]
# orig_array = np.expand_dims(orig_array, (0, 1, 2))
# orig_array, _ = quant_matrix(orig_array, k)
# orig_scale = scale(max_value, 8)
#
#
# min_mae = 10000
# for max_value in range(1, 11):
#     orig_scale_conv = scale(max_value, k)
#
#     res, rescale, shift = requantize(orig_array, orig_scale, orig_scale_conv, k)
#     lookup = create_sigmoid_lookup_table(max_value, k)
#     scale_sigmoid = scale(1, k)
#     res_sigmoid = sigmoid_quant(res, lookup)
#     sigmoid_dequant = [value / scale_sigmoid for value in res_sigmoid[0][0, 0]]
#
#     if min_mae > mean_squared_error(sigmoid_orig, sigmoid_dequant):
#         min_mae = mean_squared_error(sigmoid_orig, sigmoid_dequant)
#         print(f'MAX_VALUE: {max_value} | MIN_MSE: {min_mae}')
#         plt.plot(orig_array_copy, sigmoid_orig)
#         plt.plot(orig_array_copy, sigmoid_dequant)
#         plt.title(f'{max_value}')
#         plt.show()

# --------------------------------------------------------------------

# plt.rcParams.update({'font.size': 14})  # глобальное увеличение шрифта
# fig = plt.figure(figsize=(7, 7))
# ax = fig.add_axes([0.2, 0.2, 0.6, 0.6])
# ax.plot([4, 5, 6, 7, 8, 9],
#         [0.2781145691770681, 0.27938671283311095, 0.27920500585711855, 0.2799984457329664, 0.2799572719702809, 0.27951045109981054])
#
# ax.set_xlabel(f'Максимальное значение X\nдля определния масштаба Sigmoid')
# ax.set_ylabel('mAP .50-.95')
# ax.grid(alpha=0.3)
# plt.show()

# --------------------------------------------------------------------

# Данные X
x = np.linspace(-10, 10, 500)

# Sigmoid
sigmoid = 1 / (1 + np.exp(-x))

# SiLU (x * sigmoid(x))
silu = x * sigmoid

# === График 1: Sigmoid ===
fig1 = plt.figure(figsize=(5, 5))
ax1 = fig1.add_axes([0.2, 0.2, 0.6, 0.6])
ax1.plot(x, sigmoid, linewidth=2)
ax1.set_xlabel('x', fontsize=14)
ax1.set_ylabel('Sigmoid(x)', fontsize=14)
ax1.set_title('Функция Sigmoid', fontsize=16)
ax1.tick_params(axis='both', labelsize=12)
ax1.grid(alpha=0.3)

# === График 2: SiLU ===
fig2 = plt.figure(figsize=(5, 5))
ax2 = fig2.add_axes([0.2, 0.2, 0.6, 0.6])
ax2.plot(x, silu, linewidth=2)
ax2.set_xlabel('x', fontsize=14)
ax2.set_ylabel('SiLU(x)', fontsize=14)
ax2.set_title('Функция SiLU', fontsize=16)
ax2.tick_params(axis='both', labelsize=12)
ax2.grid(alpha=0.3)

# Показываем оба окна
plt.show()

# --------------------------------------------------------------------
# orig_n = np.array([0.4868317451194475, 0.46813479636787775, 0.4456357695490293, 0.4204804779107656, 0.38918642638034273, 0.3535249518735988, 0.3073923672408093, 0.24671051165106891, 0.1597735866678504, 0.03951663865829661])
# orig_s = np.array([0.5690713620804665, 0.5481775206255155, 0.5243233241644597, 0.4995224212499402, 0.46974127300303287, 0.43062545816281644, 0.38125883888622525, 0.31207107920677696, 0.2080595292037309, 0.06518562603505237])
#
# no_channel_8_n = np.array([0.4137360931358864, 0.3933886019984594, 0.3723791191524431, 0.3484527222215369, 0.3211288265900497, 0.2892782377892, 0.24800955742450218, 0.19100426205392207, 0.11561234382295782, 0.0253294295133218])
# max_8_n = np.array([0.426,0.408,0.388,0.363,0.336,0.302,0.262,0.204,0.124,0.025])
# std_8_n = np.array([0.432,0.413,0.39,0.366,0.339,0.306,0.265,0.210,0.127,0.027])
# mm_8_n = np.array([0.463,0.443,0.419,0.392,0.361,0.324,0.276,0.212,0.123,0.025])
#
# no_channel_8_s = np.array([0.5209859378237104, 0.49986303513773056, 0.4782521850664341, 0.45419675776214363, 0.4235608346977163, 0.38746457248748234, 0.340031556919864, 0.27284869391409755, 0.17247045776832579, 0.039129178903563946])
# max_8_s = np.array([0.529,0.509,0.486,0.463,0.433,0.395,0.350,0.280,0.173,0.042])
# std_8_s = np.array([0.537,0.516,0.494,0.469,0.439,0.401,0.355,0.285,0.181,0.044])
# mm_8_s = np.array([0.554,0.533,0.509,0.482,0.451,0.41,0.358,0.285,0.175,0.044])
#
# no_channel_7_n = np.array([0.1124370789996767, 0.10572803416059644, 0.09812910635416747, 0.08961714265427088, 0.08061439363306315, 0.07004177508453537, 0.0580940001478918, 0.042928074433765305, 0.020030432291108225, 0.002453923473046114])
# max_7_n = np.array([0.15768435166900646, 0.1477914718876613, 0.13764951686972657, 0.12579266180073245, 0.11291872874519919, 0.09873684399109092, 0.08284942272577947, 0.06167646707206006, 0.030963762916274028, 0.003993470990890053])
# std_7_n = np.array([0.2834659131841278, 0.2695557998501404, 0.254300335525037, 0.2342640432628517, 0.2126610592831069, 0.18740679217589798, 0.15775154185232404, 0.12082528440330738, 0.06895380117686045, 0.011732297759811657])
# mm_7_n = np.array([0.38241333914914943, 0.36423641572466203, 0.34166754380230013, 0.31746254326400597, 0.2881103068649311, 0.2517855141061883, 0.21006915156067793, 0.15809273383302178, 0.08480677496935407, 0.014314969479668298])
#
# print(sum(mm_7_n) / len(mm_7_n))
# no_channel_7_s = np.array([0.5007138458431717, 0.4787825020868552, 0.45648544185849593, 0.42934443927826366, 0.394820299098219, 0.3560043038658232, 0.3031607669807914, 0.2286039701087399, 0.13144486441379866, 0.028910337976580807])
# max_7_s = np.array([0.3781403932431449, 0.3600757022093394, 0.3420893307306092, 0.3212170098443984, 0.2976542288889009, 0.2671274112571568, 0.22623056295494853, 0.17454600493558103, 0.10231018875966516, 0.01931869511691052])
# std_7_s = np.array([0.42769082435418426, 0.4097388142571766, 0.38875166616647244, 0.3667101206814728, 0.3408020033037084, 0.3076595522785689, 0.26402566696879287, 0.1999587677586066, 0.12056783660343515, 0.024243429382014074])
# mm_7_s = np.array([0.5275235626518243, 0.5062466187686395, 0.48336400232623805, 0.4566418454206726, 0.4221211785138864, 0.3827295883353875, 0.33101951180237454, 0.2608433756409948, 0.15413979807960354, 0.03165853003235546])
#
# # n_7 = orig_n - [no_channel_7_n, max_7_n, std_7_n, mm_7_n]
# # s_7 = orig_s - [no_channel_7_s, max_7_s, std_7_s, mm_7_s]
# # n_8 = orig_n - [no_channel_8_n, max_8_n, std_8_n, mm_8_n]
# # s_8 = orig_s - [no_channel_8_s, max_8_s, std_8_s, mm_8_s]
#
# n_7 = np.array([no_channel_7_n, max_7_n, std_7_n, mm_7_n])
# s_7 = np.array([no_channel_7_s, max_7_s, std_7_s, mm_7_s])
# n_8 = np.array([no_channel_8_n, max_8_n, std_8_n, mm_8_n])
# s_8 = np.array([no_channel_8_s, max_8_s, std_8_s, mm_8_s])
# all_graphs = {7: [n_7, s_7],
#               8: [n_8, s_8]}
# titles = ['YOLOv8n', 'YOLOv8s']
# # y_lim = [0.38, 0.075]
# y_lim = [0.55, 0.6]
#
# x = [el for el in range(10)]
# labels = ['NC', 'MAX', '3STD', 'MM']
# x_labels = [f'{el:.2f}' for el in np.linspace(0.50, 0.95, 10)]
#
# for bit_idx, bit in enumerate([7, 8]):
#     fig, ax = plt.subplots(1, 2, figsize=(12, 5))
#     for model_idx, model in enumerate(all_graphs[bit]):
#         for line in range(model.shape[0]):
#             sns.lineplot(x=x,
#                          y=model[line],
#                          ax=ax[model_idx],
#                          label=labels[line],
#                          linewidth=2.5 if line == 3 else 1.0,
#                          linestyle='-' if line == 3 else '--')
#         ax[model_idx].set_title(titles[model_idx])
#         ax[model_idx].set_xlabel('IOU Threshold')
#         # ax[model_idx].set_ylabel('AP Error')
#         ax[model_idx].set_ylabel('AP')
#         ax[model_idx].set_xticks(x, x_labels)
#         ax[model_idx].set_ylim(0.001, y_lim[bit_idx])
#         ax[model_idx].grid(alpha=0.3)
#     fig.suptitle(f'Квантизация модели в {bit} бит')
#     plt.show()


# --------------------------------------------------------------------
# coco_dataset = {
#     "0": "person",
#     "1": "bicycle",
#     "2": "car",
#     "3": "motorcycle",
#     "4": "airplane",
#     "5": "bus",
#     "6": "train",
#     "7": "truck",
#     "8": "boat",
#     "9": "traffic light",
#     "10": "fire hydrant",
#     "11": "stop sign",
#     "12": "parking meter",
#     "13": "bench",
#     "14": "bird",
#     "15": "cat",
#     "16": "dog",
#     "17": "horse",
#     "18": "sheep",
#     "19": "cow",
#     "20": "elephant",
#     "21": "bear",
#     "22": "zebra",
#     "23": "giraffe",
#     "24": "backpack",
#     "25": "umbrella",
#     "26": "handbag",
#     "27": "tie",
#     "28": "suitcase",
#     "29": "frisbee",
#     "30": "skis",
#     "31": "snowboard",
#     "32": "sports ball",
#     "33": "kite",
#     "34": "baseball bat",
#     "35": "baseball glove",
#     "36": "skateboard",
#     "37": "surfboard",
#     "38": "tennis racket",
#     "39": "bottle",
#     "40": "wine glass",
#     "41": "cup",
#     "42": "fork",
#     "43": "knife",
#     "44": "spoon",
#     "45": "bowl",
#     "46": "banana",
#     "47": "apple",
#     "48": "sandwich",
#     "49": "orange",
#     "50": "broccoli",
#     "51": "carrot",
#     "52": "hot dog",
#     "53": "pizza",
#     "54": "donut",
#     "55": "cake",
#     "56": "chair",
#     "57": "couch",
#     "58": "potted plant",
#     "59": "bed",
#     "60": "dining table",
#     "61": "toilet",
#     "62": "tv",
#     "63": "laptop",
#     "64": "mouse",
#     "65": "remote",
#     "66": "keyboard",
#     "67": "cell phone",
#     "68": "microwave",
#     "69": "oven",
#     "70": "toaster",
#     "71": "sink",
#     "72": "refrigerator",
#     "73": "book",
#     "74": "clock",
#     "75": "vase",
#     "76": "scissors",
#     "77": "teddy bear",
#     "78": "hair drier",
#     "79": "toothbrush"
# }
#
#
# cat_0_coord = np.array([259264, 177504, 335360, 329568]) - 115200 # 412.1635
# cat_0_conf = [27451 / 32767, 15] # 32767
# cat_1_coord = np.array([115936, 172288, 228800, 378784]) - 115200
# cat_1_conf = [21515 / 32767, 15]
#
#
# cat_2_coord = np.array([260352, 176416, 336032, 330240]) - 115200 # 412.1635
# cat_2_conf = [27451 / 32767, 15] # 32767
# cat_3_coord = np.array([115968, 170592, 228832, 378848]) - 115200
# cat_3_conf = [21515 / 32767, 15]
#
# cat_0_coord = np.divide(cat_0_coord, 412.1635)
# cat_1_coord = np.divide(cat_1_coord, 412.1635)
#
# cat_2_coord = np.divide(cat_2_coord, 412.1635)
# cat_3_coord = np.divide(cat_3_coord, 412.1635)
#
#
# test_transform = transforms.Compose([
#     transforms.Resize(640),
#     transforms.ToTensor()
#     ])
#
# img = Image.open('utils/cats_2_640.jpg')
# img = test_transform(img)
# img = img.unsqueeze(0).float()
# orig_img = img.detach().cpu().numpy()[0]
#
# fig, ax = plt.subplots(1)
# ax.imshow(orig_img.transpose(1, 2, 0))
# def plot_res_np(boxes, classes, orig=False):
#     if orig == False:
#         edgecolor = 'r'
#     else:
#         edgecolor = 'b'
#     w = boxes[2] - boxes[0]
#     h = boxes[3] - boxes[1]
#     c = coco_dataset[str(int(classes[1]))]
#     proba = round(float(classes[0]), 2)
#     rect = patches.Rectangle((boxes[0], boxes[1]), w, h, linewidth=1.5, edgecolor=edgecolor, facecolor='none')
#     ax.add_patch(rect)
#     ax.text(boxes[0], boxes[1], f'{c}: {proba}%', fontsize=10, color='w')
#
#
# print(cat_0_coord, cat_1_coord)
# plot_res_np(cat_0_coord, cat_0_conf)
# plot_res_np(cat_1_coord, cat_1_conf)
# plot_res_np(cat_2_coord, cat_2_conf, orig=True)
# plot_res_np(cat_3_coord, cat_3_conf, orig=True)
#
# plt.show()
