import numpy as np
from torchvision.transforms import transforms
import random
from PIL import Image
from yolov8n_quantisation.quantisation.utils.create_dirs import *
import time
from yolov8n_quantisation.quantisation.utils.pickle_read import *
from yolov8n_quantisation.quantisation.utils.a import a
from yolov8n_quantisation.quantisation.utils.scale import *
from yolov8n_quantisation.quantisation.utils.quant_matrix import *
from yolov8n_quantisation.quantisation.utils.quant_bias import *
from yolov8n_quantisation.quantisation.utils.im2colSOLO import *
from yolov8n_quantisation.quantisation.utils.rescale_coeff import *
from yolov8n_quantisation.quantisation.utils.silu import *
from yolov8n_quantisation.quantisation.utils.maxpooling_batch import *
from yolov8n_quantisation.quantisation.utils.bbox_cls_functions import *
from yolov8n_quantisation.quantisation.utils.coco import *
from yolov8n_quantisation.quantisation.utils.max_a import *
from yolov8n_quantisation.quantisation.utils.save_weights import *
from yolov8n_quantisation.quantisation.utils.conv2d_print_fp import *
from yolov8n_quantisation.quantisation.utils.result_txt import *
from yolov8n_quantisation.quantisation.stage_0 import MAIN_DIR_NAME, BATCHNF_WEIGHTS, K


random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


test_transform = transforms.Compose([
    transforms.Resize(640),
    transforms.ToTensor()
    ])


img = Image.open('utils/cats_2_640.jpg')
img = test_transform(img)
img = img.unsqueeze(0).float()
orig_img = img.detach().cpu().numpy()

k = K

weights_path = f'{MAIN_DIR_NAME}/results/{BATCHNF_WEIGHTS}'
weights = weights_activ(weights_path)
max_bias = {}
dir_names = MAIN_DIR_NAME
# dirs(dir_names)


def max_bias_check(max_bias_dict):
    max_value = 1
    max_name = ''
    for key, bias in max_bias_dict.items():
        if bias >= max_value:
            max_name = key
            max_value = bias


def split(x):
    # a_input = a_conv
    # scale_conv_input = scale(a_input, k)
    # x_copy = np.copy(x)
    # input, rescale, shift = requantize(x_copy, scale_input, scale_conv_input, k)
    # print(input)
    # save_txt_activations(input, layer_name, type='act_split', k=k, silu=True)
    x = np.split(x, 2, axis=1)
    x1 = x[0]
    x = x[1]
    return x1, x


def upsample(data, scale):
    return np.repeat(data, scale, axis=2).repeat(scale, axis=3)


# with open('quant.txt', 'w') as f_obj:
#     pass


memory_bit = {}
bias_scales = {}


def memory_count(filename, layer):
    memory_bit[filename] = k * layer[0] * layer[1] * layer[2] * layer[3]


def conv_quant(layer_name, input, conv, bias_conv, a_conv=0, scale_input=0, start=False, padding=1, stride=2):
    conv, conv_scale = quant_matrix(conv, k)
    conv_scale = np.transpose(conv_scale)
    if start == True:
        input, input_scale = quant_matrix(input, k, start=True)
        bias_conv_scale = np.dot(input_scale, conv_scale)
        save_txt_activations(input, 'start', dir_names, type='start_img', k=k)
    else:
        bias_conv_scale = scale_input * conv_scale
        # a_conv = a(conv)
        # scale_conv_w = scale(a_conv, k)

    conv_bias = np.array(bias_conv)
    bias = np.zeros(bias_conv.shape)
    bias = bias.transpose(1, 0, 2, 3)
    conv_bias = conv_bias.transpose(1, 0, 2, 3)
    for b_batch in range(conv_bias.shape[0]):
        for b_channel in range(conv_bias.shape[1]):
            bias[b_batch, b_channel, :, :] += quant_bias(conv_bias[b_batch, b_channel, :, :],
                                                         bias_conv_scale[b_batch, b_channel])
    bias = np.int64(bias)
    save_txt_weight(conv, bias, layer_name, type='Conv2D', k=k, dir_names=dir_names)
    # save_txt_rescale_shift(conv, layer_name, type='Conv2D', k=k, )
    conv2d(layer_name, input, conv, bias, dir_names, padding, stride)
    res = im2colzxc(input, conv, padding, stride)
    # print(input.shape, conv.shape, res.shape)
    save_in_file(dir_names, conv, f'{layer_name}_conv.pickle')
    time.sleep(1.5)
    save_in_file(dir_names, bias, f'{layer_name}_bias.pickle')
    res += bias
    print(layer_name, 'conv', res.shape)
    # save_txt_activations(res, layer_name, type='act_conv', k=k)

    # res = conv2d(input, conv, conv_bias, padding)
    scale_res = np.expand_dims(bias_conv_scale, (2, 3))
    save_bias_scales(dir_names, scale_res, f'{layer_name}_scale.pickle')
    # matrix_txt_name(res / scale_res, f'{layer_name}_conv', 'quant.txt')
    return res, scale_res


lookup = create_sigmoid_lookup_table(7, k)
def silu_quant(layer_name, res_conv, a_silu, res_scale):
    orig_scale_conv = scale(7, k)
    res_conv_copy = res_conv.copy()

    res, rescale, shift = requantize(res_conv, res_scale, orig_scale_conv, k)
    # res, orig_scale_conv = requant_matrix(res_conv, 6, res_scale, k)

    save_txt_activations(res, layer_name, dir_names, type='act_conv', k=k)
    save_txt_rescale_shift(res, rescale, shift, layer_name, dir_names, 'act_conv', k)
    add_rescale_shift(layer_name, dir_names, res_conv_copy, rescale, shift)

    res_silu = sigmoid_quant(res, lookup)
    res_silu *= res_conv_copy
    res_silu = np.int64(np.round(res_silu))
    add_silu(layer_name, res_silu, dir_names)
    # save_txt_activations(res_silu, layer_name, type='act_silu', k=k, silu=True)

    scale_silu = scale(1, k) * res_scale
    print(layer_name, 'silu', res_silu.shape)

    a_input = a_silu
    scale_silu_input = scale(a_input, k)
    input, rescale, shift = requantize(res_silu, scale_silu, scale_silu_input, k)
    # input, scale_silu_input = requant_matrix(res_silu, a_input, scale_silu, k)
    res_silu = input
    scale_silu = scale_silu_input

    save_txt_activations(input, layer_name, dir_names, type='act_silu', k=k, silu=True)
    save_txt_rescale_shift(input, rescale, shift, layer_name, dir_names, 'act_silu', k, silu=True)
    add_rescale_shift(layer_name, dir_names, input, rescale, shift)
    return res_silu, scale_silu


def conv_silu_quant(layer_name, input, conv, bias_conv, a_conv=0, scale_input=0, a_silu=0, start=False, padding=0, stride=1):
    res_conv, scale_res_conv = conv_quant(layer_name, input, conv, bias_conv, a_conv, scale_input, start, padding, stride)
    res_silu, scale_res_silu = silu_quant(layer_name, res_conv, a_silu, scale_res_conv)
    return res_silu, scale_res_silu


def up_down_parsing(x):
    x_up = np.copy(x)
    x_down = np.copy(x)
    return x_up, x_down
# def c2f_quant(input, conv, bias_conv, a_conv=0, scale_input=0, start=False, padding=0, stride=1):
#     res_silu, scale_res_silu = conv_silu_quant(input, conv, bias_conv, a_conv, scale_input, start, padding, stride)
#
#     x1, x = split(res_silu, res_silu.shape[1])
#     x2 = x.clone()
#
#     x_bottle_0 = x.clone()
#     x = conv_silu_quant(x)
#     x += x_bottle_0
#
#     return res_silu, scale_res_silu


max_a_dict = max_a(f'{MAIN_DIR_NAME}/results/max_a.txt')
# -----------Conv_P1-----------
res_conv_1, scale_res_conv_1 = conv_quant('Conv_P1', orig_img, weights['conv0.0.weight'], weights['conv0.0.bias'], max_a_dict['conv_p1'], scale_input=0, start=True)
# result_txt(res_conv_1 / scale_res_conv_1)


res_relu_1, scale_res_relu_1 = silu_quant('Conv_P1', res_conv_1, max_a_dict['conv_p2'], scale_res_conv_1)
# result_txt(res_relu_1 / scale_res_relu_1)

# a_input = max_a_dict['conv_p2']
# scale_conv_input = scale(a_input, k)
# input, rescale, shift = requantize(res_relu_1, scale_res_relu_1, scale_conv_input, k)

# -----------Conv_P2-----------
res_conv_2, scale_res_conv_2 = conv_quant('Conv_P2', res_relu_1, weights['conv1.0.weight'], weights['conv1.0.bias'], max_a_dict['conv_p2'], scale_res_relu_1)
# result_txt(res_conv_2 / scale_res_conv_2)


res_relu_2, scale_res_relu_2 = silu_quant('Conv_P2', res_conv_2, max_a_dict['conv_0_c2f'], scale_res_conv_2)
# result_txt(res_relu_2 / scale_res_relu_2)


# -----------C2F_2-----------
res_conv_0_с2f, scale_res_conv_0_с2f = conv_silu_quant('C2F_2_conv_0', res_relu_2, weights['cf2_conv_0.0.weight'], weights['cf2_conv_0.0.bias'], max_a_dict['conv_0_c2f'], scale_res_relu_2, max_a_dict['conv_b_0_c2f'])
# result_txt(res_conv_0_с2f / scale_res_conv_0_с2f)
x1, x = split(res_conv_0_с2f) # scale = scale_res_conv_0_с2f
x2 = np.copy(x) # scale = scale_res_conv_0_с2f

x_bottle_0 = np.copy(x) # scale = scale_res_conv_0_с2f
# save_txt_activations(x_bottle_0, 'concat_x_bottle', type='act_conv', k=k)
x, scale_x = conv_silu_quant('C2F_2_bottle_0', x, weights['cf2_bottle_0.0.weight'], weights['cf2_bottle_0.0.bias'], max_a_dict['conv_b_0_c2f'], scale_res_conv_0_с2f, max_a_dict['conv_b_1_c2f'], start=False, padding=1, stride=1)
# result_txt(x / scale_x)
x, scale_x = conv_silu_quant('C2F_2_bottle_1', x, weights['cf2_bottle_0.2.weight'], weights['cf2_bottle_0.2.bias'], max_a_dict['conv_b_1_c2f'], scale_x, max_a_dict['conv_b_2_c2f'], start=False, padding=1, stride=1)
# # result_txt(x / scale_x)
# # save_txt_activations(x, 'concat_x', type='act_conv', k=k)

x, rescale, shift = requantize(x, scale_x, scale_res_conv_0_с2f, k)
save_txt_activations(x, 'C2F_2_bottle_1_RESCALE', dir_names, type='act_silu', k=k, silu=True)
save_txt_rescale_shift(x, rescale, shift, 'C2F_2_bottle_1_RESCALE', dir_names, 'act_silu', k, silu=True)
x += np.int64(x_bottle_0)
save_txt_activations(x, 'C2F_2_bottle_1_SUMM', dir_names, type='act_silu', k=k, silu=True)
# result_txt(x / scale_res_conv_0_с2f)

x = np.concatenate((x1, x2, x), axis=1)
save_txt_activations(x, 'C2F_2_bottle_1_CONCAT', dir_names, type='act_silu', k=k, silu=True)
# result_txt(x / scale_x)

res_c2f_2, scale_res_c2f_2 = conv_silu_quant('C2F_2_conv_1', x, weights['cf2_conv_1.0.weight'], weights['cf2_conv_1.0.bias'], max_a_dict['conv_b_2_c2f'], scale_res_conv_0_с2f, max_a_dict['conv_p3'])
# result_txt(res_c2f_2 / scale_res_c2f_2)


# -----------Conv_P3-----------
res_silu_3, scale_res_silu_3 = conv_silu_quant('Conv_P3', res_c2f_2, weights['conv3.0.weight'], weights['conv3.0.bias'], max_a_dict['conv_p3'], scale_res_c2f_2, max_a_dict['conv_2_c2f'], start=False, padding=1, stride=2)
result_txt(res_silu_3 / scale_res_silu_3)


# -----------C2F_4----------
res_conv_2_с2f, scale_res_conv_2_с2f = conv_silu_quant('C2F_4_conv_0', res_silu_3, weights['cf2_conv_2.0.weight'], weights['cf2_conv_2.0.bias'], max_a_dict['conv_2_c2f'], scale_res_silu_3, max_a_dict['conv_b1_c2f'])
# result_txt(res_conv_2_с2f / scale_res_conv_2_с2f)
x1, x = split(res_conv_2_с2f) # scale_res_conv_2_с2f
x2 = np.copy(x) # scale_res_conv_2_с2f

x_bottle_0 = np.copy(x) # scale_res_conv_2_с2f
x, scale_x = conv_silu_quant('C2F_4_bottle_0', x, weights['cf2_bottle_2.0.weight'], weights['cf2_bottle_2.0.bias'], max_a_dict['conv_b1_c2f'], scale_res_conv_2_с2f, max_a_dict['conv_b2_c2f'], start=False, padding=1, stride=1)
x, scale_x = conv_silu_quant('C2F_4_bottle_1', x, weights['cf2_bottle_2.2.weight'], weights['cf2_bottle_2.2.bias'], max_a_dict['conv_b2_c2f'], scale_x, max_a_dict['conv_b3_c2f'], start=False, padding=1, stride=1)


x, rescale, shift = requantize(x, scale_x, scale_res_conv_2_с2f, k)
save_txt_activations(x, 'C2F_4_bottle_1_RESCALE', dir_names, type='act_silu', k=k, silu=True)
save_txt_rescale_shift(x, rescale, shift, 'C2F_4_bottle_1_RESCALE', dir_names, 'act_silu', k, silu=True)
x += np.int64(x_bottle_0)
save_txt_activations(x, 'C2F_4_bottle_1_SUMM', dir_names, type='act_silu', k=k, silu=True)


x3 = np.copy(x) # scale_res_conv_2_с2f
x_bottle_1 = np.copy(x) # scale_res_conv_2_с2f
x, scale_x = conv_silu_quant('C2F_4_bottle_2', x, weights['cf2_bottle_3.0.weight'], weights['cf2_bottle_3.0.bias'], max_a_dict['conv_b3_c2f'], scale_res_conv_2_с2f, max_a_dict['conv_b4_c2f'], start=False, padding=1, stride=1)
x, scale_x = conv_silu_quant('C2F_4_bottle_3', x, weights['cf2_bottle_3.2.weight'], weights['cf2_bottle_3.2.bias'], max_a_dict['conv_b4_c2f'], scale_x, max_a_dict['conv_b5_c2f'], start=False, padding=1, stride=1)


x, rescale, shift = requantize(x, scale_x, scale_res_conv_2_с2f, k)
save_txt_activations(x, 'C2F_4_bottle_3_RESCALE', dir_names, type='act_silu', k=k, silu=True)
save_txt_rescale_shift(x, rescale, shift, 'C2F_4_bottle_3_RESCALE', dir_names, 'act_silu', k, silu=True)
x += np.int64(x_bottle_1)
save_txt_activations(x, 'C2F_4_bottle_3_SUMM', dir_names, type='act_silu', k=k, silu=True)


x = np.concatenate((x1, x2, x3, x), axis=1)
save_txt_activations(x, 'C2F_4_bottle_3_CONCAT', dir_names, type='act_silu', k=k, silu=True)

res_c2f_4, scale_res_c2f_4 = conv_silu_quant('C2F_4_conv_1', x, weights['cf2_conv_3.0.weight'], weights['cf2_conv_3.0.bias'], max_a_dict['conv_b5_c2f'], scale_res_conv_2_с2f, max_a_dict['conv_5'])
# result_txt(res_c2f_4 / scale_res_c2f_4)
x_result_1 = np.copy(res_c2f_4)           # scale = scale_res_c2f_4
scale_result_1 = scale_res_c2f_4 # =========================================================> RESULT


# -----------Conv_P4----------
res_silu_4, scale_res_silu_4 = conv_silu_quant('Conv_P4', res_c2f_4, weights['conv5.0.weight'], weights['conv5.0.bias'], max_a_dict['conv_5'], scale_res_c2f_4, max_a_dict['cf2_conv_4'], start=False, padding=1, stride=2)
# result_txt(res_silu_4 / scale_res_silu_4)


# -----------C2F_6----------
res_conv_4_с2f, scale_res_conv_4_с2f = conv_silu_quant('C2F_6_conv_0', res_silu_4, weights['cf2_conv_4.0.weight'], weights['cf2_conv_4.0.bias'], max_a_dict['cf2_conv_4'], scale_res_silu_4, max_a_dict['cf2_bconv_4'])
# result_txt(res_conv_2_с2f / scale_res_conv_2_с2f)
x1, x = split(res_conv_4_с2f) # scale_res_conv_4_с2f
x2 = np.copy(x) # scale_res_conv_4_с2f

x_bottle_0 = np.copy(x) # scale_res_conv_4_с2f
x, scale_x = conv_silu_quant('C2F_6_bottle_0', x, weights['cf2_bottle_4.0.weight'], weights['cf2_bottle_4.0.bias'], max_a_dict['cf2_bconv_4'], scale_res_conv_4_с2f,  max_a_dict['cf2_bconv1_4'], start=False, padding=1, stride=1)
x, scale_x = conv_silu_quant('C2F_6_bottle_1', x, weights['cf2_bottle_4.2.weight'], weights['cf2_bottle_4.2.bias'], max_a_dict['cf2_bconv1_4'], scale_x, max_a_dict['cf2_bconv_5'], start=False, padding=1, stride=1)


x, rescale, shift = requantize(x, scale_x, scale_res_conv_4_с2f, k)
save_txt_activations(x, 'C2F_6_bottle_1_RESCALE', dir_names, type='act_silu', k=k, silu=True)
save_txt_rescale_shift(x, rescale, shift, 'C2F_6_bottle_1_RESCALE', dir_names, 'act_silu', k, silu=True)
x += np.int64(x_bottle_0)
save_txt_activations(x, 'C2F_6_bottle_1_SUMM', dir_names, type='act_silu', k=k, silu=True)

x3 = np.copy(x) # scale_res_conv_4_с2f
x_bottle_1 = np.copy(x) # scale_res_conv_4_с2f
x, scale_x = conv_silu_quant('C2F_6_bottle_2', x, weights['cf2_bottle_5.0.weight'], weights['cf2_bottle_5.0.bias'], max_a_dict['cf2_bconv_5'], scale_res_conv_4_с2f, max_a_dict['cf2_bconv1_5'], start=False, padding=1, stride=1)
x, scale_x = conv_silu_quant('C2F_6_bottle_3', x, weights['cf2_bottle_5.2.weight'], weights['cf2_bottle_5.2.bias'], max_a_dict['cf2_bconv1_5'], scale_x, max_a_dict['cf2_6_conv_last'], start=False, padding=1, stride=1)


x, rescale, shift = requantize(x, scale_x, scale_res_conv_4_с2f, k)
save_txt_activations(x, 'C2F_6_bottle_3_RESCALE', dir_names, type='act_silu', k=k, silu=True)
save_txt_rescale_shift(x, rescale, shift, 'C2F_6_bottle_3_RESCALE', dir_names, 'act_silu', k, silu=True)
x += np.int64(x_bottle_1)
save_txt_activations(x, 'C2F_6_bottle_3_SUMM', dir_names, type='act_silu', k=k, silu=True)


# result_txt(x / scale_res_conv_2_с2f)
# print(x1.shape, x2.shape, x3.shape, x.shape, 'c2f6')

x = np.concatenate((x1, x2, x3, x), axis=1)
save_txt_activations(x, 'C2F_6_bottle_3_CONCAT', dir_names, type='act_silu', k=k, silu=True)
# # result_txt(x / scale_x)
#
res_c2f_6, scale_res_c2f_6 = conv_silu_quant('C2F_6_conv_1', x, weights['cf2_conv_5.0.weight'], weights['cf2_conv_5.0.bias'], max_a_dict['cf2_6_conv_last'], scale_res_conv_4_с2f, max_a_dict['conv7'])
# result_txt(res_c2f_6 / scale_res_c2f_6)
x_result_2 = np.copy(res_c2f_6)           # scale = scale_res_c2f_6
scale_result_2 = scale_res_c2f_6 # =========================================================> RESULT
# print(x_result_2.shape, 'x_res_2')


# -----------Conv_P5----------
res_silu_7, scale_res_silu_7 = conv_silu_quant('Conv_P5', res_c2f_6, weights['conv7.0.weight'], weights['conv7.0.bias'], max_a_dict['conv7'], scale_res_c2f_6, max_a_dict['cf2_conv_6'], start=False, padding=1, stride=2)
# result_txt(res_silu_7 / scale_res_silu_7)


# -----------C2F_8----------
res_conv_6_с2f, scale_res_conv_6_с2f = conv_silu_quant('C2F_8_conv_0', res_silu_7, weights['cf2_conv_6.0.weight'], weights['cf2_conv_6.0.bias'], max_a_dict['cf2_conv_6'], scale_res_silu_7, max_a_dict['cf2_bottle_6'])
x1, x = split(res_conv_6_с2f) # scale_res_conv_6_с2f
x2 = np.copy(x) # scale_res_conv_6_с2f

x_bottle_0 = np.copy(x) # scale_res_conv_6_с2f
x, scale_x = conv_silu_quant('C2F_8_bottle_0', x, weights['cf2_bottle_6.0.weight'], weights['cf2_bottle_6.0.bias'], max_a_dict['cf2_bottle_6'], scale_res_conv_6_с2f, max_a_dict['cf2_bottle_61'], start=False, padding=1, stride=1)
# result_txt(x / scale_x)
x, scale_x = conv_silu_quant('C2F_8_bottle_1', x, weights['cf2_bottle_6.2.weight'], weights['cf2_bottle_6.2.bias'], max_a_dict['cf2_bottle_61'], scale_x, max_a_dict['cf2_conv_7'], start=False, padding=1, stride=1)
# result_txt(x / scale_x)


x, rescale, shift = requantize(x, scale_x, scale_res_conv_6_с2f, k)
save_txt_activations(x, 'C2F_8_bottle_1_RESCALE', dir_names, type='act_silu', k=k, silu=True)
save_txt_rescale_shift(x, rescale, shift, 'C2F_8_bottle_1_RESCALE', dir_names, 'act_silu', k, silu=True)
x += np.int64(x_bottle_0)
save_txt_activations(x, 'C2F_8_bottle_1_SUMM', dir_names, type='act_silu', k=k, silu=True)


x = np.concatenate((x1, x2, x), axis=1)
save_txt_activations(x, 'C2F_8_bottle_1_CONCAT', dir_names, type='act_silu', k=k, silu=True)
# result_txt(x / scale_x)

res_c2f_8, scale_res_c2f_8 = conv_silu_quant('C2F_8_conv_1', x, weights['cf2_conv_7.0.weight'], weights['cf2_conv_7.0.bias'], max_a_dict['cf2_conv_7'], scale_res_conv_6_с2f, max_a_dict['sppf_conv_1'])
# result_txt(res_c2f_8 / scale_res_c2f_8)


# -----------SPPF----------
res_sppf_conv_0, scale_res_sppf_conv_0 = conv_silu_quant('SPPF_conv_0', res_c2f_8, weights['sppf_conv_1.0.weight'], weights['sppf_conv_1.0.bias'], max_a_dict['sppf_conv_1'], scale_res_c2f_8, max_a_dict['sppf_conv_2'], start=False, padding=0, stride=1)
# result_txt(res_sppf_conv_0 / scale_res_sppf_conv_0)
x1 = np.copy(res_sppf_conv_0)
# result_txt(x1)
x = maxpooling(res_sppf_conv_0, kernel=5, padding=2, stride=1)
# result_txt(x)
# exit()
save_txt_activations(x, 'MAXPOOLING_X1', dir_names, type='act_silu', k=k, silu=True)
# result_txt(x / scale_res_sppf_conv_0)
x2 = np.copy(x)
x = maxpooling(x, kernel=5, padding=2, stride=1)
save_txt_activations(x, 'MAXPOOLING_X2', dir_names, type='act_silu', k=k, silu=True)
# result_txt(x / scale_res_sppf_conv_0)
x3 = np.copy(x)
x = maxpooling(x, kernel=5, padding=2, stride=1)
save_txt_activations(x, 'MAXPOOLING_X3', dir_names, type='act_silu', k=k, silu=True)
# result_txt(x / scale_res_sppf_conv_0)
# print(x1.shape, x2.shape, x3.shape, x.shape, 'SPPF')
x = np.concatenate((x1, x2, x3, x), axis=1) # scale_res_sppf_conv_0
# print(x, 'SHSHSHSHSHSHSHSHHSHSHSHSHSHSHSHSHSHSH')
# result_txt(x / scale_res_sppf_conv_0)
# result_txt(x)
res_sppf_conv_1, scale_res_sppf_conv_1 = conv_silu_quant('SPPF_conv_1', x, weights['sppf_conv_2.0.weight'], weights['sppf_conv_2.0.bias'], max_a_dict['sppf_conv_2'], scale_res_sppf_conv_0, max_a_dict['cf2_conv_8'], start=False, padding=0, stride=1)
# result_txt(res_sppf_conv_1 / scale_res_sppf_conv_1)
x_result_3 = np.copy(res_sppf_conv_1)           # scale = scale_res_sppf_conv_1
scale_result_3 = scale_res_sppf_conv_1 # =========================================================> RESULT


# -----------UPSAMPLE_10----------
x_result_3 = upsample(x_result_3, 2)
# result_txt(x_result_3 / scale_result_3)


# -----------CONCAT_2X3----------
x_result_3, rescale, shift = requantize(x_result_3, scale_result_3, scale_result_2, k)
save_txt_activations(x_result_3, 'CONCAT_2X3_REQUANT', dir_names, type='act_silu', k=k, silu=True)
save_txt_rescale_shift(x_result_3, rescale, shift, 'CONCAT_2X3_REQUANT', dir_names, 'act_silu', k, silu=True)
x_result_3 = np.concatenate((x_result_3, x_result_2), axis=1)
save_txt_activations(x_result_3, 'CONCAT_2X3_CONCAT', dir_names, type='act_silu', k=k, silu=True)
# result_txt(x_result_3)
scale_result_3 = scale_result_2
# result_txt(x_result_3 / scale_result_3)


# -----------C2F_12----------
res_conv_8_с2f, scale_res_conv_8_с2f = conv_silu_quant('C2F_12_conv_0', x_result_3, weights['cf2_conv_8.0.weight'], weights['cf2_conv_8.0.bias'], max_a_dict['cf2_conv_8'], scale_result_3, max_a_dict['cf2_conv_80'])
# result_txt(res_conv_8_с2f / scale_res_conv_8_с2f)
x1, x = split(res_conv_8_с2f) # scale_res_conv_8_с2f
x2 = np.copy(x) # scale_res_conv_8_с2f
# result_txt(res_conv_8_с2f / scale_res_conv_8_с2f)

x, scale_x = conv_silu_quant('C2F_12_bottle_0', x, weights['cf2_bottle_7.0.weight'], weights['cf2_bottle_7.0.bias'], max_a_dict['cf2_conv_80'], scale_res_conv_8_с2f, max_a_dict['cf2_conv_81'], start=False, padding=1, stride=1)
x, scale_x = conv_silu_quant('C2F_12_bottle_1', x, weights['cf2_bottle_7.2.weight'], weights['cf2_bottle_7.2.bias'], max_a_dict['cf2_conv_81'], scale_x, max_a_dict['cf2_conv_9'], start=False, padding=1, stride=1)

x, rescale, shift = requantize(x, scale_x, scale_res_conv_8_с2f, k)
save_txt_activations(x, 'C2F_12_bottle_1_REQUANT', dir_names, type='act_silu', k=k, silu=True)
save_txt_rescale_shift(x, rescale, shift, 'C2F_12_bottle_1_REQUANT', dir_names, 'act_silu', k, silu=True)
x = np.concatenate((x1, x2, x), axis=1)
save_txt_activations(x, 'C2F_12_bottle_1_CONCAT', dir_names, type='act_silu', k=k, silu=True)

res_c2f_12, scale_res_c2f_12 = conv_silu_quant('C2F_12_conv_1', x, weights['cf2_conv_9.0.weight'], weights['cf2_conv_9.0.bias'], max_a_dict['cf2_conv_9'], scale_res_conv_8_с2f, max_a_dict['cf2_conv_10'])
x_result_4 = np.copy(res_c2f_12)
scale_result_4 = scale_res_c2f_12
# result_txt(x_result_4 / scale_result_4)


# -----------UPSAMPLE_13----------
x_result_3 = upsample(res_c2f_12, 2)
scale_result_3 = scale_res_c2f_12
# result_txt(x_result_3 / scale_result_3)


# -----------CONCAT_1X3----------


x_result_3, rescale, shift = requantize(x_result_3, scale_result_3, scale_result_1, k)
save_txt_activations(x_result_3, 'CONCAT_1X3_REQUANT', dir_names, type='act_silu', k=k, silu=True)
save_txt_rescale_shift(x_result_3, rescale, shift, 'CONCAT_1X3_REQUANT', dir_names, 'act_silu', k, silu=True)
x_result_3 = np.concatenate((x_result_3, x_result_1), axis=1)
save_txt_activations(x_result_3, 'CONCAT_1X3_CONCAT', dir_names, type='act_silu', k=k, silu=True)

scale_result_3 = scale_result_1
# result_txt(x_result_3 / scale_result_3)


# -----------C2F_15----------
res_conv_10_с2f, scale_res_conv_10_с2f = conv_silu_quant('C2F_15_conv_0', x_result_3, weights['cf2_conv_10.0.weight'], weights['cf2_conv_10.0.bias'], max_a_dict['cf2_conv_10'], scale_result_3, max_a_dict['cf2_bottle_8'])
x1, x = split(res_conv_10_с2f) # scale_res_conv_10_с2f
x2 = np.copy(x) # scale_res_conv_10_с2f

x, scale_x = conv_silu_quant('C2F_15_bottle_0', x, weights['cf2_bottle_8.0.weight'], weights['cf2_bottle_8.0.bias'], max_a_dict['cf2_bottle_8'], scale_res_conv_10_с2f, max_a_dict['cf2_bottle_81'], start=False, padding=1, stride=1)
x, scale_x = conv_silu_quant('C2F_15_bottle_1', x, weights['cf2_bottle_8.2.weight'], weights['cf2_bottle_8.2.bias'], max_a_dict['cf2_bottle_81'], scale_x, max_a_dict['cf2_conv_11'], start=False, padding=1, stride=1)


x, rescale, shift = requantize(x, scale_x, scale_res_conv_10_с2f, k)
save_txt_activations(x, 'C2F_15_bottle_1_RESCALE', dir_names, type='act_silu', k=k, silu=True)
save_txt_rescale_shift(x, rescale, shift, 'C2F_15_bottle_1_RESCALE', dir_names, 'act_silu', k, silu=True)
x = np.concatenate((x1, x2, x), axis=1)
save_txt_activations(x, 'C2F_15_bottle_1_CONCAT', dir_names, type='act_silu', k=k, silu=True)


res_c2f_15, scale_res_c2f_15 = conv_silu_quant('C2F_15_conv_1', x, weights['cf2_conv_11.0.weight'], weights['cf2_conv_11.0.bias'], max_a_dict['cf2_conv_11'], scale_res_conv_10_с2f, max_a_dict['conv8'])
x_result_5 = np.copy(res_c2f_15)
scale_result_5 = scale_res_c2f_15
# result_txt(x_result_5 / scale_result_5)
# print(x_result_5.shape, 'x_res_5')


# -----------Conv_P3----------
x_result_3, scale_result_3 = conv_silu_quant('Conv_16', res_c2f_15, weights['conv8.0.weight'], weights['conv8.0.bias'], max_a_dict['conv8'], scale_res_c2f_15, max_a_dict['cf2_conv_12'], start=False, padding=1, stride=2)
# result_txt(x_result_3 / scale_result_3)


# -----------CONCAT_3X4----------

x_result_4, rescale, shift = requantize(x_result_4, scale_result_4, scale_result_3, k)
save_txt_activations(x_result_4, 'CONCAT_3X4_REQUANT', dir_names, type='act_silu', k=k, silu=True)
save_txt_rescale_shift(x_result_4, rescale, shift, 'CONCAT_3X4_REQUANT', dir_names, 'act_silu', k, silu=True)
x_result_3 = np.concatenate((x_result_3, x_result_4), axis=1)
save_txt_activations(x_result_3, 'CONCAT_3X4_CONCAT', dir_names, type='act_silu', k=k, silu=True)

# result_txt(x_result_3 / scale_result_3)


# -----------C2F_18----------
res_conv_12_с2f, scale_res_conv_12_с2f = conv_silu_quant('C2F_18_conv_0', x_result_3, weights['cf2_conv_12.0.weight'], weights['cf2_conv_12.0.bias'], max_a_dict['cf2_conv_12'], scale_result_3, max_a_dict['cf2_bottle_9'])
x1, x = split(res_conv_12_с2f) # scale_res_conv_12_с2f
x2 = np.copy(x) # scale_res_conv_12_с2f

x, scale_x = conv_silu_quant('C2F_18_bottle_0', x, weights['cf2_bottle_9.0.weight'], weights['cf2_bottle_9.0.bias'], max_a_dict['cf2_bottle_9'], scale_res_conv_12_с2f, max_a_dict['cf2_bottle_90'], start=False, padding=1, stride=1)
x, scale_x = conv_silu_quant('C2F_18_bottle_1', x, weights['cf2_bottle_9.2.weight'], weights['cf2_bottle_9.2.bias'], max_a_dict['cf2_bottle_90'], scale_x, max_a_dict['cf2_conv_13'], start=False, padding=1, stride=1)


x, rescale, shift = requantize(x, scale_x, scale_res_conv_12_с2f, k)
save_txt_activations(x, 'C2F_18_bottle_1_RESCALE', dir_names, type='act_silu', k=k, silu=True)
save_txt_rescale_shift(x, rescale, shift, 'C2F_18_bottle_1_RESCALE', dir_names, 'act_silu', k, silu=True)
x = np.concatenate((x1, x2, x), axis=1)
save_txt_activations(x, 'C2F_18_bottle_1_CONCAT', dir_names, type='act_silu', k=k, silu=True)


res_c2f_18, scale_res_c2f_18 = conv_silu_quant('C2F_18_conv_1', x, weights['cf2_conv_13.0.weight'], weights['cf2_conv_13.0.bias'], max_a_dict['cf2_conv_13'], scale_res_conv_12_с2f, max_a_dict['conv9'])
x_result_6 = np.copy(res_c2f_18)
scale_result_6 = scale_res_c2f_18
# result_txt(x_result_6 / scale_result_6)


# -----------Conv_19----------
x_result_3, scale_result_3 = conv_silu_quant('Conv_19', res_c2f_18, weights['conv9.0.weight'], weights['conv9.0.bias'], max_a_dict['conv9'], scale_res_c2f_18, max_a_dict['cf2_conv_14'], start=False, padding=1, stride=2)
# result_txt(x_result_3 / scale_result_3)


# -----CONCAT_SPPFx3-----
res_sppf_conv_1, rescale, shift = requantize(res_sppf_conv_1, scale_res_sppf_conv_1, scale_result_3, k)
save_txt_activations(res_sppf_conv_1, 'CONCAT_SPPFx3_REQUANT', dir_names, type='act_silu', k=k, silu=True)
save_txt_rescale_shift(res_sppf_conv_1, rescale, shift, 'CONCAT_SPPFx3_REQUANT', dir_names, 'act_silu', k, silu=True)
x_result_3 = np.concatenate((x_result_3, res_sppf_conv_1), axis=1)
save_txt_activations(x_result_3, 'CONCAT_SPPFx3_CONCAT', dir_names, type='act_silu', k=k, silu=True)
# result_txt(x_result_3 / scale_result_3)


# -----------C2F_21----------
res_conv_14_с2f, scale_res_conv_14_с2f = conv_silu_quant('C2F_21_conv_0', x_result_3, weights['cf2_conv_14.0.weight'], weights['cf2_conv_14.0.bias'], max_a_dict['cf2_conv_14'], scale_result_3,  max_a_dict['cf2_bottle_10'])
x1, x = split(res_conv_14_с2f) # scale_res_conv_14_с2f
x2 = np.copy(x) # scale_res_conv_14_с2f

x, scale_x = conv_silu_quant('C2F_21_bottle_0', x, weights['cf2_bottle_10.0.weight'], weights['cf2_bottle_10.0.bias'], max_a_dict['cf2_bottle_10'], scale_res_conv_14_с2f, max_a_dict['cf2_bottle_101'], start=False, padding=1, stride=1)
x, scale_x = conv_silu_quant('C2F_21_bottle_1', x, weights['cf2_bottle_10.2.weight'], weights['cf2_bottle_10.2.bias'], max_a_dict['cf2_bottle_101'], scale_x, max_a_dict['cf2_conv_15'], start=False, padding=1, stride=1)


x, rescale, shift = requantize(x, scale_x, scale_res_conv_14_с2f, k)
save_txt_activations(x, 'C2F_21_bottle_1_RESCALE', dir_names, type='act_silu', k=k, silu=True)
save_txt_rescale_shift(x, rescale, shift, 'C2F_21_bottle_1_RESCALE', dir_names, 'act_silu', k, silu=True)
x = np.concatenate((x1, x2, x), axis=1)
save_txt_activations(x, 'C2F_21_bottle_1_CONCAT', dir_names, type='act_silu', k=k, silu=True)


res_c2f_21, scale_res_c2f_21 = conv_silu_quant('C2F_21_conv_1', x, weights['cf2_conv_15.0.weight'], weights['cf2_conv_15.0.bias'], max_a_dict['cf2_conv_15'], scale_res_conv_14_с2f, max_a_dict['x_down_0'])
x = np.copy(res_c2f_21)
scale_x = scale_res_c2f_21
# result_txt(x / scale_x)



# -----Detect_5-----
x_result_5_up, x_result_5_down = up_down_parsing(x_result_5)   # scale_result_5
# -----UP-----
x_result_5_up, scale_result_5_up = conv_silu_quant('x_result_5_up_0', x_result_5_up, weights['detect_5_up.0.weight'], weights['detect_5_up.0.bias'], max_a_dict['x_result_5_up_0'], scale_result_5, max_a_dict['x_result_5_up_1'], start=False, padding=1, stride=1)
x_result_5_up, scale_result_5_up = conv_silu_quant('x_result_5_up_1', x_result_5_up, weights['detect_5_up.2.weight'], weights['detect_5_up.2.bias'], max_a_dict['x_result_5_up_1'], scale_result_5_up, max_a_dict['x_result_5_up_2'], start=False, padding=1, stride=1)
x_result_5_up, scale_result_5_up = conv_quant('x_result_5_up_2', x_result_5_up, weights['detect_5_up.4.weight'], weights['detect_5_up.4.bias'], max_a_dict['x_result_5_up_2'], scale_result_5_up, start=False, padding=0, stride=1)

## x_result_5_up, rescale, shift = requantize(x_result_5_up, scale_result_5_up, scale(1, k), k)
## save_txt_activations(x_result_5_up, 'x_result_5_up_2_REQUANT', type='act_silu', k=k, silu=True)
## save_txt_rescale_shift(x_result_5_up, rescale, shift, 'x_result_5_up_2_REQUANT', 'act_silu', k, silu=True)
## print(x_result_5_up)
## print(x_result_5_up / scale(1, k))
# result_txt(x_result_5_up / scale_result_5_up)
# -----DOWN-----
x_result_5_down, scale_result_5_down = conv_silu_quant('x_result_5_down_0', x_result_5_down, weights['detect_5_down.0.weight'], weights['detect_5_down.0.bias'], max_a_dict['x_result_5_down_0'], scale_result_5, max_a_dict['x_result_5_down_1'], start=False, padding=1, stride=1)
x_result_5_down, scale_result_5_down = conv_silu_quant('x_result_5_down_1', x_result_5_down, weights['detect_5_down.2.weight'], weights['detect_5_down.2.bias'], max_a_dict['x_result_5_down_1'], scale_result_5_down, max_a_dict['x_result_5_down_2'], start=False, padding=1, stride=1)
x_result_5_down, scale_result_5_down = conv_quant('x_result_5_down_2', x_result_5_down, weights['detect_5_down.4.weight'], weights['detect_5_down.4.bias'], max_a_dict['x_result_5_down_2'], scale_result_5_down, start=False, padding=0, stride=1)
# result_txt(x_result_5_down / scale_result_5_down)


# -----Detect_6-----
x_result_6_up, x_result_6_down = up_down_parsing(x_result_6)   # scale_result_6

# -----UP-----
x_result_6_up, scale_result_6_up = conv_silu_quant('x_result_6_up_0', x_result_6_up, weights['detect_6_up.0.weight'], weights['detect_6_up.0.bias'], max_a_dict['x_result_6_up_0'], scale_result_6, max_a_dict['x_result_6_up_1'], start=False, padding=1, stride=1)
x_result_6_up, scale_result_6_up = conv_silu_quant('x_result_6_up_1', x_result_6_up, weights['detect_6_up.2.weight'], weights['detect_6_up.2.bias'], max_a_dict['x_result_6_up_1'], scale_result_6_up, max_a_dict['x_result_6_up_2'], start=False, padding=1, stride=1)
x_result_6_up, scale_result_6_up = conv_quant('x_result_6_up_2', x_result_6_up, weights['detect_6_up.4.weight'], weights['detect_6_up.4.bias'], max_a_dict['x_result_6_up_2'], scale_result_6_up, start=False, padding=0, stride=1)
# result_txt(x_result_6_up / scale_result_6_up)


# -----DOWN-----
x_result_6_down, scale_result_6_down = conv_silu_quant('x_result_6_down_0', x_result_6_down, weights['detect_6_down.0.weight'], weights['detect_6_down.0.bias'], max_a_dict['x_result_6_down_0'], scale_result_6, max_a_dict['x_result_6_down_1'], start=False, padding=1, stride=1)
x_result_6_down, scale_result_6_down = conv_silu_quant('x_result_6_down_1', x_result_6_down, weights['detect_6_down.2.weight'], weights['detect_6_down.2.bias'], max_a_dict['x_result_6_down_1'], scale_result_6_down, max_a_dict['x_result_6_down_2'], start=False, padding=1, stride=1)
x_result_6_down, scale_result_6_down = conv_quant('x_result_6_down_2', x_result_6_down, weights['detect_6_down.4.weight'], weights['detect_6_down.4.bias'], max_a_dict['x_result_6_down_2'], scale_result_6_down, start=False, padding=0, stride=1)
# result_txt(x_result_6_down / scale_result_6_down)


# -----Detect_x-----
x_up, x_down = up_down_parsing(x)   # scale_x
# -----UP-----
x_up, scale_x_up = conv_silu_quant('x_up_0', x_up, weights['detect_x_up.0.weight'], weights['detect_x_up.0.bias'], max_a_dict['x_up_0'], scale_x, max_a_dict['x_up_1'], start=False, padding=1, stride=1)
x_up, scale_x_up = conv_silu_quant('x_up_1', x_up, weights['detect_x_up.2.weight'], weights['detect_x_up.2.bias'], max_a_dict['x_up_1'], scale_x_up, max_a_dict['x_up_2'], start=False, padding=1, stride=1)
x_up, scale_x_up = conv_quant('x_up_2', x_up, weights['detect_x_up.4.weight'], weights['detect_x_up.4.bias'], max_a_dict['x_up_2'], scale_x_up, start=False, padding=0, stride=1)

# result_txt(x_up / scale_x_up)
# -----DOWN-----
x_down, scale_x_down = conv_silu_quant('x_down_0', x_down, weights['detect_x_down.0.weight'], weights['detect_x_down.0.bias'], max_a_dict['x_down_0'], scale_x, max_a_dict['x_down_1'], start=False, padding=1, stride=1)
x_down, scale_x_down = conv_silu_quant('x_down_1', x_down, weights['detect_x_down.2.weight'], weights['detect_x_down.2.bias'], max_a_dict['x_down_1'], scale_x_down, max_a_dict['x_down_2'], start=False, padding=1, stride=1)
x_down, scale_x_down = conv_quant('x_down_2', x_down, weights['detect_x_down.4.weight'], weights['detect_x_down.4.bias'], max_a_dict['x_down_2'], scale_x_down, start=False, padding=0, stride=1)

# result_txt(x_down / scale_x_down)

# ------------------------------------BBOX_CLS------------------------------------
x_result_5_up = np.float64(x_result_5_up) / scale_result_5_up
x_result_6_up = np.float64(x_result_6_up) / scale_result_6_up
x_up = np.float64(x_up) / scale_x_up

x_result_5_down = np.float64(x_result_5_down) / scale_result_5_down
x_result_6_down = np.float64(x_result_6_down) / scale_result_6_down
x_down = np.float64(x_down) / scale_x_down

up = [x_result_5_up, x_result_6_up, x_up]


stride = np.array([8., 16., 32.])
box = np.concatenate((x_result_5_up.reshape(1, 64, -1), x_result_6_up.reshape(1, 64, -1), x_up.reshape(1, 64, -1)), 2)
# result_txt(box, 1)
# print(box.shape)

b, c, a = box.shape
anchor, strides = make_anchors(up, stride)

box = softmax(box.reshape(b, 4, 16, a).transpose((0, 2, 1, 3)), axis=1)
# conv2d('dfldfldfl_numpy', box, weights['dfl.weight'], np.zeros(box.shape), MAIN_DIR_NAME, 0, 1)
dfl = im2colzxc(box, weights['dfl.weight'], padding=0, stride=1).reshape(b, 4, a)
save_in_file(dir_names, weights['dfl.weight'], f'dfl.pickle')
result_txt(dfl, 1)

anchor = np.expand_dims(anchor, 0)
dbox = dist2bbox(dfl, anchor, xywh=True, dim=1) * strides
# print(dbox.shape)
# result_txt(dbox, 1)

cls = np.concatenate((x_result_5_down.reshape(1, 80, -1), x_result_6_down.reshape(1, 80, -1), x_down.reshape(1, 80, -1)), 2)
cls = sigmoid(cls)
# print(cls.shape)
# result_txt(cls, 1)

dbox_cls = np.concatenate((dbox, cls), 1)


preds = coord(dbox_cls)
if preds != None:
    for i, pred in enumerate(preds):
        orig_img = orig_img[i]
        pred[:, :4] = scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
    boxes, classes = convert_res(pred)
else:
    boxes, classes = None, None


print(boxes)
print(classes)
plot_res_np(orig_img, boxes, classes)
