import pandas as pd
import os
from yolov8n_quantisation.quantisation.stage_0 import MAIN_DIR_NAME, MAX_ACTIVATIONS_MODE
from yolov8n_quantisation.quantisation.utils.stage_5_common_func import n_max, load_from_file, n_max_plus_std, quant_matrix, write_best_koeff, create_std_koef, n_update_std
import torch
from yolov8n_quantisation.quantisation.utils.max_a import max_a
import numpy as np
from tqdm import tqdm


max_a_dict = {}
with open(f'{MAIN_DIR_NAME}/results/max_a_all.txt', 'r') as f_obj:
    lines = f_obj.readlines()
    for ind, el in enumerate(lines):
        row_mass = el.split(': ')
        key = row_mass[0]
        value = row_mass[1].replace('[', '')
        value = value.replace(']', '')
        value = [tnsr for tnsr in value.split(', ')]
        value = [tnsr.replace('tensor(', '') for tnsr in value]
        value = [tnsr.replace(')', '') for tnsr in value]
        res_value = []
        for tnsr in value:
            if "device='cuda:0'" not in tnsr:
                res_value.append(float(tnsr))
        max_a_dict[key] = res_value


df = pd.DataFrame(max_a_dict)


if MAX_ACTIVATIONS_MODE.lower() != 'min_mae':
    n_max(df, MAX_ACTIVATIONS_MODE)
else:
    all_layers = os.listdir(f'{MAIN_DIR_NAME}/batches')

    with open(f'{MAIN_DIR_NAME}/results/best_koeff.txt', 'w') as f:
        pass

    all_koefs = {}
    for layer_name in all_layers:
        all_files = os.listdir(f'{MAIN_DIR_NAME}/batches/{layer_name}')
        all_files_split = [el.split('_')[1] for el in all_files]
        all_files_split = [el.split('.')[0] for el in all_files_split]
        res = []
        batch = 0
        for _ in range(10):
            for i in range(0, batch + 500):
                for ind, el in enumerate(all_files_split):
                    if str(i) == el:
                        res.append(all_files[ind])

            zero_file = load_from_file(file_name=res[0], layer_name=layer_name)[0].to(torch.device('cpu'))
            activ_files = torch.zeros((500, zero_file.shape[0], zero_file.shape[1], zero_file.shape[2]))

            for ind, filename in enumerate(tqdm(res)):
                if ind == 500:
                    break
                activation = load_from_file(file_name=filename, layer_name=layer_name)[0].to(torch.device('cpu'))
                activ_files[ind] += activation

            koef = np.linspace(-2, 4, 50)
            for target_column in list(df)[1:]:
                mae_koef = 99999999999
                best_koef = 99999999999
                if target_column == layer_name:
                    for _, i in enumerate(tqdm(koef)):
                        n_max_plus_std(df, target_column, koeff_std=i, n=0)
                        max_a_dict = max_a(f'{MAIN_DIR_NAME}/results/max_a.txt')
                        a = max_a_dict[layer_name]
                        activ_files_q, activ_files_q_scale = quant_matrix(torch.clone(activ_files), a)
                        activ_dequant = activ_files_q / activ_files_q_scale

                        mae = torch.abs(torch.sum(activ_files - activ_dequant) / (
                                    activ_files.shape[0] * activ_files.shape[1] * activ_files.shape[2] *
                                    activ_files.shape[3]))
                        if mae.item() <= mae_koef:
                            print(i, mae)
                            mae_koef = mae.item()
                            best_koef = i
                        if (layer_name, i) not in all_koefs.keys():
                            all_koefs[(layer_name, i)] = [mae]
                        else:
                            all_koefs[(layer_name, i)].append(mae)
                    write_best_koeff(layer_name, best_koef)
            batch += 500

    create_std_koef()
    n_update_std(df)
