from yolov8n_quantisation.quantisation.stage_0 import MAIN_DIR_NAME, MAX_ACTIVATIONS_MODE, K
from yolov8n_quantisation.quantisation.utils.max_a import max_a
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import pickle
import gzip


def n_max(df, numb_type='max', n=1):
    with open(f'{MAIN_DIR_NAME}/results/max_a.txt', 'w') as f_obj:
        f_obj.write(f'start: 1.0\n')
        for column_name in list(df)[1:]:
            if numb_type == 'max':
                n_max_value = df[column_name].abs().max()
            elif numb_type == 'mode':
                n_max_value = df[column_name].value_counts().index[0]
            elif numb_type == 'median':
                n_max_value = df[column_name].median()
            elif numb_type == 'std':
                n_max_value = df[column_name].mean() + 3 * df[column_name].describe().loc['std']
            else:
                max_mass = df[column_name].sort_values().unique()
                n_max_value = max_mass[-n-1]
            f_obj.write(f'{column_name}: {n_max_value}\n')


def n_max_plus_std(df, target_column, koeff_std=1, n=1):
    with open(f'{MAIN_DIR_NAME}/results/max_a.txt', 'w') as f_obj:
        f_obj.write(f'start: 1.0\n')
        for column_name in list(df)[1:]:
            if column_name != target_column:
                n_max_value = df[column_name].mean() + 3 * df[column_name].describe().loc['std']
            else:
                std = df[target_column].mean() + koeff_std * df[target_column].describe().loc["std"]
                n_max_value = std
            f_obj.write(f'{column_name}: {n_max_value}\n')


def load_from_file(file_name, layer_name):
    return pickle.load(gzip.open(f'{MAIN_DIR_NAME}/batches/{layer_name}/{file_name}', 'rb'))


def write_best_koeff(name, koeff):
    with open(f'{MAIN_DIR_NAME}/best_koeff.txt', 'a') as f_obj:
        f_obj.write(f'{name}: {koeff}\n')


def new_clip(matrix, a):
    matrix[matrix > a] = a
    matrix[matrix < -a] = -a
    return matrix


def quant_matrix(matrix, a):
    scale = (2 ** (K - 1) - 1) / a
    matrix = new_clip(matrix, a)
    matrix *= scale
    matrix = torch.round(matrix).to(torch.int64)
    return matrix, scale


def create_std_koef():
    std_koef = {}
    with open(f'{MAIN_DIR_NAME}/results/best_koeff.txt', 'r') as f_obj:
        lines = f_obj.readlines()
        for line in lines:
            line = line.strip()
            line_split = line.split(': ')
            key = line_split[0]
            value = float(line_split[1])
            if key not in std_koef.keys():
                std_koef[key] = [value]
            else:
                std_koef[key].append(value)
        for key, value in std_koef.items():
            std_koef[key] = sum(value) / len(value)

    with open(f'{MAIN_DIR_NAME}/results/std_koeff_update.txt', 'w') as f_obj:
        f_obj.write('conv_p1: 3\n')
        for key, value in std_koef.items():
            f_obj.write(f'{key}: {value}\n')


def n_update_std(df):
    new_koef = {}
    with open(f'{MAIN_DIR_NAME}/results/std_koeff_update.txt', 'r') as f_obj:
        lines = f_obj.readlines()
        for line in lines:
            line = line.strip()
            key = line.split(': ')[0]
            value = float(line.split(': ')[1])
            new_koef[key] = value
    with open(f'{MAIN_DIR_NAME}/results/max_a.txt', 'w') as f_obj:
        f_obj.write(f'start: 1.0\n')
        for column_name in list(df)[1:]:
            print(column_name, new_koef[column_name])
            std = df[column_name].mean() + new_koef[column_name] * df[column_name].describe().loc["std"]
            n_max_value = std
            f_obj.write(f'{column_name}: {n_max_value}\n')

