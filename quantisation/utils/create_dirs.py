import os


def dirs(dir_names):
    paths = [f'{dir_names}', f'{dir_names}/batches', f'{dir_names}/bias_scales', f'{dir_names}/results', f'{dir_names}/results/runs_val', f'{dir_names}/first_pixel', f'{dir_names}/quant_weights_yolov8n', f'{dir_names}/quant_activations',
             f'{dir_names}/quant_activations/conv2d', f'{dir_names}/quant_activations/silu', f'{dir_names}/weights_pickle']
    try:
        for path in paths:
            if os.path.isdir(path) == False:
                os.mkdir(path)
        print('DIRS SUCCESSFULLY CREATED ✅')
    except Exception as e:
        print(f'DIRS ERROR ❌: {e}')
        pass

