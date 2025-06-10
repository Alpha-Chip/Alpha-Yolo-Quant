# -----CONSTANTS-----

# MODEL NAME (dir: input_data)
MODEL_NAME = 'yolov8n.pt'

# BIT SIZE
K = 8

# MAX_VALUES MODE: (DEFAULT: max. VARIANTS: mode, median, std, n=..., min_mae)
MAX_ACTIVATIONS_MODE = 'max'

# -----------------------------------------------------------------------------------
# DIR NAME
if 'n' in MODEL_NAME:
    MAIN_DIR_NAME = f'{K}_nano'
elif 's' in MODEL_NAME:
    MAIN_DIR_NAME = f'{K}_small'

# MODEL PARAMS
if 'n' in MODEL_NAME:
    D = 0.33
    W = 0.25
    R = 2.0
    detect_1_channels = 80
elif 's' in MODEL_NAME:
    D = 0.33
    W = 0.50
    R = 2.0
    detect_1_channels = 128

# WEIGHTS NAMES (dir: results)
ORIG_WEIGHTS = 'orig_weights.pickle'
BATCHNF_WEIGHTS = 'weights_batchnf.pickle'
QUANT_WEIGHTS = f'QUANT_WEIGHTS_{K}.pickle'
