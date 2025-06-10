from yolov8n_quantisation.quantisation.stage_0 import MAIN_DIR_NAME
import os
from datetime import datetime


def write_run_result(mAP, stage, comments='Default'):
    cur_time = datetime.now()
    # run_label = f'{cur_time.date().day}_{cur_time.date().month}_{cur_time.date().year}'
    if stage == 4:
        with open(f'{MAIN_DIR_NAME}/results/ORIG_MODEL_MAP.txt', 'w') as f_obj:
            f_obj.write(f'DATE: {cur_time.date().day}.{cur_time.date().month}.{cur_time.date().year} '
                        f'TIME: {cur_time.time().hour}:{cur_time.time().minute}:{cur_time.time().second}\n')
            f_obj.write(f'ORIG MODEL mAP(.50 - .95): {mAP}\n')
    elif stage == 7:
        # with open(f'{MAIN_DIR_NAME}/results/runs_val/{run_label}', 'a') as f_obj:
        with open(f'{MAIN_DIR_NAME}/results/runs_val/results.txt', 'a') as f_obj:
            f_obj.write(f'DATE: {cur_time.date().day}.{cur_time.date().month}.{cur_time.date().year} '
                        f'TIME: {cur_time.time().hour}:{cur_time.time().minute}:{cur_time.time().second}\n')
            f_obj.write(f'Comments: {comments}\n')
            f_obj.write(f'QUANT MODEL mAP(.50 - .95): {mAP}\n')
            f_obj.write(f'---------------\n')
            f_obj.write(f'\n')
