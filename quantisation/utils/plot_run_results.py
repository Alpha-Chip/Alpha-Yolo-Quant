from yolov8n_quantisation.quantisation.stage_0 import MAIN_DIR_NAME
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os


def orig_x_quant_maps():
    # last_run = os.listdir(f'{MAIN_DIR_NAME}/results/runs_val/')[-1]
    last_run = 'results.txt'
    with open(f'{MAIN_DIR_NAME}/results/runs_val/{last_run}', 'r') as f_obj:
        lines = f_obj.readlines()
        all_maps = []
        comments = []
        for line in lines:
            if 'QUANT MODEL' in line:
                map_value = line.split(': ')[1].replace('[', '')
                map_value = map_value.replace(']', '')
                map_value = [float(el) for el in map_value.split(', ')]
                all_maps.append(map_value)
            if 'Comments' in line:
                comment = line.split(': ')[1]
                comments.append(comment)
        # quant_map = all_maps[-1]
        quant_map = all_maps
    return quant_map, comments


def plot_run_results():
    quant_map, comments = orig_x_quant_maps()
    fig, ax = plt.subplots(1, 2)
    fig.tight_layout(pad=3)
    fig.set_size_inches(10, 5, forward=True)
    plot_map = []
    for ind, maps in enumerate(quant_map):
        map_50_95 = sum(maps) / len(maps)
        print(map_50_95, comments[ind])
        plot_map.append(map_50_95)
        sns.lineplot(x=np.arange(len(maps)),
                     y=maps,
                     ax=ax[0],
                     label=comments[ind])

    sns.lineplot(x=np.arange(len(plot_map)),
                y=plot_map,
                ax=ax[1])

    ax[0].set_xlabel('Iou threshold', labelpad=25)
    ax[0].set_ylabel('mAP value')
    ax[0].set_xticks(ticks=np.arange(10), labels=["{:.2f}".format(round(el, 2)) for el in np.linspace(.50, .95, 10)], fontsize=8)
    ax[0].set_title('mAP')
    ax[0].grid(alpha=0.3)
    ax[0].legend(prop={'size': 9})

    ax[1].set_xlabel('Experiment name', labelpad=1)
    ax[1].set_ylabel('mAP(.50-.95) value')
    ax[1].set_xticks(ticks=np.arange(len(quant_map)), labels=[el for el in comments], rotation=45, fontsize=8)
    ax[1].set_title('mAP(.50-.95)')
    ax[1].grid(alpha=0.3)
    plt.savefig(f'{MAIN_DIR_NAME}/results/runs_val/results', dpi=1200)
    # plt.show()
