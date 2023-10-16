import os
from datetime import datetime
import matplotlib.pyplot as plt


def create_path(task_name, option=""):
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")

    result_path  = "./result/"  + f"{current_datetime}_{task_name}_{option}"
    weights_path = "./weights/" + f"{current_datetime}_{task_name}_{option}"

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    return result_path, weights_path

def plot_or_save(frame_idx, data, name, save_path, 
                 title_fontsize=50, label_fontsize=40, ticks_fontsize=30, labelpad=20):
    plt.figure(figsize=(20, 20))
    plt.title(name    , fontsize=title_fontsize)
    plt.plot(frame_idx[:len(data)], data)

    plt.xlabel('frame', fontsize=label_fontsize, labelpad=labelpad)
    plt.ylabel(name   , fontsize=label_fontsize, labelpad=labelpad)

    plt.xticks(fontsize=ticks_fontsize)
    plt.yticks(fontsize=ticks_fontsize)

    if save_path is not None:
        plt.savefig(save_path + "/" + name + ".jpg")
    else:
        plt.show()