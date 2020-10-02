import matplotlib.pyplot as plt
import os
import numpy as np

full_res_threshold = 5.4
SMALL_SIZE = 12
MEDIUM_SIZE = 16
BIGGER_SIZE = 20


# plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
# plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
# plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
# plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
# plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
# plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def get_data(metric, run_id):
    runs = os.listdir('../results')
    prefix = str(run_id).zfill(5)
    run = None
    for cur_run in runs:
        if cur_run.startswith(prefix):
            run = cur_run
            break
    assert run is not None
    metric_file_path = '../../results/' + run + '/metric-' + metric + '.txt'
    with open(metric_file_path) as file:
        line = file.readline()
        kimg = []
        score = []
        while line:
            kimg.append(int(line[17:23]) / 1000)
            score.append(float(line[50 + len(metric):-3]))
            line = file.readline()
    return kimg, score


def simple_plot(x, y, xlabel, ylabel, title, filename, y_min, y_max, ticks, logarithmic=False, full_res_line=False):
    fig, ax = plt.subplots()
    ax.plot(x, y, 'orange')
    ax.yaxis.set_ticks(np.arange(0, max(y) + 1, ticks))
    ax.xaxis.set_ticks(np.arange(0, max(x) + 1, 1.0))
    ax.set_xlim([min(x), max(x) + 1])
    ax.set_ylim([y_min, y_max])
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.grid()
    if logarithmic:
        ax.set_yscale('log')

    if full_res_line:
        ax.axvline(full_res_threshold, 0, 1, label='Full Resolution', c='green', ls='--')
        ax.legend()

    fig.savefig(filename + '.png')
    plt.show()


def compare_plot(x_base, y_base, x_compare, y_compare, xlabel, ylabel, title, filename, x_min, x_max, y_min, y_max, xticks,
                 yticks, label):
    fig, ax = plt.subplots()
    ax.plot(x_base, y_base, 'orange', label='Baseline Model')
    ax.plot(x_compare, y_compare, 'green', label=label)
    ax.xaxis.set_ticks(np.arange(x_min, x_max, xticks))
    ax.yaxis.set_ticks(np.arange(y_min, y_max, yticks))
    ax.set_xlim(x_min, x_max)
    ax.set_ylim([y_min, y_max])
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.grid()
    ax.legend(loc=3)

    fig.savefig(filename + '.png')
    plt.show()


# x_base, y_base = get_data('fid50k', 15)
# x_compare, y_compare = get_data('fid50k', 22)
# compare_plot(x_base, y_base, x_compare, y_compare, 'Million Images', 'FID',
#              'Fréchet Inception Distance', 'fid_new_training', 6, 16, 6, 9, 3, 1, 'Model without noise')

x_base, y_base = get_data('fid50k', 15)
x_compare, y_compare = get_data('fid50k', 31)
compare_plot(x_base, y_base, x_compare, y_compare, 'Million Images Seen by the Discriminator',
             'Fréchet Inception Distance',
             'Fréchet Inception Distance over the Training',
             'fid_validation', 6, 15.5, 5, 14, 3, 1, 'Model without MinibatchStdDev')

# x, y = get_data('ppl_zfull', 15)
# simple_plot(x, y, 'Million Images', 'Perceptual Path Length',
#             'Perceptual Path Length (full) in Z for the Baseline Model', 'ppl_zfull_base', 0, 250, 50)
#
# x, y = get_data('ppl_wfull', 15)
# simple_plot(x, y, 'Million Images', 'Perceptual Path Length',
#             'Perceptual Path Length (full) in W for the Baseline Model', 'ppl_wfull_base', 0, 100, 50)
#
# x, y = get_data('ppl_zend', 15)
# simple_plot(x, y, 'Million Images', 'Perceptual Path Length',
#             'Perceptual Path Length (end) in Z for the Baseline Model', 'ppl_zend_base', 0, 250, 50)
#
# x, y = get_data('ppl_wend', 15)
# simple_plot(x, y, 'Million Images', 'Perceptual Path Length',
#             'Perceptual Path Length (end) in W for the Baseline Model', 'ppl_wend_base', 0, 100, 50)
