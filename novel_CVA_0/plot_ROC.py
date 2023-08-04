import matplotlib.pyplot as plt
import option
import numpy as np
import os


def plot_roc(data, savepath):
    fig, ax = plt.subplots()
    for key in data:
        roc = data[key]
        fpr = roc[:, 0]
        tpr = roc[:, 1]
        ax.plot(fpr, tpr, label=key)
    ax.legend()
    plt.grid()
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    mx_tick = np.arange(0, 1.05, 0.1)
    my_tick = np.arange(0, 1.05, 0.1)
    plt.xticks(mx_tick)
    plt.yticks(my_tick)
    plt.title('ROC')
    plt.savefig(savepath, dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    args = option.parser.parse_args()
    VAD3_i3d_fpr = np.load(os.path.join(args.test_path, 'i3d', 'VAD3-fpr.npy'))
    VAD3_i3d_tpr = np.load(os.path.join(args.test_path, 'i3d', 'VAD3-tpr.npy'))
    UCF_i3d_fpr = np.load(os.path.join(args.test_path, 'i3d', 'UCF-fpr.npy'))
    UCF_i3d_tpr = np.load(os.path.join(args.test_path, 'i3d', 'UCF-tpr.npy'))
    VAD3_c3d_fpr = np.load(os.path.join(args.test_path, 'c3d', 'VAD3-fpr.npy'))
    VAD3_c3d_tpr = np.load(os.path.join(args.test_path, 'c3d', 'VAD3-tpr.npy'))
    UCF_c3d_fpr = np.load(os.path.join(args.test_path, 'c3d', 'UCF-fpr.npy'))
    UCF_c3d_tpr = np.load(os.path.join(args.test_path, 'c3d', 'UCF-tpr.npy'))

    data = {'VAD3_i3d': np.hstack((VAD3_i3d_fpr.reshape((-1, 1)), VAD3_i3d_tpr.reshape(-1, 1))),
            'UCF_i3d': np.hstack((UCF_i3d_fpr.reshape((-1, 1)), UCF_i3d_tpr.reshape(-1, 1))),
            'VAD3_c3d': np.hstack((VAD3_c3d_fpr.reshape((-1, 1)), VAD3_c3d_tpr.reshape(-1, 1))),
            'UCF_c3d': np.hstack((UCF_c3d_fpr.reshape((-1, 1)), UCF_c3d_tpr.reshape(-1, 1)))}

    save_path = os.path.join(args.test_path, 'roc.png')
    plot_roc(data, save_path)
