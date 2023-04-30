import matplotlib.pyplot as plt
import option
import numpy as np
import os


def plot_roc(data, savepath, title):
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
    plt.title(title)
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
    VAD3_sf_fpr = np.load(os.path.join(args.test_path, 'slowfast', 'VAD3-fpr.npy'))
    VAD3_sf_tpr = np.load(os.path.join(args.test_path, 'slowfast', 'VAD3-tpr.npy'))
    UCF_sf_fpr = np.load(os.path.join(args.test_path, 'slowfast', 'UCF-fpr.npy'))
    UCF_sf_tpr = np.load(os.path.join(args.test_path, 'slowfast', 'UCF-tpr.npy'))
    VAD3_mvit_fpr = np.load(os.path.join(args.test_path, 'mViTv2', 'VAD3-fpr.npy'))
    VAD3_mvit_tpr = np.load(os.path.join(args.test_path, 'mViTv2', 'VAD3-tpr.npy'))
    UCF_mvit_fpr = np.load(os.path.join(args.test_path, 'mViTv2', 'UCF-fpr.npy'))
    UCF_mvit_tpr = np.load(os.path.join(args.test_path, 'mViTv2', 'UCF-tpr.npy'))
    VAD3_vst_fpr = np.load(os.path.join(args.test_path, 'vst', 'VAD3-fpr.npy'))
    VAD3_vst_tpr = np.load(os.path.join(args.test_path, 'vst', 'VAD3-tpr.npy'))
    UCF_vst_fpr = np.load(os.path.join(args.test_path, 'vst', 'UCF-fpr.npy'))
    UCF_vst_tpr = np.load(os.path.join(args.test_path, 'vst', 'UCF-tpr.npy'))

    data_VAD3 = {'i3d': np.hstack((VAD3_i3d_fpr.reshape((-1, 1)), VAD3_i3d_tpr.reshape(-1, 1))),
                 'c3d': np.hstack((VAD3_c3d_fpr.reshape((-1, 1)), VAD3_c3d_tpr.reshape(-1, 1))),
                 'sf': np.hstack((VAD3_sf_fpr.reshape((-1, 1)), VAD3_sf_tpr.reshape(-1, 1))),
                 'mvitv2': np.hstack((VAD3_mvit_fpr.reshape((-1, 1)), VAD3_mvit_tpr.reshape(-1, 1))),
                 'vst': np.hstack((VAD3_vst_fpr.reshape((-1, 1)), VAD3_vst_tpr.reshape(-1, 1))),
                 }
    data_UCF = {'i3d': np.hstack((UCF_i3d_fpr.reshape((-1, 1)), UCF_i3d_tpr.reshape(-1, 1))),
                'c3d': np.hstack((UCF_c3d_fpr.reshape((-1, 1)), UCF_c3d_tpr.reshape(-1, 1))),
                'sf': np.hstack((UCF_sf_fpr.reshape((-1, 1)), UCF_sf_tpr.reshape(-1, 1))),
                'mvitv2': np.hstack((UCF_mvit_fpr.reshape((-1, 1)), UCF_mvit_tpr.reshape(-1, 1))),
                'vst': np.hstack((UCF_vst_fpr.reshape((-1, 1)), UCF_vst_tpr.reshape(-1, 1))),
                }

    save_path = os.path.join(args.test_path, 'roc_VAD3.png')
    plot_roc(data_VAD3, save_path, 'ROC (VAD3)')
    save_path = os.path.join(args.test_path, 'roc_UCF.png')
    plot_roc(data_UCF, save_path, 'ROC (UCF)')

