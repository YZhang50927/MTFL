from torch.utils.data import DataLoader
import torch
import os
import numpy as np
from utils.RTFM_utils import save_best_record
from model import Model
from dataset import Dataset
from test import test
import option
from utils.RTFM_utils import Visualizer
from utils.utils import get_feature_size

viz = Visualizer(env='RTMF_TEST', use_incoming_socket=False)

if __name__ == '__main__':
    args = option.parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_loader_UCF = DataLoader(Dataset(args, test_mode=True, test_dataset='UCF'),
                                 batch_size=1, shuffle=False,
                                 num_workers=args.workers, pin_memory=True)
    test_loader_VAD3 = DataLoader(Dataset(args, test_mode=True, test_dataset='VAD3'),
                                  batch_size=1, shuffle=False,
                                  num_workers=args.workers, pin_memory=True)
    # LOAD MODEL
    feature_size = get_feature_size(args.feature_type)
    model = Model(feature_size, args.batch_size)
    model.load_state_dict(torch.load(args.load_model))
    model.to(device).eval()
    print('Succ: load model '+str(args.load_model)+'\n')
    test_info = {"Feature": [], "UCF_AUC": [], "VAD3_AUC": []}

    # Test
    fpr_UCF, tpr_UCF, auc_UCF = test(test_loader_UCF, model, 'UCF', viz, device)
    fpr_VAD3, tpr_VAD3, auc_VAD3 = test(test_loader_VAD3, model, 'VAD3', viz, device)

    test_info["Feature"].append(args.feature_type)
    test_info["UCF_AUC"].append(auc_UCF)
    test_info["VAD3_AUC"].append(auc_VAD3)

    save_path = os.path.join(args.test_path, args.feature_type)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_best_record(test_info, os.path.join(save_path, '{}-AUC.txt'.format(args.feature_type)))
    np.save(os.path.join(save_path, 'VAD3-fpr.npy'), fpr_VAD3)
    np.save(os.path.join(save_path, 'VAD3-tpr.npy'), tpr_VAD3)
    np.save(os.path.join(save_path, 'UCF-fpr.npy'), fpr_UCF)
    np.save(os.path.join(save_path, 'UCF-tpr.npy'), tpr_UCF)
