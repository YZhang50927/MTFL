from torch.utils.data import DataLoader
import torch.optim as optim
import torch

import sys
sys.path.append("..")
from utils.RTFM_utils import save_best_record
from model import Model
from dataset import Dataset
from train import train
from test import test
import option
from tqdm import tqdm
from utils.RTFM_utils import Visualizer
from config import *
import os


if __name__ == '__main__':
    args = option.parser.parse_args()
    config = Config(args)
    viz = Visualizer(env=args.env, use_incoming_socket=False)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    train_nloader = DataLoader(Dataset(args, test_mode=False, is_normal=True),
                               batch_size=args.batch_size, shuffle=True,
                               num_workers=args.workers, pin_memory=True, drop_last=True)
    train_aloader = DataLoader(Dataset(args, test_mode=False, is_normal=False),
                               batch_size=args.batch_size, shuffle=True,
                               num_workers=args.workers, pin_memory=True, drop_last=True)
    test_loader_UCF = DataLoader(Dataset(args, test_mode=True, test_dataset='UCF'),
                                 batch_size=1, shuffle=False,
                                 num_workers=args.workers, pin_memory=True)
    test_loader_VAD3 = DataLoader(Dataset(args, test_mode=True, test_dataset='VAD3'),
                                  batch_size=1, shuffle=False,
                                  num_workers=args.workers, pin_memory=True)


    model = Model(int(args.feature_size), args.batch_size)

    for name, value in model.named_parameters():
        print(name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    model_path = os.path.join(args.save_models, args.model_name, 'split'+args.split)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    optimizer = optim.Adam(model.parameters(),
                           lr=config.lr[0], weight_decay=0.005)

    output_path = args.output_path  # put your own path here
    acc_UCF = test(test_loader_UCF, model, 'UCF', device)
    acc_VAD3 = test(test_loader_VAD3, model, 'VAD3', device)
    test_info = {"epoch": [], "UCF_TOP1ACC": [], "VAD3_TOP1ACC": []}
    best_VAD3_ACC = -1
    best_UCF_ACC = -1

    for step in tqdm(
            range(1, args.max_epoch + 1),
            total=args.max_epoch,
            dynamic_ncols=True
    ):
        if step > 1 and config.lr[step - 1] != config.lr[step - 2]:
            for param_group in optimizer.param_groups:
                param_group["lr"] = config.lr[step - 1]

        if (step - 1) % len(train_nloader) == 0:
            loadern_iter = iter(train_nloader)

        if (step - 1) % len(train_aloader) == 0:
            loadera_iter = iter(train_aloader)

        train(loadern_iter, loadera_iter, model, args.batch_size, optimizer, viz, device)

        if step % 5 == 0 and step > 200:

            acc_UCF = test(test_loader_UCF, model, 'UCF', device)
            acc_VAD3 = test(test_loader_VAD3, model, 'VAD3', device)
            test_info["epoch"].append(step)
            test_info["UCF_TOP1ACC"].append(acc_UCF)
            test_info["VAD3_TOP1ACC"].append(acc_VAD3)

            if test_info["UCF_TOP1ACC"][-1] > best_UCF_ACC or test_info["VAD3_TOP1ACC"][-1] > best_VAD3_ACC:
                torch.save(model.state_dict(),
                           os.path.join(model_path, args.model_name + '-{}.pkl'.format(step)))
                if test_info["UCF_TOP1ACC"][-1] > best_UCF_ACC:
                    best_UCF_ACC = test_info["UCF_TOP1ACC"][-1]
                    save_path = os.path.join(output_path, args.model_name, 'split'+args.split, 'UCF')
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    save_best_record(test_info, os.path.join(save_path, '{}-step-ACC.txt'.format(step)))
                if test_info["VAD3_TOP1ACC"][-1] > best_VAD3_ACC:
                    best_VAD3_ACC = test_info["VAD3_TOP1ACC"][-1]
                    save_path = os.path.join(output_path, args.model_name, 'split'+args.split, 'VAD3')
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    save_best_record(test_info, os.path.join(save_path, '{}-step-ACC.txt'.format(step)))
    torch.save(model.state_dict(), os.path.join(model_path, args.model_name + 'final.pkl'))
