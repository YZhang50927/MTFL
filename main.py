from torch.utils.data import DataLoader
import torch.optim as optim
import torch
from utils.RTFM_utils import save_best_record
from model import Model
from dataset import Dataset
from train import train
from test import test
import option
from tqdm import tqdm
from utils.RTFM_utils import Visualizer
from utils.utils import get_feature_size
from config import *

viz = Visualizer(env='VST_BASELINE', use_incoming_socket=False)

if __name__ == '__main__':
    args = option.parser.parse_args()
    config = Config(args)
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

    feature_size = get_feature_size(args.feature_type)
    model = Model(feature_size, args.batch_size)

    for name, value in model.named_parameters():
        print(name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    model_path = os.path.join(args.save_models, args.feature_type)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    optimizer = optim.Adam(model.parameters(),
                           lr=config.lr[0], weight_decay=0.005)

    test_info = {"epoch": [], "UCF_AUC": [], "VAD3_AUC": []}
    best_VAD3_AUC = -1
    best_UCF_AUC = -1
    output_path = args.output_path  # put your own path here
    _, _, auc_UCF = test(test_loader_UCF, model, 'UCF', viz, device)
    _, _, auc_VAD3 = test(test_loader_VAD3, model, 'VAD3', viz, device)

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

            fpr_UCF, tpr_UCF, auc_UCF = test(test_loader_UCF, model, 'UCF', viz, device)
            fpr_VAD3, tpr_VAD3, auc_VAD3 = test(test_loader_VAD3, model, 'VAD3', viz, device)
            test_info["epoch"].append(step)
            test_info["UCF_AUC"].append(auc_UCF)
            test_info["VAD3_AUC"].append(auc_VAD3)

            if test_info["UCF_AUC"][-1] > best_UCF_AUC or test_info["VAD3_AUC"][-1] > best_VAD3_AUC:
                torch.save(model.state_dict(),
                           os.path.join(model_path,
                                        args.model_name + '-' + args.feature_type + '-{}.pkl'.format(step)))
                if test_info["UCF_AUC"][-1] > best_UCF_AUC:
                    best_UCF_AUC = test_info["UCF_AUC"][-1]
                    save_path = os.path.join(output_path, args.feature_type, 'UCF')
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    save_best_record(test_info, os.path.join(save_path, '{}-step-AUC.txt'.format(step)))
                    np.save(os.path.join(save_path, '{}-step-fpr.npy'.format(step)), fpr_UCF)
                    np.save(os.path.join(save_path, '{}-step-tpr.npy'.format(step)), tpr_UCF)
                if test_info["VAD3_AUC"][-1] > best_VAD3_AUC:
                    best_VAD3_AUC = test_info["VAD3_AUC"][-1]
                    save_path = os.path.join(output_path, args.feature_type, 'VAD3')
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    save_best_record(test_info, os.path.join(save_path, '{}-step-AUC.txt'.format(step)))
                    np.save(os.path.join(save_path, '{}-step-fpr.npy'.format(step)), fpr_VAD3)
                    np.save(os.path.join(save_path, '{}-step-tpr.npy'.format(step)), tpr_VAD3)
    torch.save(model.state_dict(), os.path.join(model_path, args.model_name + '-' + args.feature_type + 'final.pkl'))
