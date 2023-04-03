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
from config import *

viz = Visualizer(env='VAD3', use_incoming_socket=False)

if __name__ == '__main__':
    args = option.parser.parse_args()
    config = Config(args)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    test_loader = DataLoader(Dataset(args, test_mode=True),
                             batch_size=1, shuffle=False,
                             num_workers=args.workers, pin_memory=True)
