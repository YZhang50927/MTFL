import argparse

parser = argparse.ArgumentParser(description='RTFM')
parser.add_argument('--feature_type', default='i3d', help='the model used for feature extraction, i3d, c3d, or ViT')
parser.add_argument('--modality', default='RGB', help='the type of the input, RGB,AUDIO, or MIX')
parser.add_argument('--gpus', default=0, type=int, choices=[0], help='gpus')
parser.add_argument('--lr', type=str, default='[0.001]*3000', help='learning rates for steps(list form)')
parser.add_argument('--batch-size', type=int, default=16, help='number of instances in a batch of data (default: 16)')
parser.add_argument('--workers', default=10, help='number of workers in dataloader')
parser.add_argument('--model-name', default='rtfm', help='name to save model')
parser.add_argument('--pretrained-ckpt', default=None, help='ckpt for pretrained model')
parser.add_argument('--num-classes', type=int, default=1, help='number of class')
parser.add_argument('--plot-freq', type=int, default=10, help='frequency of plotting (default: 10)')
parser.add_argument('--max-epoch', type=int, default=3000, help='maximum iteration to train (default: 100)')
parser.add_argument('--save_models', default='/media/mount_loc/yiling/models/baseline_RTMF', help='saving path')
parser.add_argument('--output_path', default='/media/mount_loc/yiling/results', help='result path')

