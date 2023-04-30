import argparse

parser = argparse.ArgumentParser(description='RTFM')
parser.add_argument('--feature_type', default='mvit', help='the model used for feature extraction, i3d, c3d, or vst')
parser.add_argument('--modality', default='RGB', help='the type of the input, RGB,AUDIO, or MIX')
parser.add_argument('--gpu', default="0", type=str, choices=["0", "1"], help='gpu')
parser.add_argument('--lr', type=str, default='[0.001]*9000', help='learning rates for steps(list form)')
parser.add_argument('--batch-size', type=int, default=16, help='number of instances in a batch of data (default: 16)')
parser.add_argument('--workers', default=10, type=int, help='number of workers in dataloader')
parser.add_argument('--model-name', default='mvit_rtfm', help='name to save model')
parser.add_argument('--env', default='mvit_baseline', help='viz env')
parser.add_argument('--pretrained-ckpt', default=None, help='ckpt for pretrained model')
parser.add_argument('--num-classes', type=int, default=1, help='number of class')
parser.add_argument('--plot-freq', type=int, default=10, help='frequency of plotting (default: 10)')
parser.add_argument('--max-epoch', type=int, default=9000, help='maximum iteration to train (default: 100)')
parser.add_argument('--save_models', default='/media/DataDrive/yiling/models/RTMF', help='saving path')
parser.add_argument('--output_path', default='/media/DataDrive/yiling/results', help='result path')
parser.add_argument('--load_model', default='/media/DataDrive/yiling/Test/models/rtfm-i3d-1560.pkl', help='model path')
parser.add_argument('--test_path', default='/media/DataDrive/yiling/Test/results', help='Test results of final model')


