import argparse

parser = argparse.ArgumentParser(description='VST_Novel')
parser.add_argument('--lf_dir', type=str, default='/media/DataDrive/yiling/features/VST_Temporal_Variation/L64R1',
                    help='long feature path')
parser.add_argument('--mf_dir', type=str, default='/media/DataDrive/yiling/features/VST_Temporal_Variation/L32R1',
                    help='media feature path')
parser.add_argument('--sf_dir', type=str, default='/media/DataDrive/yiling/features/VST_Temporal_Variation/L8R1',
                    help='short feature path')
parser.add_argument('--feature_size', type=int, default=1024, help='VST feature size')
parser.add_argument('--seg_num', type=int, default=32, help='the number of segments')
parser.add_argument('--experiment', type=str, default='CVA_0', help='experiment name')

parser.add_argument('--gpu', default="0", type=str, choices=["0", "1"], help='gpu')
parser.add_argument('--lr', type=float, default='0.0001', help='learning rates for steps(list form)')
parser.add_argument('--batch-size', type=int, default=64, help='number of instances in a batch of data (default: 16)')
parser.add_argument('--workers', default=8, help='number of workers in dataloader')
parser.add_argument('--env', default='test', help='viz env')
parser.add_argument('--model-name', default='my_detection_model', help='name to save model')
parser.add_argument('--pretrained-ckpt', default=None, help='ckpt for pretrained model')
parser.add_argument('--plot-freq', type=int, default=10, help='frequency of plotting (default: 10)')
parser.add_argument('--max-epoch', type=int, default=2000, help='maximum iteration to train (default: 100)')
parser.add_argument('--save_models', default='/media/DataDrive/yiling/models/MyDetection', help='saving path')
parser.add_argument('--output_path', default='/media/DataDrive/yiling/results/MyDetection', help='result path')
parser.add_argument('--load_model', default='/media/DataDrive/yiling/Test/models/MTFL/my_detection_model-CVA_0_ln-565.pkl', help='model path')
parser.add_argument('--test_path', default='/media/DataDrive/yiling/Test/results/MTFL/', help='Test results of final model')
parser.add_argument('--test_class', default='all', help='The test class')


