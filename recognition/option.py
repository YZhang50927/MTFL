import argparse

############ Test args ########################
test_parser = argparse.ArgumentParser(description='MTFL_recognition_test')
# input path
test_parser.add_argument('--lf_dir', type=str, default='features/L64', help='long frame length feature path')
test_parser.add_argument('--mf_dir', type=str, default='features/L32', help='media frame length feature path')
test_parser.add_argument('--sf_dir', type=str, default='features/L8', help='short frame length feature path')
test_parser.add_argument('--test_anno', default='annotation/Anomaly_videos.txt', help='test annotation file')
test_parser.add_argument('--detection_model', default='/media/DataDrive/yiling/Test/models/MTFL/MTFL.pkl',
                         help='model path')
# output path
test_parser.add_argument('--output_dir', default='results',
                         help='The path to store the generated scores and AUC results')
# feature size depending on which feature extractor used
test_parser.add_argument('--feature_size', type=int, default=1024, help='feature dim (default: VST feature)')
test_parser.add_argument('--seg_num', type=int, default=32, help='the number of snippets')
# running cfg
test_parser.add_argument('--gpu', default="0", type=str, choices=["0", "1"], help='gpu')
test_parser.add_argument('--workers', default=8, help='number of workers in dataloader')


############ Train args ########################
train_parser = argparse.ArgumentParser(description='MTFL_recognition_train')
# input path
train_parser.add_argument('--lf_dir', type=str, default='/media/DataDrive/yiling/features/VST_Temporal_Variation/L64R1',
                          help='long feature path')
train_parser.add_argument('--mf_dir', type=str, default='/media/DataDrive/yiling/features/VST_Temporal_Variation/L32R1',
                          help='media feature path')
train_parser.add_argument('--sf_dir', type=str, default='/media/DataDrive/yiling/features/VST_Temporal_Variation/L8R1',
                          help='short feature path')
train_parser.add_argument('--train_anno', default='/media/DataDrive/yiling/annotation/VAD_train_annotation.txt',
                          help='the annotation file for training')
train_parser.add_argument('--test_anno', default='/media/DataDrive/yiling/annotation/UCF_test_annotation_with_frames.txt',
                          help='the annotation file for test')
# output path and saving info
train_parser.add_argument('--model-name', default='MTFL', help='name to save model')
train_parser.add_argument('--save_models', default='/media/DataDrive/yiling/models/MyDetection',
                          help='the path for saving models')
train_parser.add_argument('--output_dir', default='/media/DataDrive/yiling/results/MyDetection',
                          help='The path to store AUC results')
# training cfg and paras
train_parser.add_argument('--gpu', default="0", type=str, choices=["0", "1"], help='gpu id')
train_parser.add_argument('--feature_size', type=int, default=1024, help='feature dim (default: VST feature)')
train_parser.add_argument('--seg_num', type=int, default=32, help='the number of snippets')
train_parser.add_argument('--lr', type=float, default='0.0001', help='learning rates for steps(list form)')
train_parser.add_argument('--batch-size', type=int, default=64, help='batch size')
train_parser.add_argument('--workers', default=8, help='number of workers in dataloader')
train_parser.add_argument('--max-epoch', type=int, default=2000, help='maximum iteration to train (default: 100)')








parser = argparse.ArgumentParser(description='RTFM_MD_recognition')
parser.add_argument('--feature_size', type=int, default=1024, help='VST feature size')
parser.add_argument('--split', default='1', type=str, help='which split used for training and testing')
parser.add_argument('--seg_num', type=int, default=32, help='the number of segments')
parser.add_argument('--gpu', default="0", type=str, choices=["0", "1"], help='gpu')
parser.add_argument('--lr', type=float, default='0.0001', help='learning rates for steps(list form)')
parser.add_argument('--batch-size', type=int, default=32, help='number of instances in a batch of data (default: 16)')
parser.add_argument('--workers', default=8, type=int, help='number of workers in dataloader')
parser.add_argument('--env', default='MTFL_MD_recognition', help='viz env')
parser.add_argument('--model-name', default='MTFL_MD_recognition_loss', help='name to save model')
parser.add_argument('--pretrained-ckpt', default=None, help='ckpt for pretrained model')
parser.add_argument('--plot-freq', type=int, default=10, help='frequency of plotting (default: 10)')
parser.add_argument('--max-epoch', type=int, default=4000, help='maximum iteration to train (default: 100)')
parser.add_argument('--save_models', default='/media/DataDrive/yiling/models/recognition', help='saving path')
parser.add_argument('--output_path', default='/media/DataDrive/yiling/results/recognition', help='result path')
parser.add_argument('--load_model', default='/media/DataDrive/yiling/Test/models/rtfm-vst-665.pkl--', help='model path')
parser.add_argument('--test_path', default='/media/DataDrive/yiling/Test/results', help='Test results of final model')


