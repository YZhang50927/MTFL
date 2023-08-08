
import torch
import os
import numpy as np
from model import Model
import option
from utils.utils import get_feature_size
from feature_extraction.feature_reader import read_features
import time
from torchprofile import profile_macs

if __name__ == '__main__':
    args = option.parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    feature_path = os.path.join('/media/DataDrive/yiling/Test/inference/features', args.feature_type, 'Normal_Videos_576_x264.txt')
    features = read_features(feature_path, model=args.feature_type)
    features = features.unsqueeze(0).unsqueeze(0)
    data = features.to(device)
    data = data.permute(1, 0, 2, 3)
    # LOAD MODEL
    feature_size = get_feature_size(args.feature_type)
    model = Model(feature_size, args.batch_size)
    model.load_state_dict(torch.load(args.load_model))
    model.to(device).eval()
    print('Succ: load model '+str(args.load_model)+'\n')

    total_params = sum(p.numel() for p in model.parameters())  # params

    flops = 0
    time_list = []
    for i in range(1000):
        if flops == 0:
            flops = profile_macs(model, data.to(device))
        start_time = time.time()
        score_abnormal, score_normal, feat_select_abn, feat_select_normal, feat_abn_bottom, feat_select_normal_bottom, scores, \
            scores_nor_bottom, scores_nor_abn_bag, feat_magnitudes = model(inputs=data)
        end_time = time.time()
        inference_time = end_time - start_time
        time_list.append(inference_time)

    ave_time = np.mean(time_list)
    with open('/media/DataDrive/yiling/Test/inference/stats.txt', 'a') as f:
        f.write(f'Model: {args.feature_type} RTFM\n')
        f.write(f'FLOPs: {flops/1e6} M\n')
        f.write(f'Time: {ave_time * 1e6} us x{len(time_list)}\n')
        f.write(f'Total parameters: {total_params/1e6} M\n')
        f.write("\n")
