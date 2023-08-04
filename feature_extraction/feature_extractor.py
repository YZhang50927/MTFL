import argparse
import logging
import os
from os import path, mkdir

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import time
from torchprofile import profile_macs

from video_loader import VideoIter
import sys

sys.path.append('..')
from utils.utils import register_logger, get_torch_device
from utils import transforms_video
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

# Transformer
from mmcv import Config, DictAction
from mmaction.models import build_model
from mmcv.runner import get_dist_info, init_dist, load_checkpoint

import warnings

warnings.filterwarnings("ignore", message="The pts_unit 'pts' gives wrong results. Please use pts_unit 'sec'.")
warnings.filterwarnings('ignore', message='No handlers found: "aten::pad". Skipped.')


def get_args():
    parser = argparse.ArgumentParser(description="Transformer Feature Extractor Parser")
    # io
    parser.add_argument('--dataset_path', default='/media/DataDrive/yiling/VAD3',
                        help="path to dataset")
    parser.add_argument('--clip_length', type=int, default=16,
                        help="define the length of each input sample.")
    parser.add_argument('--num_workers', type=int, default=0,
                        help="define the number of workers used for loading the videos")
    parser.add_argument('--frame_interval', type=int, default=1,
                        help="define the sampling interval between frames.")
    parser.add_argument('--log_every', type=int, default=50,
                        help="log the writing of clips every n steps.")
    parser.add_argument('--log_file', type=str,
                        help="set logging file.")
    parser.add_argument('--save_dir', type=str, default="/media/DataDrive/yiling/features/VST_Temporal_Variation/L16R1",
                        help="set output root for the features.")
    parser.add_argument('--use_splits', type=bool, default=False,
                        help="use full anomalous data or splits")
    parser.add_argument('--gpu', type=int, default=0, help="gpu id")

    # optimization
    parser.add_argument('--batch_size', type=int, default=16,  # default 16
                        help="batch size")

    # model
    parser.add_argument('--model_type', default='swinB',
                        type=str,
                        # required=True,
                        help="type of feature extractor",
                        choices=['c3d', 'i3d', 'swinB'])
    parser.add_argument('--pretrained_3d',
                        default='/media/DataDrive/yiling/models/VST_finetune/hflip_speed_120_2d/best_top1_acc_epoch_15.pth',
                        type=str,
                        help="load default 3D pretrained model.")

    return parser.parse_args()


def to_segments(data, num=32):
    """
	These code is taken from:
	https://github.com/rajanjitenpatel/C3D_feature_extraction/blob/b5894fa06d43aa62b3b64e85b07feb0853e7011a/extract_C3D_feature.py#L805
	:param data: list of features of a certain video
	:return: list of 32 segments
	"""
    data = np.array(data)
    Segments_Features = []
    thirty2_shots = np.round(np.linspace(0, len(data) - 1, num=num + 1)).astype(int)
    for ss, ee in zip(thirty2_shots[:-1], thirty2_shots[1:]):
        if ss == ee:
            temp_vect = data[min(ss, data.shape[0] - 1), :]
        else:
            temp_vect = data[ss:ee, :].mean(axis=0)

        temp_vect = temp_vect / np.linalg.norm(temp_vect)
        if np.linalg.norm == 0:
            logging.error("Feature norm is 0")
            exit()
        if len(temp_vect) != 0:
            Segments_Features.append(temp_vect.tolist())

    return Segments_Features


class FeaturesWriter:
    def __init__(self, num_videos, chunk_size=16):
        self.path = None
        self.dir = None
        self.data = None
        self.chunk_size = chunk_size
        self.num_videos = num_videos
        self.dump_count = 0

    def _init_video(self, video_name, dir):
        self.path = path.join(dir, f"{video_name}.txt")
        self.dir = dir
        self.data = dict()

    def has_video(self):
        return self.data is not None

    def dump(self):
        logging.info(f'{self.dump_count} / {self.num_videos}:	Dumping {self.path}')
        self.dump_count += 1
        if not path.exists(self.dir):
            os.mkdir(self.dir)
        # 32 segments
        features = to_segments([self.data[key] for key in sorted(self.data)])
        # 16 clip length
        # features = [self.data[key] for key in sorted(self.data)]
        with open(self.path, 'w') as fp:
            for d in features:
                d = [str(x) for x in d]
                fp.write(' '.join(d) + '\n')

    def _is_new_video(self, video_name, dir):
        new_path = path.join(dir, f"{video_name}.txt")
        if self.path != new_path and self.path is not None:
            return True

        return False

    def store(self, feature, idx):
        self.data[idx] = list(feature)

    def write(self, feature, video_name, idx, dir):
        if not self.has_video():
            self._init_video(video_name, dir)

        if self._is_new_video(video_name, dir):
            self.dump()
            self._init_video(video_name, dir)

        self.store(feature, idx)


def get_features_loader(dataset_path, clip_length, frame_interval, batch_size, num_workers, save_dir, use_splits):
    mean = [0.400, 0.388, 0.372]  # VAD 3 mean and std in RGB
    std = [0.247, 0.245, 0.243]
    size = 224
    resize = size, size
    crop = size

    res = transforms.Compose([
        transforms_video.ToTensorVideo(),
        transforms_video.ResizeVideo(resize),
        transforms_video.CenterCropVideo(crop),
        transforms_video.NormalizeVideo(mean=mean, std=std)
    ])

    if os.path.exists(save_dir):
        proc_v = []
        for root, dirs, files in os.walk(save_dir):
            for file in files:
                proc_v.append(file)
        proc_v = [v.split(".")[0] for v in proc_v]
        if len(proc_v) > 0:
            logging.info(
                f"[Data] Already {len(proc_v)} files have been processed"
            )

    data_loader = VideoIter(
        dataset_path=dataset_path,
        proc_video=proc_v,
        clip_length=clip_length,
        frame_stride=frame_interval,
        video_transform=res,
        use_splits=use_splits,
        return_label=False,
    )

    data_iter = torch.utils.data.DataLoader(
        data_loader,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return data_loader, data_iter


def load_VST(checkpoint, device):
    config = '../Video-Swin-Transformer/configs/recognition/swin/swin_base_patch244_window877_kinetics400_22k_VAD.py'
    cfg = Config.fromfile(config)
    model = build_model(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, checkpoint, map_location='cpu')

    return model.to(device)


def main():
    args = get_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    #os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.cuda.set_device(args.gpu)
    device = get_torch_device()

    register_logger(log_file=args.log_file)

    cudnn.benchmark = True

    if not path.exists(args.save_dir):
        mkdir(args.save_dir)

    data_loader, data_iter = get_features_loader(args.dataset_path,
                                                 args.clip_length,
                                                 args.frame_interval,
                                                 args.batch_size,
                                                 args.num_workers,
                                                 args.save_dir,
                                                 args.use_splits, )

    model = load_VST(args.pretrained_3d, device)
    # total_params = sum(p.numel() for p in model.parameters())  # params

    features_writer = FeaturesWriter(num_videos=data_loader.video_count)
    loop_i = 0
    # flops = 0
    # time_list = []
    with torch.no_grad():
        for data, clip_idxs, dirs, vid_names in data_iter: # 1 batch
            # if flops == 0:
            #     flops = profile_macs(model.backbone, data.to(device))
            # start_time = time.time()
            outputs = model.extract_feat(data.to(device))
            outputs = outputs.mean(dim=[2, 3, 4])
            outputs = outputs.detach().cpu().numpy()
            # end_time = time.time()
            # inference_time = end_time - start_time
            # time_list.append(inference_time)


            for i, (dir, vid_name, clip_idx) in enumerate(zip(dirs, vid_names, clip_idxs)):
                if loop_i == 0:
                    logging.info(
                        f"Video {features_writer.dump_count} / {features_writer.num_videos} : Writing clip {clip_idx} of video {vid_name}")

                loop_i += 1
                loop_i %= args.log_every

                dir = path.join(args.save_dir, dir)
                features_writer.write(feature=outputs[i],
                                      video_name=vid_name,
                                      idx=clip_idx,
                                      dir=dir, )

    features_writer.dump()
    # ave_time = np.mean(time_list)
    # with open('/media/DataDrive/yiling/Test/inference/stats.txt', 'a') as f:
    #     f.write(f'Model: {args.model_type}\n')
    #     f.write(f'FLOPs: {flops / 1e9} G x{len(time_list)}\n')
    #     f.write(f'Time: {ave_time * 1e6} us x{len(time_list)}\n')
    #     f.write(f'Total parameters: {total_params / 1e6} M\n')
    #     f.write("\n")


if __name__ == "__main__":
    main()
