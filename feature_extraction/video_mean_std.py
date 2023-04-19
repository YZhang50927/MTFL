import os
import cv2
import numpy as np
import torch
from tqdm import tqdm

def calculate_mean_and_std(video_dir):

    # 检查CUDA是否可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 初始化均值和方差
    mean = torch.zeros(3).to(device)
    std = torch.zeros(3).to(device)
    num_frames = 0
    video_cnt = 0
    # 遍历整个视频数据集
    for root, dirs, files in os.walk(video_dir):
        for filename in files:
            if not filename.endswith('.mp4'):
                continue

            video_cnt += 1
            print('\nProcessing video {}:\n'.format(video_cnt))
            # 打开视频文件
            cap = cv2.VideoCapture(os.path.join(root, filename))

            # 遍历视频中的每一帧
            with tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))) as pbar:
                while cap.isOpened():
                    ret, frame = cap.read() #bgr
                    if not ret:
                        break

                    # 将帧转换为张量并移动到GPU设备上 H,W,C->C,H,W BGR
                    frame = torch.from_numpy(frame).permute(2, 0, 1).float().div(255).to(device)

                    # 计算每个通道的均值和方差
                    mean += torch.mean(frame, dim=(1, 2))
                    std += torch.std(frame, dim=(1, 2))
                    num_frames += 1

                    pbar.update(1)

                cap.release()

    # 计算均值和方差
    mean /= num_frames # bgr
    std /= num_frames

    # 将结果输出到txt文件
    with open('/media/DataDrive/yiling/VAD3/mean_std.txt', 'w') as f:
        f.write('Mean: {}\n'.format(mean.tolist()))
        f.write('Std: {}\n'.format(std.tolist()))

    return mean, std


if __name__=='__main__':
    calculate_mean_and_std('/media/DataDrive/yiling/VAD3')
