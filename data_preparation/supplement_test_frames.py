import cv2
import os

# 视频文件列表的txt文件路径
file_list_path = '/media/mount_loc/yiling/annotation/UCF_test_annotation.txt'
root = '/media/mount_loc/yiling/VAD3'
output = '/media/mount_loc/yiling/annotation/UCF_test_annotation_with_frames.txt'

# 打开txt文件，读取视频文件列表
with open(file_list_path, 'r') as f:
    file_list = f.read().splitlines()

# 遍历视频文件列表，逐个获取帧数并记录到txt文件中
for i, file_info in enumerate(file_list):
    # 按照空格分割文件信息，第一列是文件名
    file_name = file_info.split()[0]
    file = os.path.join(root, file_name)

    # 打开视频文件
    cap = cv2.VideoCapture(file)

    # 获取视频的帧数
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 将帧数插入到列表中的第3列
    file_info_list = file_info.split()
    file_info_list.insert(2, str(frame_count))
    file_list[i] = ' '.join(file_info_list)

    # 关闭视频文件
    cap.release()

# 将带有帧数信息的视频文件列表写回到txt文件中
with open(output, 'w') as f:
    f.write('\n'.join(file_list))
