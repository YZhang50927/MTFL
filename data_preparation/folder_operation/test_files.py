import os
import shutil

# 源文件夹路径
source_dir = "/media/mount_loc/yiling/VAD3"

# 目标文件夹路径
target_dir = '/media/mount_loc/yiling/VAD3_test'

# 视频文件列表的txt文件路径
file_list_path = '/media/mount_loc/yiling/annotation/VAD_test_annotation.txt'

# 打开txt文件，读取视频文件列表
with open(file_list_path, 'r') as f:
    file_list = f.read().splitlines()

# 遍历视频文件列表，逐个复制到目标文件夹
for file_info in file_list:
    # 按照空格分割文件信息，第一列是文件名
    file_name = file_info.split()[0]

    # 拼接源文件路径和目标文件路径
    source_file_path = os.path.join(source_dir, file_name)
    # 拼接目标文件路径
    target_file_path = os.path.join(target_dir, file_name)

    # 构建目标文件夹路径
    target_subdir_path = os.path.dirname(target_file_path)

    # 如果目标文件夹不存在，则创建它
    if not os.path.exists(target_subdir_path):
        os.makedirs(target_subdir_path)

    # 复制文件
    shutil.copy2(source_file_path, target_file_path)
