import os
import shutil

# 源文件夹和目标文件夹路径
src_folder = "/media/mount_loc/yiling/features/I3D_32Train_32Test"
dst_folder = "/media/mount_loc/yiling/features/I3D_32Train_16Test"

# 遍历源文件夹中的所有文件和目录
for root, dirs, files in os.walk(src_folder):
    for file_name in files:
        # 构造源文件路径和目标文件路径
        src_file_path = os.path.join(root, file_name)
        dst_file_path = src_file_path.replace(src_folder, dst_folder, 1)

        # 如果目标文件夹中已存在该文件，则跳过
        if os.path.exists(dst_file_path):
            continue

        # 创建目标文件夹
        dst_dir_path = os.path.dirname(dst_file_path)
        os.makedirs(dst_dir_path, exist_ok=True)

        # 复制文件
        shutil.copy2(src_file_path, dst_file_path)

