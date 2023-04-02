import os
import shutil

# 定义源文件夹和目标文件夹路径
file_list_path = '/media/mount_loc/yiling/annotation/VAD_test_annotation.txt'
src_folder_1 = "/media/mount_loc/yiling/test"
src_folder_2 = "/media/mount_loc/ivo/features/C3D_features"
dst_folder = "/media/mount_loc/yiling/features/C3D_32Train_32Test"

# 读取 txt 文件，获取需要复制的文件的相对路径信息
with open(file_list_path) as f:
    file_paths = [line.strip().split()[0] for line in f]

# 遍历两个源文件夹
for src_folder in [src_folder_1, src_folder_2]:
    for root, dirs, files in os.walk(src_folder):
        for file_name in files:
            exist = False
            name, ext = os.path.splitext(file_name)
            for file_path in file_paths:
                if name in file_path:
                    path, ext = os.path.splitext(file_path)
                    relative_path = path + '.txt'
                    exist = True
                    break
            if exist:
                # 构造目标文件夹中的路径，并创建目标文件夹
                dst_file_path = os.path.join(dst_folder, relative_path)
                dst_dir_path = os.path.dirname(dst_file_path)
                os.makedirs(dst_dir_path, exist_ok=True)

                # 复制文件
                src_file_path = os.path.join(root, file_name)
                shutil.copy2(src_file_path, dst_file_path)
