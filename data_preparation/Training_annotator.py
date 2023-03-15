import os

# 指定目标文件夹和txt文件路径
folder_path = "/media/mount_loc/yiling/VAD3"
txt_path = "/home/yiling/workspace/data_preparation/annotation/Test_annotation.txt"
output_path = "/home/yiling/workspace/data_preparation/annotation/Train_annotation.txt"

# 从txt文件中读取文件名列表
with open(txt_path, "r") as f:
    txt_filenames = set([line.strip().split()[0] for line in f.readlines()])

# 遍历目标文件夹中的所有文件，包括子目录中的文件
extra_filenames = []
for root, dirs, files in os.walk(folder_path):
    for filename in files:
        filepath = os.path.join(root, filename)
        if os.path.isfile(filepath) and filename not in txt_filenames:
            foldername = os.path.basename(os.path.dirname(filepath))
            extra_filenames.append(filename + " " + foldername)

# 将结果写入到txt文件中
with open(output_path, "w") as f:
    for filename in extra_filenames:
        f.write(filename + "\n")
