import os
import random

# 设置要操作的文件夹路径和要删除的文件数量
raw = "H:\\Projects\\VAD\\Datasets\\baseline\\AnnotationVideo"
anomaly_type = "EMVvsVRU"
participant = "TruckVsPedestrian"
num_files_to_delete = 30

if __name__ == "__main__":
    # 获取文件夹中的所有文件
    all_files = os.listdir(f"{raw}\\{anomaly_type}\\{participant}")

    # 随机选择要删除的文件
    files_to_delete = random.sample(all_files, num_files_to_delete)

    # 删除选定的文件
    for file_name in files_to_delete:
        file_path = os.path.join(f"{raw}\\{anomaly_type}\\{participant}", file_name)
        os.remove(file_path)
