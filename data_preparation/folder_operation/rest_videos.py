import os
import shutil

# 输入两个文件夹路径
video_folder = "/media/mount_loc/yiling/VAD3/Normal"
txt_folder = "/media/mount_loc/yiling/features/I3D/Normal"
output_folder = "/media/mount_loc/yiling/rest"

if __name__ == "__main__":
    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取txt文件夹中的所有文件名
    txt_files = os.listdir(txt_folder)
    txt_files = [os.path.splitext(f)[0] for f in txt_files]
    # 遍历视频文件夹
    for video_file in os.listdir(video_folder):
        # 如果文件名不在txt文件中，则将其复制到输出文件夹
        if os.path.splitext(video_file)[0] not in txt_files:
            shutil.copy(os.path.join(video_folder, video_file), output_folder)
