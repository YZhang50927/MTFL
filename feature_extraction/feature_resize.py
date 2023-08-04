import os
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# 读取特征的函数，根据你的特征文件格式进行相应的读取操作
def read_features_from_txt(file_path):
    # 在这里实现读取特征的代码
    # 返回特征数据，假设是一个列表
    with open(file_path, 'r') as f:
        lines = f.readlines()
        features = []
        for line in lines:
            # 在这里处理每行的特征数据，将其转换为你需要的格式
            # 假设每行特征数据是以空格分隔的数字字符串
            feature = [float(x) for x in line.strip().split()]
            features.append(feature)
    return features

# 平分特征为32份的函数
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
        if len(temp_vect) != 0:
            Segments_Features.append(temp_vect.tolist())

    return Segments_Features

# 输入文件夹和输出文件夹路径
input_folder = "/media/DataDrive/yiling/features/VST_Temporal_Variation/L64R8"
output_folder = "/media/DataDrive/yiling/features/VST_Temporal_Variation/L64R8_32clips"
clip_num = 32

# 处理单个文件的函数，包括读取特征、平分特征和写入文件的操作
def process_file(file_path):
    # 读取特征
    features = read_features_from_txt(file_path)

    # 将特征平分为32份
    segmented_features = to_segments(features, num=clip_num)

    # 根据当前文件夹结构创建对应的输出文件夹
    relative_path = os.path.relpath(os.path.dirname(file_path), input_folder)
    file_name = os.path.basename(file_path)
    os.makedirs(output_folder, exist_ok=True)
    output_subfolder = os.path.join(output_folder, relative_path)
    os.makedirs(output_subfolder, exist_ok=True)
    output_file = os.path.join(output_subfolder, file_name)

    with open(output_file, 'w') as fp:
        for d in segmented_features:
            d = [str(x) for x in d]
            fp.write(' '.join(d) + '\n')

if __name__=='__main__':
    with ThreadPoolExecutor() as executor:
        total_files = sum(len(files) for _, _, files in os.walk(input_folder))
        with tqdm(total=total_files, desc="Processing files") as pbar:
            # 递归遍历文件夹中的所有文件，并提交处理任务给线程池
            for root, _, files in os.walk(input_folder):
                for file_name in files:
                    if file_name.endswith(".txt"):
                        file_path = os.path.join(root, file_name)
                        # 提交处理任务给线程池
                        executor.submit(process_file, file_path)

                        # 更新进度条
                        pbar.update(1)

    # 完成所有文件的处理
    print("特征文件处理完成！")

