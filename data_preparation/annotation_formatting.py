import os

file_list = '/media/mount_loc/yiling/UCF_Crimes/Anomaly_Train.txt'
search_path = '/media/mount_loc/yiling/VAD3'
output = '/media/mount_loc/yiling/annotation/UCF_train_annotation.txt'


def find_file(filename, search_path):
    for root, dir, files in os.walk(search_path):
        if filename in files:
            file_path = os.path.join(root, filename)
            folder = os.path.basename(os.path.dirname(file_path))
            return os.path.join(folder, filename)

    return None

if __name__ == "__main__":
    with open(file_list, 'r') as f:
        lines = f.readlines()

    with open(output, 'w') as f:
        for i, line in enumerate(lines):
            file = line.split(' ')[0].strip()
            filename = os.path.basename(file)
            file_path = find_file(filename, search_path)
            if file_path is not None:
                new_line = line.replace(filename, file_path)
                f.write(new_line)
            else:
                print(f"文件 {filename} 不存在！")
                f.write(line)