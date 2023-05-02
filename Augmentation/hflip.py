import os
import subprocess

# 源目录和目标目录
src_dir = '/media/DataDrive/yiling/VAD3'
dst_dir = '/media/DataDrive/yiling/VAD3_hflip'

cnt = 0
# 遍历源目录中的所有文件和子目录，并获取视频文件列表
for root, dirs, files in os.walk(src_dir):
    # 构造目标目录中的子目录路径
    rel_path = os.path.relpath(root, src_dir)
    dst_subdir = os.path.join(dst_dir, rel_path)
    os.makedirs(dst_subdir, exist_ok=True)

    # 遍历当前目录下的视频文件，并进行翻转
    for file in files:
        if file.endswith('.mp4'):
            # 构造输入和输出文件路径
            input_path = os.path.join(root, file)
            cnt += 1

            # 构造输出文件名
            name, ext = os.path.splitext(file)
            output_name = name + '_hflip' + ext
            output_path = os.path.join(dst_subdir, output_name)

            if os.path.exists(output_path):
                print(f'Skipping {file} (already processed)')
                continue  # 跳过已经处理过的视频
            print(f'Processing {cnt} videos: {output_name}')  # 打印处理进度

            bitrate = subprocess.check_output(
                ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=bit_rate', '-of',
                 'default=noprint_wrappers=1:nokey=1', input_path])
            bitrate = int(bitrate)
            # 构造FFmpeg命令
            command = ['ffmpeg', '-threads', '8', '-i', input_path, '-vf', 'hflip', '-c:a', 'copy', '-b:v', f'{bitrate}', output_path]

            # 启动FFmpeg进程
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       universal_newlines=True)

            # 获取标准输出和标准错误流，并输出到屏幕上
            stdout, stderr = process.communicate()
            #print(stdout)
            print(stderr)

            # 等待进程结束
            process.wait()

            print(f'Processed')  # 打印处理进度
