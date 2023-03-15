# Unify the format
# H:\\Projects\\VAD\\Datasets\\baseline\\Accidents\\Real dataset\\*
# H:\Projects\VAD\Datasets\baseline\UCF\Arrest
# H:\Projects\VAD\Datasets\baseline\UCF\Assault
# H:\Projects\VAD\Datasets\baseline\UCF\Robbery
import cv2
import os
import sys
import shutil

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
dir = "H:\\Projects\\VAD\\Datasets\\baseline\\UCF\\Robbery"
outdir = "H:\\Projects\\VAD\\Datasets\\baseline\\UniformFormat\\Robbery"
target_res = (320, 240)
target_fps = 30.0
video_to_resize = []
video_fail = []
classname = 'Robbery'
if __name__ == "__main__":
    
    counter = 0
    for file in os.listdir(dir):
        counter += 1
        if not file.endswith('.mp4'):
            continue
        if not os.path.exists(f"{outdir}"):
            os.makedirs(f"{outdir}")

        print(f"Processing video {counter}")

        cap = cv2.VideoCapture(f"{dir}\\{file}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        #newname = classname + str(counter).zfill(3) + os.path.splitext(file)[-1]

        if fps==target_fps and width==target_res[0] and height==target_res[1]:
            print(f"Skip {dir}\\{file}")
            shutil.copyfile(f"{dir}\\{file}", f"{outdir}\\{file}")
            cap.release()
            continue

        video_to_resize.append(f"{dir}\\{file}")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(f"{outdir}\\{file}", fourcc, target_fps, target_res)

        if not cap.isOpened():
            print(f"Error opening video: {dir}\\{file}")
            video_fail.append(f"{dir}\\{file}")
            sys.exit(1)

        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                b = cv2.resize(frame, target_res, interpolation=cv2.INTER_AREA)
                out.write(b)
            else:
                break
        cap.release()
        out.release()
        #os.rename(f"{outdir}\\{file}", f"{outdir}\\{newname}")

