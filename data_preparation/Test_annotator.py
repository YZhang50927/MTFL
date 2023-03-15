import cv2
import sys
import os
import random
# Directory containing raw videos
raw = "H:\\Projects\\VAD\\Datasets\\baseline\\AnnotationVideo"
annotation_file = f"H:\\Projects\\VAD\\workspace\\data_preparation\\annotation\\doc\\Test_annotation.txt"
initial_index = 0

anomaly_type = "RoadAccidents_VRUvsVRU"
participant = "MotorbikeVsPedestrian"
picknum = 3

if __name__ == "__main__":
    pathDir = os.listdir(f"{raw}\\{anomaly_type}\\{participant}")
    file = random.sample(pathDir, picknum)
    # Loop over all files in directory
    for filename in file:
        if not filename.endswith(".mp4"):
            continue

        # Get index of video file
        index = int(filename.split('.')[0][-3:])

        # Only start at initial_index
        if index < initial_index:
            continue

        cap = cv2.VideoCapture(raw + f"\\{anomaly_type}\\{participant}\\{filename}")
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"{filename}: {num_frames}")

        if not cap.isOpened():
            print("Error opening video")
            sys.exit(1)

        mem = []
        actions = []

        counter = 0
        max_frame = 0
        marked_frame = 0

        playing = True
        marking = False

        while cap.isOpened():
            if playing:
                if counter == max_frame:
                    ret, frame = cap.read()
                    if ret:
                        if len(mem) == 200:
                            mem.pop(0)
                        mem.append(frame)
                    else:
                        break
                    counter += 1
                    max_frame += 1

                else:
                    frame = mem[counter - max_frame - 1]
                    counter += 1
            else:
                frame = mem[counter - max_frame - 1]
                cv2.putText(frame, f"{len(mem) + counter - max_frame}", (50, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1, color=(255, 255, 255))

            if marking:
                cv2.circle(frame, (int(frame.shape[1]/2), 50), 10, (0, 0, 255), -1)

            cv2.imshow("Replay", frame)
            key = cv2.waitKey(60)
            if key & 0xFF == ord(' '):  # Press space to pause
                playing = not playing
            elif key & 0xFF == ord('a') and not playing:  # a to reverse 10 frames
                if (max_frame - counter) < len(mem) - 11:
                    counter -= 10
            elif key & 0xFF == ord('d') and not playing:  # d to skip 10 frames
                if (max_frame - counter) > 10:
                    counter += 10
            elif key & 0xFF == ord(',') and not playing:  # , to reverse 1 frame
                if (max_frame - counter) < len(mem) - 1:
                    counter -= 1
            elif key & 0xFF == ord('.') and not playing:  # . to skip 1 frame
                if (max_frame - counter) > 0:
                    counter += 1
            elif key & 0xFF == ord('m'):  # m to mark this frame
                if marking:
                    actions.append((marked_frame, counter))
                else:
                    marked_frame = counter
                marking = not marking
                print(f'marking {counter}, waiting for the end frame: {marking}')
            elif key & 0xFF == ord('z'):  # z to undo last marking
                print(f'cancel last marking')
                if not marking:
                    actions.pop(-1)
                marking = not marking
            elif key & 0xFF == ord('s'):  # s to skip this video
                break
            elif key & 0xFF == ord('q'):  # q to quit
                sys.exit(1)

        num_action = len(actions)
        #line = f"{participant}{anomaly_type}/{filename} {num_frames} {participant}{anomaly_type}"
        line = f"{filename} {anomaly_type}"
        for i in range(num_action):
            line += f" {actions[i][0]} {actions[i][1]}"
        line += '\n'

        with open(annotation_file, "a") as f:
            f.write(line)

        cap.release()
        cv2.destroyWindow("Replay")

    cv2.destroyAllWindows()
