# MTFL demo

## Folder Structure
```flow
demo/  
│  
├── detection/       # MTFL detection
│   └── ...
├── recognition/     # MTFL recognition
│   └── ... 
├── utils/  
│   ├── swin_config/ # VST config for loading feature extractor
│   │   └── ...  
│   ├── feature_extractor.py   
│   ├── ...   
│   └── video_preprocessing/ # scripts for annotation and unifying video format
│       └── ...
├── test_videos/     # put your test video here
├── Annotation/      # put your annotation here
├── features/        # feature path
│   ├── L8  
│   ├── L32   
│   └── L64  
├── results/       
│   ├── AUC          # detection AUC
│   ├── scores       # detection scores
│   └── rec_results  # recognition labels   
└── README.md 
```
## Video Preprocessing

If you need to perform video annotation or standardize video formats, 
run 'utils/video_preprocessing/video_annotator.py' or 'utils/video_preprocessing/video_format_unifier.py'.

### Annotation
Please note that running 'video_annotator.py' on the server may encounter issues related 
to 'qt.qpa.plugin', while running it locally on Windows would not have any issues. 
Therefore, you can run it in your local environment and then upload the generated annotation files to the server for subsequent tasks.
Make sure you have OpenCV installed in your local environment to run it.
```
pip install opencv-python
```

To run video_annotator.py, you need to specify the following parameters in the code according to your needs:
```
--root_dir: Denotes the root directory where video files are stored. 
--video_subdir: Specifies the subdirectory containing the videos to annotate.
--annotation_file: Represents the directory where the generated annotation file should be saved.
--anomaly_type: Indicates the type of the videos. 
```
Feel free to use anomaly types other than those in the VAD dataset for the testing data annotations, 
as they will be processed as 'anomaly' in both detection and recognition, without affecting the inference.


When the video playback window starts, the corresponding hotkeys are as follows: 
* space to pause
* 'a' to reverse 10 frames
* 'd' to skip 10 frames
* ',' to reverse 1 frame
* '.' to skip 1 frame
* 'm' to mark the current frame as the start or end of an anomalous event
* 'z' to cancel the last marking
* 's' to skip this video
* 'q' to exit.

Each line of the generated annotation includes:
* the relative path of the video with respect to the root directory
* the video label
* the number of frames in the video
* pairs of start and end frames for anomalous events.

Using relative paths here is to explicitly define the relative path of a video with
respect to the 'test_videos' folder, and to maintain this relative structure for the 
corresponding features and results storage.

### Unifying Format
To run video_format_unifier.py, you need to specify the following parameters in the code according to your needs:
```
--video_dir: Path to the input directory containing the video files.
--out_dir: Path to the output directory where processed videos will be saved.
```
The videos in [video_dir] will be modified to a resolution of 320x240 and a frame rate of 30fps, 
and they will be saved with the same names in the [out_dir]. If you want to try other format, feel free to change
<target_res> and <target_fps> in the code.


## Feature Extraction
Both recognition and detection models require multi-timescale features using tubelets 8, 32, and 64 frames.
To extract features, you need to upload the videos to the 'test_videos' directory and then run the following command:
```
python utils/feature_extractor.py --clip_length [8/32/64]
```
In the default settings, test videos should be stored in the 'test_videos' directory, and the extracted features will be 
organized within the 'features' folder following the same directory structure as 'test_videos'. 
For example, the feature of video 'test_videos/A/B.mp4' extracted with a frame length 8 is saved as 'features/L8/A/B.txt'.

You can modify the parameters inside the <VST Feature Extractor Parser> as needed, and 
the parameters you may want to experiment with are listed as below:
* You can change the input video path and the save path of features.
```
python utils/feature_extractor.py --clip_length [8/32/64] --dataset_path [your data path] --save_dir [your feature path] 
```

* You can change the used pretrained feature extractor by specifying the model path. 
```
python utils/feature_extractor.py --clip_length [8/32/64] --pretrained_3d [model path]
```
Note: if you use VST pretrained on Kinetics400, you need to change <num_classes> to 400 in line 21 of 
'utils/swin_config/_base/models/swin/swin_tiny.py' to adapt the model size. For VST pretrained on VAD, the <num_classes>
is 18.


The paths of two feature extractors are listed in the below table, and the default extractor is the VST model pretrained on VAD. 

| Model                         | Path |
|-------------------------------|------|
| VST pretrained on VAD         |/media/DataDrive/yiling/models/VST_finetune/hflip_speed_120_2d/best_top1_acc_epoch_15.pth  |
| VST pretrained on Kinetics400 |/media/DataDrive/yiling/models/pretrained_feature_extraction_models/swin_base_patch244_window877_kinetics400_22k.pth|


## Detection
### Inference
To test a detection checkpoint model on your test videos, run:
```
python detection/test.py --test_anno [your_anno_file.txt] --detection_model [checkpoint path]
```

There are two checkpoints. Make sure that the features for inference corresponds to the checkpoint being evaluated. 
The paths for the MTFL detection checkpoints corresponding to the two feature extractors are 
as shown in the table below.

| Checkpoint                                                                               | UCF AUC(%) | VAD AUC(%) | Path                                                                                                             |
|------------------------------------------------------------------------------------------|------------|------------|------------------------------------------------------------------------------------------------------------------|
| MTFL detection model trained using features extracted with VST pretrained on VAD         | 89.78      | 88.42      |/media/DataDrive/yiling/Test/models/MTFL/MTFL-vst-VAD.pkl                                                        |
| MTFL detection model trained using features extracted with VST pretrained on Kinetics400 | 87.61      | 87.06      | /media/DataDrive/yiling/Test/models/MTFL/MTFL-vst-kinetics400.pkl|



To obtain AUC and score results, the following prerequisites are necessary:

* Test videos should be stored in the 'test_videos' directory.
* The corresponding annotation file need to be placed in the 'annotation' folder. Annotation format can be found under Video Preprocessing->Annotation.
```
[Path of the video relative to 'test_videos'] [video label] [total frames] [start_frame1] [end_frame1] [start_frame2]...
```
* Multi-temporal scale features of the videos should be stored in the 'features' directory. See Feature Extraction.



The detection AUC and the scores for each video will be generated within the 'results' folder. 
The directory structure of the generated results, in relation to both 'results/AUC' and 'results/scores', mirrors the 
structure of the corresponding test videos in the 'test_videos' directory. For example, 
the score of video 'test_videos/A/B.mp4' is saved as 'results/scores/A/B.png' 

Additionally, if you want to change paths to input and output data or any running configs, 
feel free to change the args in 'detection/option.py'.

### Train
To train a detection model, run:
```
python detection/train.py --train_anno [your_train_anno_file.txt] --test_anno [your_test_anno_file.txt] 
--lf_dir [path to long frame length features] --mf_dir [path to medium frame length features] --sf_dir 
[path to short frame length features] --save_models [path for saving checkpoints] --output_dir [path for saving checkpoint AUC]
```

Other training parameters can be found in 'detection/option.py'

## Recognnition
### Inference
To test a recognition checkpoint model on your test videos, run
```
python recognition/test.py --test_anno [your_anno_file.txt] --recognition_model [checkpoint path]
```

The prerequisites are same as Detection, and the modifiable parameters are in 'recognition/option.py'. 
The recognition results of all input will be saved as 'results/rec_results/output_pred.txt'. 

Because the evaluation results for recognition are obtained through 4-fold cross-validation
, there are seven recognition checkpoints by saving the checkpoints that performed the best on 
UCF and VAD separately during training on different splits, as shown in the table below. You can choose one from them.

| Checkpoint                                                                                       | UCF Acc(%) | VAD Acc(%) | Path                                                                    |
|--------------------------------------------------------------------------------------------------|------------|------------|-------------------------------------------------------------------------|
| MTFL recognition model trained on VAD split1 with the best performance on UCF-Crime              | 39.88      | -          | /media/DataDrive/yiling/Test/models/MTFL_recog/split_1_best_UCF.pkl     |
| MTFL recognition model trained on VAD split1 with the best performance on VAD                    | -          | 45.87      | /media/DataDrive/yiling/Test/models/MTFL_recog/split_1_best_VAD.pkl     |
| MTFL recognition model trained on VAD split2 with the best performance on UCF-Crime              | 47.02      | -          | /media/DataDrive/yiling/Test/models/MTFL_recog/split_2_best_UCF.pkl     |
| MTFL recognition model trained on VAD split2 with the best performance on VAD                    | -          | 49.31      | /media/DataDrive/yiling/Test/models/MTFL_recog/split_2_best_VAD.pkl     |
| MTFL recognition model trained on VAD split3 with the best performance on UCF-Crime              | 49.40      | -          | /media/DataDrive/yiling/Test/models/MTFL_recog/split_3_best_UCF.pkl     |
| MTFL recognition model trained on VAD split3 with the best performance on VAD                    | -          | 53.88      | /media/DataDrive/yiling/Test/models/MTFL_recog/split_3_best_VAD.pkl     |
| MTFL recognition model trained on VAD split4 with the best performance on both VAD and UCF-Crime | 45.83      | 52.29      | /media/DataDrive/yiling/Test/models/MTFL_recog/split_4_best_UCF_VAD.pkl |

All recognition models use the VST pretrained on VAD for feature extraction. 
The training of recognition models requires separate training on 4 splits and generates multiple checkpoints. 
Therefore, there is no provided model corresponding to VST pretrained on Kinetics400, 
taking into account both the necessity and workload. If needed, you can use the recognition/train.py 
script for training.

### Train
To train a recognition model, run:
```
python recognition/train.py --train_anno [your_train_anno_file.txt] --test_anno [your_test_anno_file.txt]
--lf_dir [path to long frame length features] --mf_dir [path to medium frame length features] --sf_dir 
[path to short frame length features] --save_models [path for saving checkpoints] --output_dir [path for saving checkpoint AUC]
```

Note: there are four pairs of training annotation and testing annotation files corresponding to four splits for each dataset. 
Make sure the correspondence between the training and testing files; otherwise, there are data leakage issues. 
The annotation files of all splits in UCF-Crime and VAD are in '/media/DataDrive/yiling/annotation/recognition/splits'.
Other training parameters can be found in 'recognition/option.py'

