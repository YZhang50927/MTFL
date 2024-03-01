# MTFL: Multi-Timescale Feature Learning for Weakly-supervised Anomaly Detection in Surveillance Videos

This repo is the official Pytorch implementation of our paper:

> [**MTFL: Multi-Timescale Feature Learning for Weakly-supervised Anomaly Detection in Surveillance Videos**](PAPER LINK)
>
<!--Author list-->

## Introduction
![intro](figures/intro.png)

Detection of anomaly events is relevant for public safety and requires a combination of fine-grained motion information and long-time action recognition. Therefore, we propose a Multi-Timescale Feature Learning (MTFL) method to enhance the representation of anomaly features. We employ short, medium, and long temporal tubelets to extract spatio-temporal video 
features using the Video Swin Transformer. Experimental results demonstrate that 
MTFL outperforms state-of-the-art methods on the UCF-Crime dataset, achieving an 
anomaly detection performance 89.78% AUC. Moreover, it performs 95.32% AUC on the 
ShanghaiTech and 84.57% AP on the XD-Violence dataset, complementary to several
SotA results. Building upon MTFL, we also propose an anomaly recognition network 
that employs partial features for classification, achieving a leading accuracy on 
UCF-Crime, outperforming the existing recognition literature. Furthermore, 
we introduce an extended dataset for UCF-Crime, 
namely Video Anomaly Detection Dataset~(VADD), 
involving 2,591 videos in 18 classes with extensive coverage of realistic anomalies. 

## Models and Dataset
* [VADD]()
* [MTFL checkpoints for anomaly detection]()
* [MTFL checkpoints for anomaly recognition]()

## Environment setup
```
pip install -r requirements.txt
```
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


Two feature extractors used in our model are provided as below:
* [Video Swin Transformer pretrained on VAD](link)
* [VST pretrained on Kinetics-400](link)


## Anomaly Detection
### Inference
To test a detection checkpoint model on your test videos, run:
```
python detection/test.py --test_anno [your_anno_file.txt] --detection_model [checkpoint path]
```

All anomaly detection MTFL checkpoints are listed in the below table. Make sure that the features for inference corresponds to the checkpoint being evaluated.

| Detection Checkpoint        | feature       | UCF   | ShanghaiTech | XD-Violence | VADD |
|-----------------------------|---------------|-------|--------------|-------------|---|
| [MTFL_AD_VST_Kinetics400]() | VST-RGB       | 87.61 | 95.32        | 84.57       | - |
| [MTFL_AD_VST_VADD]()        | VST<sub>Aug</sub>_RGB | 89.79 | 95.70 | 79.40 | 88.42 |



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

## Anomaly Recognnition
### Inference
To test a recognition checkpoint model on your test videos, run
```
python recognition/test.py --test_anno [your_anno_file.txt] --recognition_model [checkpoint path]
```

The prerequisites are same as Detection, and the modifiable parameters are in 'recognition/option.py'. 
The recognition results of all input will be saved as 'results/rec_results/output_pred.txt'. 

Because the evaluation results for recognition are obtained through 4-fold cross-validation
, there are seven recognition checkpoints by saving the checkpoints that performed the best on 
UCF and VADD separately during training on different splits, as shown in the table below. You can choose one from them.

| Recognition Checkpoint                                                                             | UCF Acc(%) | VAD Acc(%) |
|----------------------------------------------------------------------------------------------------|------------|------------|
| [MTFL recognition model trained on VADD split1 with the best performance on UCF-Crime]()           | 39.88      | -          |
| [MTFL recognition model trained on VADD split1 with the best performance on VAD]()                 | -          | 45.87      |
| [MTFL recognition model trained on VADD split2 with the best performance on UCF-Crime]()              | 47.02      | -          |
| [MTFL recognition model trained on VADD split2 with the best performance on VAD]()                    | -          | 49.31      |
| [MTFL recognition model trained on VADD split3 with the best performance on UCF-Crime]()              | 49.40      | -          |
| [MTFL recognition model trained on VADD split3 with the best performance on VAD]()                    | -          | 53.88      |
| [MTFL recognition model trained on VADD split4 with the best performance on both VAD and UCF-Crime]()  | 45.83      | 52.29      |

All recognition models use the VST pretrained on VADD for feature extraction. 
The training of recognition models requires separate training on 4 splits and generates multiple checkpoints. 
Therefore, there is no provided recognition model corresponding to VST pretrained on Kinetics400, 
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
Other training parameters can be found in 'recognition/option.py'

<!--## Citation

If you find this repo useful for your research, please consider citing our paper:-->

