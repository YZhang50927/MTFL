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
│   ├── Video_annotator.py  
│   └── ...  
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
├── env.txt          # environment  
└── README.md 
```

## Detection
### Inference
To test the checkpoint model on your test videos, run:
```
python detection/test.py --test_anno [your_anno_file.txt] --detection_model [checkpoint path]
```

To obtain AUC and score results, the following prerequisites are necessary:

* Test videos should be stored in the 'test_videos' directory.
* The corresponding annotation file need to be placed in the 'annotation' folder.
```
[Path of the video relative to 'test_videos'] [video label] [total frames] [start_frame1] [end_frame1] [start_frame2]...
```
* Multi-temporal scale features of the videos should be stored in the 'features' directory. 
These features can be extracted using the 'utils/feature_extractor' and will be organized within the 'features' folder
following the default directory structure.
```
python utils/feature_extractor.py --clip_length [8/32/64]
```

The detection AUC and the scores for each video will be generated within the 'results' folder. 
The directory structure of the generated results, in relation to both 'results/AUC' and 'results/scores', mirrors the 
structure of the corresponding test videos in the 'test_videos' directory. For example, 
the score of video 'test_videos/A/B.mp4' is saved as 'results/scores/A/B.png' 

Additionally, if you want to change paths to input and output data or any running configs, 
feel free to change the args in 'detection/option.py' and 'utils/feature_extractor'

### Train
To train a detection model, run:
```
python detection/train.py --train_anno [your_train_anno_file.txt] --test_anno [your_test_anno_file.txt] 
--lf_dir [path to long frame length features] --mf_dir [path to medium frame length features] --sf_dir 
[path to short frame length features] --save_models [path for saving checkpoints] --output_dir [path for saving checkpoint AUC]
```

Other training parameters can be found in 'detection/option.py'