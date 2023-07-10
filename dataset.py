import torch.utils.data as data
import os
import torch
from feature_extraction.feature_reader import read_features
torch.set_default_tensor_type('torch.FloatTensor')


class Dataset(data.Dataset):
    def __init__(self, args, is_normal=True, transform=None, test_mode=False, test_dataset='UCF'):
        self.modality = args.modality
        self.is_normal = is_normal
        self.dataset = test_dataset
        self.type = args.feature_type
        self.transform = transform
        self.test_mode = test_mode
        self._parse_list()
        self.num_frame = 0
        self.labels = None

    def _parse_list(self):
        # feature type
        if self.type == 'i3d':
            features_path = '/media/DataDrive/yiling/features/I3D_32Train_32Test'
        elif self.type == 'c3d':
            features_path = '/media/DataDrive/yiling/features/C3D_32Train_32Test'
        elif self.type == 'vst':
            features_path = '/media/DataDrive/yiling/features/VST_VAD_hflip_speed120_80_2D'
        # test dataset
        if self.dataset == 'VAD3':
            test_annotation_path = '/media/DataDrive/yiling/annotation/VAD_test_annotation_with_frames.txt'
        elif self.dataset == 'UCF':
            test_annotation_path = '/media/DataDrive/yiling/annotation/UCF_test_annotation_with_frames.txt'
        # test or train
        if self.test_mode:
            annotation_path = test_annotation_path
        else:
            annotation_path = '/media/DataDrive/yiling/annotation/VAD_train_annotation.txt'
        self.list = self._get_features_list(features_path, annotation_path)

    def __getitem__(self, index):
        label = self.get_label()  # get video level label 0/1

        if self.test_mode:
            feature_subpath, start_end_couples, num_frames = self.list[index]
            features = read_features(feature_subpath, model=self.type)
            if self.transform is not None:
                features = self.transform(features)
            return torch.unsqueeze(features, 0), start_end_couples, num_frames  #[N,B,T,F]
        else:
            features = read_features(self.list[index].strip('\n'), model=self.type)
            if self.transform is not None:
                features = self.transform(features)
            return torch.unsqueeze(features, 0), label

    def get_label(self):

        if self.is_normal:
            label = torch.tensor(0.0)
        else:
            label = torch.tensor(1.0)

        return label

    def __len__(self):
        return len(self.list)

    def get_num_frames(self):
        return self.num_frame

    def _get_features_list(self, features_path, annotation_path):
        assert os.path.exists(features_path)
        features_list = []
        with open(annotation_path) as f:
            lines = f.read().splitlines(keepends=False)
            for line in lines:
                items = line.split()
                file = items[0].split(".")[0]
                file = file.replace("/", os.sep)
                feature_path = os.path.join(features_path, file + '.txt')
                if self.test_mode:
                    start_end_couples = [int(x) for x in items[3:]]
                    num_frames = int(items[2])
                    features_list.append((feature_path, start_end_couples, num_frames))
                else:
                    if "Normal" in feature_path and self.is_normal:
                        features_list.append(feature_path)
                    else:
                        if "Normal" not in feature_path and not self.is_normal:
                            features_list.append(feature_path)

        return features_list

