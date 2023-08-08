import torch.utils.data as data
import os
import torch
import numpy as np
from utils.RTFM_utils import to_segments
torch.set_default_tensor_type('torch.FloatTensor')


def read_features(feature_path):
    with open(feature_path, 'r') as file:
        lines = file.readlines()
    features = []
    for line in lines:
        # 假设txt文件中每行是一维特征数据，以空格分隔
        feature = [float(value) for value in line.strip().split()]
        features.append(feature)
    #features = to_segments(features, self.seg_num) # B x T x D
    features = torch.tensor(features).float() # B x T x D
    return features


class Dataset(data.Dataset):
    def __init__(self, args, is_normal=True, transform=None, test_mode=False, test_dataset='UCF'):
        self.is_normal = is_normal
        self.test_dataset = test_dataset
        self.transform = transform
        self.test_mode = test_mode
        self.seg_num = args.seg_num

        self._parse_list(args.lf_dir, args.mf_dir, args.sf_dir)

    def _parse_list(self, lf_dir, mf_dir, sf_dir):
        # test dataset
        if self.test_dataset == 'VAD3':
            test_annotation_path = '/media/DataDrive/yiling/annotation/VAD_test_annotation_with_frames.txt'
        elif self.test_dataset == 'UCF':
            test_annotation_path = '/media/DataDrive/yiling/annotation/UCF_test_annotation_with_frames.txt'
        # test or train
        if self.test_mode:
            annotation_path = test_annotation_path
        else:
            annotation_path = '/media/DataDrive/yiling/annotation/VAD_train_annotation.txt'
        self.list = self._get_features_list(lf_dir, mf_dir, sf_dir, annotation_path)

    def __getitem__(self, index):
        label = self.get_label()
        if self.test_mode:
            lf_path, mf_path, sf_path, num_frames, start_end_couples = self.list[index]
            l_features = read_features(lf_path)
            m_features = read_features(mf_path)
            s_features = read_features(sf_path)
            # features = [l_features, m_features, s_features] # 3, T, D
            # B X 3 X T X D
            # features = torch.cat((l_features.unsqueeze(0), m_features.unsqueeze(0), s_features.unsqueeze(0)), dim=0)
            return l_features, m_features, s_features, label, start_end_couples, num_frames
        else:
            lf_path, mf_path, sf_path = self.list[index]
            l_features = read_features(lf_path)
            m_features = read_features(mf_path)
            s_features = read_features(sf_path)
            # features = [l_features, m_features, s_features]
            # features = torch.cat((l_features.unsqueeze(0), m_features.unsqueeze(0), s_features.unsqueeze(0)), dim=0)
            return l_features, m_features, s_features, label

    def get_label(self):
        if self.is_normal:
            label = torch.tensor(0.0)
        else:
            label = torch.tensor(1.0)

        return label

    def __len__(self):
        return len(self.list)

    def _get_features_list(self, lf_dir, mf_dir, sf_dir, annotation_path):
        assert os.path.exists(lf_dir)
        assert os.path.exists(mf_dir)
        assert os.path.exists(sf_dir)
        features_list = []
        with open(annotation_path) as f:
            lines = f.read().splitlines(keepends=False)
            for line in lines:
                items = line.split()
                file = items[0].split(".")[0]
                file = file.replace("/", os.sep)
                lf_path = os.path.join(lf_dir, file + '.txt')
                mf_path = os.path.join(mf_dir, file + '.txt')
                sf_path = os.path.join(sf_dir, file + '.txt')
                cls_name = items[1]
                if self.test_mode:
                    # if cls_name=='RoadAccidents_VRUvsVRU':
                    #     start_end_couples = [int(x) for x in items[3:]]
                    #     num_frames = int(items[2])
                    #     features_list.append((lf_path, mf_path, sf_path, num_frames, start_end_couples))
                    start_end_couples = [int(x) for x in items[3:]]
                    num_frames = int(items[2])
                    features_list.append((lf_path, mf_path, sf_path, num_frames, start_end_couples))
                elif ("Normal" == cls_name) == self.is_normal:
                    features_list.append((lf_path, mf_path, sf_path))

        return features_list

