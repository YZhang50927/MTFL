import torch.utils.data as data
import os
import torch
torch.set_default_tensor_type('torch.FloatTensor')

class_to_int = {
    'Normal': 0,
    'Abuse': 1,
    'Arrest': 2,
    'Arson': 3,
    'Assault': 4,
    'Burglary': 5,
    'Explosion': 6,
    'Fighting': 7,
    'Robbery': 8,
    'Shooting': 9,
    'Shoplifting': 10,
    'Stealing': 11,
    'Vandalism': 12,
    'RoadAccidents_EMVvsEMV': 13,
    'RoadAccidents_EMVvsVRU': 14,
    'RoadAccidents_VRUvsVRU': 15,
    'DangerousThrowing': 16,
    'Littering': 17
}

def read_features(feature_path):
    with open(feature_path, 'r') as file:
        lines = file.readlines()
    features = []
    for line in lines:
        # ??txt????????????,?????
        feature = [float(value) for value in line.strip().split()]
        features.append(feature)
    #features = to_segments(features, self.seg_num) # B x T x D
    features = torch.tensor(features).float() # B x T x D
    return features


class Dataset(data.Dataset):
    def __init__(self, args, is_normal=True, transform=None, test_mode=False, test_dataset='UCF'):
        self.split = args.split
        self.is_normal = is_normal
        self.dataset = test_dataset
        self.transform = transform
        self.test_mode = test_mode
        self._parse_list()
        self.num_frame = 0
        self.labels = None

    def _parse_list(self):
        # test dataset
        if self.dataset == 'VAD3':
            test_annotation_path = f"/media/DataDrive/yiling/annotation/recognition/splits/VAD/VAD_test_00{self.split}.txt"
        elif self.dataset == 'UCF':
            test_annotation_path = f"/media/DataDrive/yiling/annotation/recognition/splits/UCF/UCF_test_00{self.split}.txt"
        # test or train
        if self.test_mode:
            annotation_path = test_annotation_path
        else:
            annotation_path = f"/media/DataDrive/yiling/annotation/recognition/splits/VAD/VAD_train_00{self.split}.txt"

        feature_path = f"/media/DataDrive/yiling/features/recognition/split{self.split}"
        self.list = self._get_features_list(feature_path, annotation_path)

    def __getitem__(self, index):

        feature_path, label = self.list[index]
        features = read_features(feature_path)
        label = torch.tensor(label)
        return torch.unsqueeze(features, 0), label  # [N,B,T,F]

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
                cls = int(items[1])
                if self.test_mode:
                    features_list.append((feature_path, cls))
                elif (cls == class_to_int['Normal']) == self.is_normal:
                    features_list.append((feature_path, cls))

        return features_list

