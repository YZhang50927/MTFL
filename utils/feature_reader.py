from os import path
import numpy as np
import torch


def to_segments(data, num=32):
    """
	These code is taken from:
	https://github.com/rajanjitenpatel/C3D_feature_extraction/blob/b5894fa06d43aa62b3b64e85b07feb0853e7011a/extract_C3D_feature.py#L805
	:param data: list of features of a certain video
	:return: list of 32 segments
	"""

    Segments_Features = []
    thirty2_shots = np.round(np.linspace(0, len(data) - 1, num=num + 1)).astype(int)
    for ss, ee in zip(thirty2_shots[:-1], thirty2_shots[1:]):
        if ss == ee:
            temp_vect = data[min(ss, data.shape[0] - 1), :]
        else:
            temp_vect = data[ss:ee, :].mean(axis=0)

        temp_vect = temp_vect / np.linalg.norm(temp_vect)
        if np.linalg.norm == 0:
            print("Feature norm is 0")
            exit()
        if len(temp_vect) != 0:
            Segments_Features.append(temp_vect.tolist())

    return Segments_Features


def read_features(file_path, cache=None, model='mvit'):
    if cache is not None and file_path in cache:
        return cache[file_path]

    if not path.exists(file_path):
        raise Exception(f"Feature doesn't exist: {file_path}")
    features = np.load(file_path, allow_pickle=True)
    seg_feat = to_segments(features, 32)

    if cache is not None:
        cache[file_path] = seg_feat

    return torch.tensor(seg_feat)


#if __name__ == '__main__':
#    feat_path = '/media/DataDrive/yiling/features/SLOWFAST_8x8_R50/Abuse/Abuse001_x264.npy'
#    output = read_features(feat_path)
