from os import path
import numpy as np
import torch

def read_features(file_path, cache=None, model='i3d'):
    if cache is not None and file_path in cache:
        return cache[file_path]

    if not path.exists(file_path):
        raise Exception(f"Feature doesn't exist: {file_path}")
    features = None
    with open(file_path, 'r') as fp:
        data = fp.read().splitlines(keepends=False)
        if model.lower() == 'c3d':
            features = np.zeros((len(data), 4096))
        elif model.lower() == 'vst':
            features = np.zeros((len(data), 1024))
        elif model.lower() == 'concatenated':
            features = np.zeros((len(data), 12288))
        elif model.lower() == 'timesformer' or model.lower() == 'timesformerlarge':
            features = np.zeros((len(data), 768))
        elif model.lower() == 'slowfast':
            features = np.zeros((len(data), 2304))
        else:  # Else (for I3D)
            features = np.zeros((len(data), 1024))
        for i, line in enumerate(data):
            features[i, :] = [float(x) for x in line.split(' ')]

    features = torch.from_numpy(features).float()
    if cache is not None:
        cache[file_path] = features
    return features

