import matplotlib.pyplot as plt
import torch
from sklearn.metrics import auc, roc_curve, precision_recall_curve
from tqdm import tqdm
import numpy as np

def top_k_accuracy(scores, labels, topk=(1, 5)):
    """Calculate top k accuracy score.

    Args:
        scores (list[np.ndarray]): Prediction scores for each class.
        labels (list[int]): Ground truth labels.
        topk (tuple[int]): K value for top_k_accuracy. Default: (1, ).

    Returns:
        list[float]: Top k accuracy score for each k.
    """
    res = []
    labels = np.array(labels)[:, np.newaxis]
    for k in topk:
        max_k_preds = np.argsort(scores, axis=1)[:, -k:][:, ::-1]
        match_array = np.logical_or.reduce(max_k_preds == labels, axis=1)
        topk_acc_score = match_array.sum() / match_array.shape[0]
        res.append(topk_acc_score)

    return res


def test(dataloader, model, test_dataset, device):
    with torch.no_grad():
        model.eval()
        outputs = torch.zeros(0, device=device)
        labels = torch.zeros(0, device=device)

        for input, label in tqdm(dataloader):
            input = input.to(device)
            label = label.to(device)
            input = input.permute(1, 0, 2, 3)
            score_abnormal, score_normal, feat_select_abn, feat_select_normal, feat_abn_bottom, feat_select_normal_bottom, scores, \
                scores_nor_bottom, scores_nor_abn_bag, feat_magnitudes = model(inputs=input)
            outputs = torch.cat((outputs, score_abnormal))
            labels = torch.cat((labels, label))

        outputs = outputs.cpu().detach().numpy()
        labels = labels.cpu().detach().numpy()

        if test_dataset == 'UCF': # all road accidents in UCF are labelled as 13
            for row in outputs:
                max_value = max(row[13], row[14], row[15])
                row[13] = max_value
                row[14] = 0.0
                row[15] = 0.0

        res = top_k_accuracy(outputs, labels)
        print('\n' + str(test_dataset) + ' top1 : ' + str(res[0]) + ' top5 : ' + str(res[1]) + '\n')

        return res[0]
