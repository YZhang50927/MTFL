import matplotlib.pyplot as plt
import torch
from sklearn.metrics import auc, roc_curve, precision_recall_curve
from tqdm import tqdm
import numpy as np


def test(dataloader, model, test_dataset, viz, device):
    with torch.no_grad():
        model.eval()
        pred = torch.zeros(0, device=device)
        gt = []

        for input, start_end_couples, num_frames in tqdm(dataloader):
            input = input.to(device)
            input = input.permute(1, 0, 2, 3)
            score_abnormal, score_normal, feat_select_abn, feat_select_normal, feat_abn_bottom, feat_select_normal_bottom, scores, \
                scores_nor_bottom, scores_nor_abn_bag, feat_magnitudes = model(inputs=input)
            logits = torch.squeeze(scores, 2)
            logits = torch.mean(logits, 0)
            sig = logits
            segment = num_frames.item() // 32
            sig = sig.repeat_interleave(segment)
            pred = torch.cat((pred, sig))
            # make ground truth
            frames = 32 * segment
            cur_gt = np.zeros(frames)
            for i in range(0, len(start_end_couples) - 1, 2):
                if start_end_couples[i].item() != -1 and start_end_couples[i + 1].item() != -1:
                    couple = start_end_couples[i:i + 2]
                    cur_gt[couple[0].item():couple[1].item()] = 1.0
            gt = np.append(gt, cur_gt)

        pred = list(pred.cpu().detach().numpy())
        pred = np.array(pred)

        fpr, tpr, threshold = roc_curve(list(gt), pred)
        rec_auc = auc(fpr, tpr)
        print('\n' + str(test_dataset) + ' auc : ' + str(rec_auc) + '\n')

        # precision, recall, th = precision_recall_curve(list(gt), pred)
        # pr_auc = auc(recall, precision)
        # viz.plot_lines(str(test_dataset)+' PR_AUC', y=pr_auc)
        viz.plot_lines(str(test_dataset)+' AUC', y=rec_auc)
        # viz.lines(name=str(test_dataset) + ' scores', Y=pred)
        viz.lines(str(test_dataset)+' ROC', Y=tpr, X=fpr)
        return fpr, tpr, rec_auc
