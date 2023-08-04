import matplotlib.pyplot as plt
import torch
from sklearn.metrics import auc, roc_curve, precision_recall_curve
from tqdm import tqdm
import numpy as np


def test(dataloader, model, test_dataset, viz, device):
    with torch.no_grad():
        model.eval()
        pred = torch.zeros(0, device=device)
        gt = torch.zeros(0, device=device)

        for input, label, start_end_couples, num_frames in tqdm(dataloader):
            input = input.to(device)
            input = input.permute(1, 0, 2, 3)
            score_abnormal, score_normal, feat_select_abn, feat_select_normal, feat_abn_bottom, feat_select_normal_bottom, scores, \
                scores_nor_bottom, scores_nor_abn_bag, feat_magnitudes = model(inputs=input)
            logits = torch.squeeze(scores) # remove the first dimension (batch_size=1)
            segment = num_frames.item() // 32
            logits = logits.repeat_interleave(segment, dim=0)
            pred = torch.cat((pred, logits), dim=0)
            # make ground truth
            cur_gt = torch.zeros(logits.size(), device=device)
            for i in range(0, len(start_end_couples) - 1, 2):
                if start_end_couples[i].item() != -1 and start_end_couples[i + 1].item() != -1:
                    couple = start_end_couples[i:i + 2]
                    cur_gt[couple[0].item():couple[1].item(), label] = 1.0
            gt = torch.cat((gt, cur_gt), dim=0)
        pred = pred.view(-1)
        pred = pred.cpu().detach().numpy()
        gt = gt.view(-1)
        gt = gt.cpu().detach().numpy()
        fpr, tpr, threshold = roc_curve(gt, pred)
        rec_auc = auc(fpr, tpr)
        print('\n' + str(test_dataset) + ' auc : ' + str(rec_auc) + '\n')

        # precision, recall, th = precision_recall_curve(list(gt), pred)
        # pr_auc = auc(recall, precision)
        # viz.plot_lines(str(test_dataset)+' PR_AUC', y=pr_auc)
        viz.plot_lines(str(test_dataset)+' AUC', y=rec_auc)
        # viz.lines(name=str(test_dataset) + ' scores', Y=pred)
        viz.lines(str(test_dataset)+' ROC', Y=tpr, X=fpr)
        return fpr, tpr, rec_auc
