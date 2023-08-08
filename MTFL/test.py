import matplotlib.pyplot as plt
import torch
from sklearn.metrics import auc, roc_curve, precision_recall_curve
from tqdm import tqdm
import numpy as np


def get_gt(start_end_couples, num_frames, device):
    gt = torch.zeros(num_frames).to(device)
    if start_end_couples is not None and num_frames is not None:
        for i in range(0, len(start_end_couples) - 1, 2):
            if start_end_couples[i].item() != -1 and start_end_couples[i + 1].item() != -1:
                couple = start_end_couples[i:i + 2]
                gt[couple[0].item():couple[1].item()] = 1.0

    return gt

def test(dataloader, model, seg_num, test_dataset, viz, device):
    with torch.no_grad():
        model.eval()
        pred = torch.zeros(0, device=device)
        gt = torch.zeros(0, device=device)

        for input1, input2, input3, label, start_end_couples, num_frames in tqdm(dataloader):
            input1 = input1.to(device)
            input2 = input2.to(device)
            input3 = input3.to(device)
            score_abnormal, score_normal, feat_select_abn, feat_select_normal, feat_select_abn, feat_select_abn,\
                scores, feat_select_abn, feat_select_abn, feat_magnitudes = model(input1, input2, input3)
            sig = torch.squeeze(scores, dim=(0,2)) # T
            segment = num_frames.item() // seg_num
            sig = sig.repeat_interleave(segment) # Frames
            if len(sig) < num_frames.item():
                last_ele = sig[-1]
                sig = torch.cat((sig, last_ele.repeat(num_frames.item()-len(sig)))) # 1 x Frames

            pred = torch.cat((pred, sig))
            cur_gt = get_gt(start_end_couples, num_frames, device)
            gt = torch.cat((gt, cur_gt))

        pred = pred.cpu().detach().numpy()
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
