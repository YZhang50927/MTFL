import tqdm
import torch


class DetectorValidation:

    def __init__(self, data_iter):
        self.data_iter = data_iter

    def get_scores(self, regressor):
        with torch.no_grad():
            y_trues, y_preds = torch.Tensor(), torch.Tensor()
            for features, couples, length, anomaly in tqdm.tqdm(self.data_iter):
                scores = regressor(features).squeeze()

                y_true = torch.zeros(length)
                y_pred = torch.zeros(length)

                segments_len = int(torch.div(length, 32))
                for couple in couples.view(-1, 2):
                    if couple[0] != -1:
                        y_true[couple[0]: couple[1]] = 1

                for i in range(32):
                    segment_start_frame = i * segments_len
                    segment_end_frame = (i + 1) * segments_len
                    y_pred[segment_start_frame: segment_end_frame] = scores[i]

                y_trues = torch.cat((y_trues, y_true), 0)
                y_preds = torch.cat((y_preds, y_pred), 0)
        return y_trues, y_preds

    def get_scores_normal(self, regressor):
        with torch.no_grad():
            y_preds_coarse, y_preds_fine = torch.Tensor(), torch.Tensor()
            for features, couples, length, anomaly in tqdm.tqdm(self.data_iter):
                if anomaly >= 0:
                    continue
                scores = regressor(features).squeeze()
                y_pred = torch.zeros(length)

                segments_len = int(torch.div(length, 32))

                for i in range(32):
                    segment_start_frame = i * segments_len
                    segment_end_frame = (i + 1) * segments_len
                    y_pred[segment_start_frame: segment_end_frame] = scores[i]

                y_preds_fine = torch.cat((y_preds_fine, y_pred), 0)
                y_preds_coarse = torch.cat((y_preds_coarse, scores), 0)
        return y_preds_coarse, y_preds_fine

    def get_scores_anomalous(self, regressor):
        with torch.no_grad():
            y_trues = y_preds = y_preds_coarse = y_trues_coarse = torch.Tensor()
            for features, couples, length, anomaly in tqdm.tqdm(self.data_iter):
                if anomaly == -1:
                    continue
                scores = regressor(features).squeeze()

                y_true = torch.zeros(length)
                y_pred = torch.zeros(length)
                y_true_coarse = torch.zeros(32)

                segments_len = int(torch.div(length, 32))
                for couple in couples.view(-1, 2):
                    if couple[0] != -1:
                        y_true[couple[0]: couple[1]] = 1
                        y_true_coarse[couple[0] // segments_len: couple[1] // segments_len + 1] = 1


                for i in range(32):
                    segment_start_frame = i * segments_len
                    segment_end_frame = (i + 1) * segments_len
                    y_pred[segment_start_frame: segment_end_frame] = scores[i]

                y_trues = torch.cat((y_trues, y_true), 0)
                y_preds = torch.cat((y_preds, y_pred), 0)
                y_trues_coarse = torch.cat((y_trues_coarse, y_true_coarse), 0)
                y_preds_coarse = torch.cat((y_preds_coarse, scores), 0)
        return y_trues, y_preds, y_trues_coarse, y_preds_coarse