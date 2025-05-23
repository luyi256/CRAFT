import torch
from sklearn.metrics import average_precision_score, roc_auc_score
import numpy as np



def get_node_classification_metrics(predicts: torch.Tensor, labels: torch.Tensor):
    """
    get metrics for the node classification task
    :param predicts: Tensor, shape (num_samples, )
    :param labels: Tensor, shape (num_samples, )
    :return:
        dictionary of metrics {'metric_name_1': metric_1, ...}
    """
    predicts = predicts.cpu().detach().numpy()
    labels = labels.cpu().numpy()

    roc_auc = roc_auc_score(y_true=labels, y_score=predicts)

    return {'roc_auc': roc_auc}



def get_link_prediction_metrics(y_pred_pos: np.ndarray, y_pred_neg: np.ndarray):

    y_pred_pos = y_pred_pos.reshape(-1, 1)
    optimistic_rank = (y_pred_neg > y_pred_pos).sum(axis=1)
    pessimistic_rank = (y_pred_neg >= y_pred_pos).sum(axis=1)
    ranking_list = 0.5 * (optimistic_rank + pessimistic_rank) + 1
    mrr_list = 1./ranking_list.astype(np.float32)

    return {
            'mrr': mrr_list.mean(),
            }

