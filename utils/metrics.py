import torch
from sklearn.metrics import average_precision_score, roc_auc_score
import numpy as np

def get_link_prediction_metrics(predicts: torch.Tensor, labels: torch.Tensor):
    """
    get metrics for the link prediction task
    :param predicts: Tensor, shape (num_samples, )
    :param labels: Tensor, shape (num_samples, )
    :return:
        dictionary of metrics {'metric_name_1': metric_1, ...}
    """
    predicts = predicts.cpu().detach().numpy()
    labels = labels.cpu().numpy()

    average_precision = average_precision_score(y_true=labels, y_score=predicts)
    roc_auc = roc_auc_score(y_true=labels, y_score=predicts)

    return {'average_precision': average_precision, 'roc_auc': roc_auc}

def get_full_rank_mrr(predicts: torch.Tensor, labels: torch.Tensor):
    """
    get metrics for the link prediction task
    :param predicts: Tensor, shape (num_samples, num_nodes)
    :param labels: Tensor, shape (num_samples, )
    :return:
        dictionary of metrics {'metric_name_1': metric_1, ...}
    """
    sorted_indices = torch.argsort(predicts, dim=1, descending=True)
    
    # Compute the rank of the correct label for each query
    ranks = (sorted_indices == labels.unsqueeze(1)).nonzero(as_tuple=True)[1] + 1  # Add 1 for 1-based indexing
    
    # Compute reciprocal ranks
    reciprocal_ranks = 1.0 / ranks.float()
    
    # Calculate MRR (mean of reciprocal ranks)
    mrr = reciprocal_ranks.mean().item()

    return {'mrr': mrr}


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
