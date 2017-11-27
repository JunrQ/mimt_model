""" Model validation metrics. Using Inferencer interface from tensorpack.
"""
from functional import seq
from sklearn import metrics
from sklearn.preprocessing import binarize
import numpy as np


def _filter_all_negative(logits, label):
    """ AUC and mAP metrics only work when positive sample presents.
    However, in some batches, a class' ground truth labels could be all negative.
    Thus, we need to filter out those classes to avoid computation error.
    """
    keep_mask = np.any(label, axis=0)
    return logits[:, keep_mask], label[:, keep_mask]

def pred_from_score(scores, threshold):
    """ Convert real number score (aka. confidence) to 0 or 1.

    Args:
        scores: Real number scores, ndarray of shape [N, L].
        threshold: Threshold of true prediction. A scalar or
            an array of length L for a per-label threshold.
    """
    cut = scores - threshold
    return binarize(cut, threshold=0.0, copy=False)

def calcu_one_metric(scores, labels, metric, threshold=None):
    ans = None

    if metric == 'mean_average_precision':
        scores, labels = _filter_all_negative(scores, labels)
        ans = metrics.average_precision_score(labels, scores)

    elif metric == 'macro_auc':
        scores, labels = _filter_all_negative(scores, labels)
        ans = metrics.roc_auc_score(labels, scores, average='macro')

    elif metric == 'micro_auc':
        scores, labels = _filter_all_negative(scores, labels)
        ans = metrics.roc_auc_score(labels, scores, average='micro')

    elif metric == 'macro_f1':
        scores, labels = _filter_all_negative(scores, labels)
        pred = pred_from_score(scores, threshold)
        ans = metrics.f1_score(labels, pred, average='macro')

    elif metric == 'micro_f1':
        scores, labels = _filter_all_negative(scores, labels)
        pred = pred_from_score(scores, threshold)
        ans = metrics.f1_score(labels, pred, average='micro')

    elif metric == 'ranking_mean_average_precision':
        ans = metrics.label_ranking_average_precision_score(labels, scores)

    elif metric == 'coverage':
        cove = metrics.coverage_error(labels, scores)
        # see http://scikit-learn.org/stable/modules/model_evaluation.html#coverage-error
        ans = cove - 1

    elif metric == 'ranking_loss':
        ans = metrics.label_ranking_loss(labels, scores)

    elif metric == 'one_error':
        top_score = np.argmax(scores, axis=1)
        top_label = labels[range(len(top_score)), top_score]
        ans = 1 - np.sum(top_label) / len(top_label)

    else:
        raise f"unsuppored metric: {metric}"

    return ans


def calcu_metrics(scores, labels, queries, threshold):
    """ Calculate metrics specified by queries from logits and labels.
    """
    ans = seq(queries).map(
        lambda m: calcu_one_metric(scores, labels, m, threshold)).list()
    return ans