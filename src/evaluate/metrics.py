import numpy as np
from sklearn.metrics import (
    roc_curve,
    auc,
    average_precision_score,
)


def bedroc_score(y_true, y_pred, decreasing=True, alpha=20.0):
    """BEDROC metric implemented according to Truchon and Bayley.

    The Boltzmann Enhanced Descrimination of the Receiver Operator
    Characteristic (BEDROC) score is a modification of the Receiver Operator
    Characteristic (ROC) score that allows for a factor of *early recognition*.

    References:
        The original paper by Truchon et al. is located at `10.1021/ci600426e
        <http://dx.doi.org/10.1021/ci600426e>`_.

    Args:
        y_true (array_like):
            Binary class labels. 1 for positive class, 0 otherwise.
        y_pred (array_like):
            Prediction values.
        decreasing (bool):
            True if high values of ``y_pred`` correlates to positive class.
        alpha (float):
            Early recognition parameter.

    Returns:
        float:
            Value in interval [0, 1] indicating degree to which the predictive
            technique employed detects (early) the positive class.
    """

    assert len(y_true) == len(
        y_pred
    ), "The number of scores must be equal to the number of labels"

    big_n = len(y_true)
    n = sum(y_true == 1)

    if decreasing:
        order = np.argsort(-y_pred)
    else:
        order = np.argsort(y_pred)

    m_rank = (y_true[order] == 1).nonzero()[0] + 1
    s = np.sum(np.exp(-alpha * m_rank / big_n))
    r_a = n / big_n
    rand_sum = r_a * (1 - np.exp(-alpha)) / (np.exp(alpha / big_n) - 1)
    fac = (
        r_a
        * np.sinh(alpha / 2)
        / (np.cosh(alpha / 2) - np.cosh(alpha / 2 - alpha * r_a))
    )
    cte = 1 / (1 - np.exp(alpha * (1 - r_a)))
    return s * fac / rand_sum + cte


def roc_auc(y_pred, y_scores):
    fpr, tpr, _ = roc_curve(y_pred, y_scores, pos_label=1)
    return auc(fpr, tpr)


def pr_auc(y_pred, y_scores):
    return average_precision_score(y_pred, y_scores)


def precision_at_L(df, L, col_score="score", col_pred="tp"):
    assert L > 0
    p = []

    for m in list(df["ligand"].unique()):
        m_df = df[df["ligand"] == m]
        m_df.sort_values(by=[col_score], ascending=False, inplace=True)

        # Calculate precision@n for given ligand
        Xi_L = sum(m_df[col_pred].head(L).values)
        p.append(Xi_L / L)

    return np.nanmean(p)


def recall_at_L(df, L, col_score="score", col_pred="tp"):
    assert L > 0
    r = []

    for m in list(df["ligand"].unique()):
        m_df = df[df["ligand"] == m]
        m_df.sort_values(by=[col_score], ascending=False, inplace=True)

        # Calculate recall@n for given ligand
        Xi = sum(m_df[col_pred].values)
        Xi_L = sum(m_df[col_pred].head(L).values)

        if Xi > 0:
            r.append(Xi_L / Xi)
        else:
            r.append(np.nan)

    return np.nanmean(r)

def _max_metric(y_true, y_pred, f):
    # Get number of positives and negatives
    P = sum(y_true)
    N = len(y_true) - P

    # Calculate FPR and TPr from ROC curve (because is faster...)
    FPR, TPR, THRESHOLD = roc_curve(y_true, y_pred, drop_intermediate=False)

    # Calculate metric over each threshold
    max_score = 0
    opt_thr = 0
    for fpr, tpr, thr in zip(FPR, TPR, THRESHOLD):
        # Confusion matrix unravelling from TPR and FPR
        tn = (1 - fpr) * N
        tp = tpr * P
        fn = P - tp
        fp = N - tn

        # Metric calculation
        score = f(tn, fp, fn, tp)
        if score >= max_score:
            max_score = score
            opt_thr = thr

    return max_score, opt_thr


def _f1(_, fp, fn, tp):
    try:
        return tp / (tp + 0.5 * (fp + fn))
    except ZeroDivisionError:
        return np.nan


def _mcc(tn, fp, fn, tp):
    try:
        return ((tp * tn) - (fp * fn)) / np.sqrt(
            (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
        )
    except ZeroDivisionError:
        return np.nan

def _balanced_accuracy(tn, fp, fn, tp):
    try:
        tpr = tp / (tp + fn)
        tnr = tn / (tn + fp)

        return (tpr + tnr) / 2
    except ZeroDivisionError:
        return np.nan


def max_f1(y_true, y_pred): return _max_metric(y_true, y_pred, _f1)
def max_mcc(y_true, y_pred): return _max_metric(y_true, y_pred, _mcc)
def max_balanced_accuracy(y_true, y_pred): return _max_metric(y_true, y_pred, _balanced_accuracy)
