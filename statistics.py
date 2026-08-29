"""
Statistical comparison utilities corresponding to the paper's
paired t-test and McNemar-test statements.

IMPORTANT:
A valid paired t-test needs paired per-case/per-fold measurements.
Do NOT run a t-test on only the single summary percentages in Table 1.
"""
import numpy as np
from scipy.stats import ttest_rel
from statsmodels.stats.contingency_tables import mcnemar

def paired_t_test(metric_a, metric_b):
    a=np.asarray(metric_a,dtype=float)
    b=np.asarray(metric_b,dtype=float)
    if a.shape != b.shape or a.size < 2:
        raise ValueError("Need >=2 paired measurements with the same shape.")
    return ttest_rel(a,b)

def mcnemar_from_predictions(y_true, pred_a, pred_b, exact=True):
    y_true=np.asarray(y_true)
    a=np.asarray(pred_a)
    b=np.asarray(pred_b)

    a_ok=(a==y_true)
    b_ok=(b==y_true)

    table=np.array([
        [np.sum(a_ok & b_ok), np.sum(a_ok & ~b_ok)],
        [np.sum(~a_ok & b_ok), np.sum(~a_ok & ~b_ok)]
    ])
    result=mcnemar(table, exact=exact, correction=not exact)
    return table, result
