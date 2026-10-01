"""Shared feature builders."""
import numpy as np


def digits(X):
    """(x // 10^k) % 10 for k=-4..3 on every numeric col. Exposes exact-value structure
    that 255-bin histograms cannot resolve. Round first: float32 noise at k<-1."""
    X = X.copy()
    for c in X.select_dtypes('number').columns:
        v = np.round(X[c].astype('float64'), 4)
        for k in range(-4, 4):
            d = (np.floor(v / 10.0 ** k) % 10).astype('int8')
            if d.nunique() > 1:
                X['%s_d%d' % (c, k)] = d
    return X


def additive(n_features):
    """interaction_constraints: each tree may split on one feature only -> additive model."""
    return [[i] for i in range(n_features)]
