"""Reusable CV harness. Every experiment goes through run_cv so scores stay comparable."""
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED, NFOLDS = 42, 5

# Folder layout (see README.md). Scripts anywhere in the repo can import these.
ROOT = Path(__file__).resolve().parents[1]   # repo root
DATA, OOF, TEST, SUBS = ROOT / 'data', ROOT / 'outputs/oof', ROOT / 'outputs/test', ROOT / 'submissions'

# Downcasting on read: int64 -> int8 etc. cuts memory ~6x.
DTYPES = {'id': 'int32', 'Age': 'int8', 'Annual_Income_USD': 'float32',
          'Daily_Commute_km': 'float32', 'Number_of_Cars_Owned': 'int8',
          'Charging_Stations_Near_Home': 'int8', 'Charging_Stations_Near_Work': 'int8',
          'Environmental_Concern_Level': 'float32', 'Gender': 'category',
          'City_Type': 'category', 'Current_Car_Type': 'category',
          'Home_Charging_Possible': 'category', 'Subsidy_Available': 'category',
          'Range_Anxiety_Level': 'category'}


def load():
    tr = pd.read_csv(DATA / 'train.csv', dtype=DTYPES)
    te = pd.read_csv(DATA / 'test.csv', dtype=DTYPES)
    y = (tr.pop('Will_Buy_EV') == 'Yes').astype('int8')
    return tr.drop(columns='id'), y, te.drop(columns='id'), te.id


def run_cv(make_model, X, y, name):
    """Returns (oof_predictions, mean_auc). Identical folds every call -> comparable scores."""
    skf = StratifiedKFold(NFOLDS, shuffle=True, random_state=SEED)
    oof, scores = np.zeros(len(X)), []
    for itr, iva in skf.split(X, y):
        m = make_model()
        m.fit(X.iloc[itr], y.iloc[itr])
        oof[iva] = m.predict_proba(X.iloc[iva])[:, 1]
        scores.append(roc_auc_score(y.iloc[iva], oof[iva]))
    print(name.ljust(22), 'folds=' + ' '.join('%.5f' % s for s in scores))
    print(' '.ljust(22), 'mean=%.5f  std=%.5f  oof=%.5f'
          % (np.mean(scores), np.std(scores), roc_auc_score(y, oof)))
    return oof, np.mean(scores)


def fold_scores(oof, y):
    """Per-fold AUC recovered from an OOF vector (same fixed splits)."""
    skf = StratifiedKFold(NFOLDS, shuffle=True, random_state=SEED)
    return np.array([roc_auc_score(y.iloc[iva], oof[iva]) for _, iva in skf.split(oof, y)])


def paired(oof_new, oof_ref, y, name):
    """Paired per-fold comparison. Fold difficulty cancels -> ~40x sharper than the
    unpaired 0.0008 floor. REAL if mean > 2*sd and every fold moves the same way."""
    d = fold_scores(oof_new, y) - fold_scores(oof_ref, y)
    up = int((d > 0).sum())
    real = abs(d.mean()) > 2 * d.std(ddof=1) and up in (0, NFOLDS)
    print('%-22s paired: mean %+.5f  sd %.5f  up %d/%d  -> %s'
          % (name, d.mean(), d.std(ddof=1), up, NFOLDS, 'REAL' if real else 'noise'))
    return d


if __name__ == '__main__':
    X, y, Xte, te_id = load()
    cats = X.select_dtypes('category').columns.tolist()
    print('memory: %.1f MB   categoricals: %s' % (X.memory_usage(deep=True).sum() / 1e6, cats))
    print()

    # Model A: logistic regression needs one-hot encoding + feature scaling.
    Xd = pd.get_dummies(X, columns=cats, drop_first=True)
    oof_lr, _ = run_cv(lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
                       Xd, y, 'LogisticRegression')

    # Model B: LightGBM consumes category dtype directly, no encoding needed.
    oof_lgb, _ = run_cv(lambda: lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05,
                                                   num_leaves=31, verbose=-1, random_state=SEED),
                        X, y, 'LightGBM')

    np.save(OOF / 'oof_lr.npy', oof_lr)
    np.save(OOF / 'oof_lgb.npy', oof_lgb)
    print()
    print('50/50 blend oof=%.5f' % roc_auc_score(y, 0.5 * oof_lr + 0.5 * oof_lgb))
