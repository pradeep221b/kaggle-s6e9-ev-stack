"""One runner for every base model, identical on every machine.

    py run_stream.py --model lgb  --seed 7          # 5-fold OOF + fold-avg test  -> outputs/oof/oof_lgb_s7.npy, outputs/test/test_lgb_s7.npy
    py run_stream.py --model tabm --seed 7 --refit  # additionally refit on 100% train -> outputs/test/test_tabm_s7_full.npy
    py run_stream.py --model lgb  --seed 7 --smoke  # fold 0 only, few rounds, sanity check

Models: lgb | xgb | cb   (public 0.94608 depth-5 recipe on feats_v10, nested TE, honest inner early stop)
        tabm | ftt | realmlp | tabr   (pytabkit on GPU; in-fold TE(income) + digits, like nnmodel.TEMLP; tabr = retrieval NN)
Folds are ALWAYS StratifiedKFold(5, shuffle, seed 42) on the untouched data/train.csv -> arrays from any machine merge.
Skips a model/seed whose OOF file is already complete. Safe to re-run after a crash.
"""
import argparse, sys, time, warnings, logging, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings('ignore')
for _n in ('pytorch_lightning', 'lightning', 'lightning.pytorch'): logging.getLogger(_n).setLevel(logging.ERROR)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
from cv import load, OOF, TEST, SEED, NFOLDS

ap = argparse.ArgumentParser()
ap.add_argument('--model', required=True, choices=['lgb', 'xgb', 'cb', 'tabm', 'ftt', 'realmlp', 'tabr'])
ap.add_argument('--seed', type=int, default=42)
ap.add_argument('--refit', action='store_true', help='also refit on 100%% train for the test prediction')
ap.add_argument('--smoke', action='store_true')
A = ap.parse_args()
TAG = '%s_s%d' % (A.model, A.seed)
GBDT = A.model in ('lgb', 'xgb', 'cb')

X, y, Xte, _ = load()
yv = y.values

# ---------------- GBDT streams ----------------
if GBDT:
    import lightgbm as lgb, xgboost as xgb, catboost as cb
    from feats_v10 import build, add_te
    Dtr, Dte = build(X, Xte)
    P_LGB = dict(objective='binary', metric='auc', learning_rate=0.006, num_leaves=32, max_depth=5,
                 min_child_samples=15, feature_fraction=0.30, bagging_fraction=0.85, bagging_freq=1,
                 lambda_l1=0.071, lambda_l2=2.0, max_bin=2048, n_jobs=-1, verbose=-1)
    P_XGB = dict(objective='binary:logistic', eval_metric='auc', tree_method='hist', device='cuda',
                 learning_rate=0.008, max_depth=5, min_child_weight=20, subsample=0.85, colsample_bytree=0.70,
                 colsample_bylevel=0.75, gamma=0.30, reg_alpha=0.10, reg_lambda=2.0,
                 n_estimators=300 if A.smoke else 25000, early_stopping_rounds=400)
    P_CB = dict(loss_function='Logloss', eval_metric='AUC', learning_rate=0.025, depth=5, l2_leaf_reg=3.0,
                iterations=300 if A.smoke else 12000, task_type='GPU', verbose=0)

    def fit_gbdt(Xtr, ytr, Xva, Xt, rounds=None):
        """rounds=None -> early stop on an inner 10% split (val fold never touches the stopping decision).
        rounds=int  -> fixed length, used for the full-data refit."""
        if rounds is None:
            Xa, Xb, ya, yb = train_test_split(Xtr, ytr, test_size=0.1, stratify=ytr, random_state=A.seed)
        else:
            Xa, ya, Xb, yb = Xtr, ytr, None, None
        if A.model == 'lgb':
            m = lgb.train({**P_LGB, 'random_state': A.seed}, lgb.Dataset(Xa, ya),
                          rounds or (300 if A.smoke else 30000),
                          valid_sets=None if rounds else [lgb.Dataset(Xb, yb)],
                          callbacks=None if rounds else [lgb.early_stopping(400, verbose=False)],
                          keep_training_booster=True)
            it = rounds or m.best_iteration
            return (m.predict(Xva, num_iteration=it) if Xva is not None else None), m.predict(Xt, num_iteration=it), it
        if A.model == 'xgb':
            p = {**P_XGB, 'random_state': A.seed}
            if rounds: p.update(n_estimators=rounds); p.pop('early_stopping_rounds')
            m = xgb.XGBClassifier(**p).fit(Xa, ya, eval_set=None if rounds else [(Xb, yb)], verbose=False)
            return (m.predict_proba(Xva)[:, 1] if Xva is not None else None), m.predict_proba(Xt)[:, 1], rounds or m.best_iteration
        p = {**P_CB, 'random_seed': A.seed}
        if rounds: p.update(iterations=rounds)
        m = cb.CatBoostClassifier(**p).fit(Xa, ya, eval_set=None if rounds else (Xb, yb),
                                           early_stopping_rounds=None if rounds else 400)
        return (m.predict_proba(Xva)[:, 1] if Xva is not None else None), m.predict_proba(Xt)[:, 1], rounds or m.get_best_iteration()

# ---------------- NN streams ----------------
else:
    from feats import digits
    from nnmodel import TEMLP
    from pytabkit import TabM_D_Classifier, FTT_D_Classifier, RealMLP_TD_Classifier, TabR_S_D_Classifier
    Xd = digits(X); Xted = digits(Xte).reindex(columns=Xd.columns, fill_value=0)  # test may grow a digit col
    if A.model in ('ftt', 'tabr'): Xd, Xted = X, Xte  # tokens/retrieval are expensive; raw 13 + income_te

    class NN(TEMLP):
        def fit(self, X, y):
            k = dict(device='cuda', random_state=self.seed, batch_size=1024, verbosity=0)
            if A.model == 'tabm': self.m = TabM_D_Classifier(n_epochs=self.n_epochs, patience=16, **k)
            elif A.model == 'ftt': self.m = FTT_D_Classifier(max_epochs=self.n_epochs, es_patience=8, **k)
            elif A.model == 'tabr': self.m = TabR_S_D_Classifier(n_epochs=self.n_epochs, patience=8, **k)
            else: self.m = RealMLP_TD_Classifier(n_epochs=self.n_epochs, **k)
            self.m.fit(self._f(X, y), y); return self
    EPOCHS = {'tabm': 60, 'ftt': 40, 'realmlp': 48, 'tabr': 30}[A.model]
    if A.smoke: EPOCHS = 2

# ---------------- run ----------------
f_oof, f_test = OOF / ('oof_%s.npy' % TAG), TEST / ('test_%s.npy' % TAG)
done = f_oof.exists() and (np.load(f_oof) == 0).mean() < 0.01
if done and not A.smoke:
    print('%s OOF already complete, skip CV' % TAG, flush=True)
else:
    t = time.time(); oof, tpred, iters = np.zeros(len(y)), np.zeros(len(Xte)), []
    folds = list(StratifiedKFold(NFOLDS, shuffle=True, random_state=SEED).split(X, y))
    for f, (itr, iva) in enumerate(folds[:1] if A.smoke else folds):
        if GBDT:
            Xtr, (Xva, Xt) = add_te(Dtr.iloc[itr], yv[itr], [Dtr.iloc[iva], Dte])
            oof[iva], tp, it = fit_gbdt(Xtr, yv[itr], Xva, Xt); iters.append(it)
        else:
            m = NN(EPOCHS, A.seed).fit(Xd.iloc[itr], y.iloc[itr])
            oof[iva], tp = m.predict_proba(Xd.iloc[iva])[:, 1], m.predict_proba(Xted)[:, 1]
        tpred += tp / NFOLDS
        print('%s fold%d=%.5f %s (%.0fs)' % (TAG, f, roc_auc_score(y.iloc[iva], oof[iva]),
              ('iters=%d' % iters[-1]) if GBDT else '', time.time() - t), flush=True)
        if not A.smoke: np.save(f_oof, oof); np.save(f_test, tpred)
    if not A.smoke:
        print('%s oof=%.5f%s' % (TAG, roc_auc_score(y, oof), ('  mean_iters=%d' % np.mean(iters)) if GBDT else ''), flush=True)
        if GBDT: np.save(OOF / ('iters_%s.npy' % TAG), np.array(iters))

if A.refit and not A.smoke:
    t = time.time()
    if GBDT:
        rounds = int(np.load(OOF / ('iters_%s.npy' % TAG)).mean())
        Xtr, (Xt,) = add_te(Dtr, yv, [Dte])
        _, tp, _ = fit_gbdt(Xtr, yv, None, Xt, rounds=rounds)
    else:
        tp = NN(EPOCHS, A.seed).fit(Xd, y).predict_proba(Xted)[:, 1]
    np.save(TEST / ('test_%s_full.npy' % TAG), tp)
    print('%s full refit done (%.0fs)' % (TAG, time.time() - t), flush=True)
