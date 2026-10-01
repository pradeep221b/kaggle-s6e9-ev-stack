"""Faithful, HONEST port of najiama's public recipe (LB 0.94612 single model). Our earlier replication
(naji_exp.py) used lr .02 / fixed 8000 rounds / bagging on / no *_org_mean and scored 0.94500 -> wrongly buried.
Recipe: drop cars; digits(-4..3) of every numeric; *_org_mean from the 10k original; every numeric also as a string
category; dual in-fold TargetEncoder (smooth='auto' and 10) on all categories; frequency enc; 4 spike flags;
LGBM depth 5, lr .005, colsample .3, max_bin 1024, up to 40k rounds. Only change: early stop on an INNER 10% of the
train fold (patience 700), never on the scoring fold. Fixed 5 folds seed 42 -> stackable.
Writes outputs/oof/oof_naji_s<seed>.npy, outputs/test/test_naji_s<seed>.npy, outputs/oof/iters_naji_s<seed>.npy.
Usage: py experiments/naji_stream.py [seed]"""
import sys, time, numpy as np, pandas as pd, lightgbm as lgb
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import TargetEncoder
from sklearn.metrics import roc_auc_score
from cv import DATA, OOF, TEST, SEED, NFOLDS

SEED_M = int(sys.argv[1]) if len(sys.argv) > 1 else 42
import os, json
OVERRIDE = json.loads(os.environ.get('NAJI_PARAMS', '{}'))   # e.g. Optuna winner; tag gets a suffix so it is a separate stream
TAG = 'naji%s_s%d' % (os.environ.get('NAJI_TAG', ''), SEED_M)
T = 'Will_Buy_EV'
train = pd.read_csv(DATA / 'train.csv'); test = pd.read_csv(DATA / 'test.csv')
orig = pd.read_csv(Path(__file__).resolve().parents[1] / 'data/EV_Adoption_and_Range_Anxiety_Dataset.csv')
train[T] = train[T].map({'Yes': 1, 'No': 0}); orig[T] = orig[T].map({'Yes': 1, 'No': 0})
train['is_train'] = 1; test['is_train'] = 0; test[T] = np.nan
c = pd.concat([train, test], ignore_index=True).drop(columns=['Number_of_Cars_Owned'])
cat_cols = c.select_dtypes(include=['object', 'string']).columns.tolist()
num_cols = [k for k in c.columns if k not in cat_cols + ['id', 'is_train', T]]
for k in list(num_cols):
    for p in range(-4, 4):
        c[f'{k}_digit{p}'] = (c[k].fillna(0) // (10 ** p) % 10).astype('int8'); num_cols.append(f'{k}_digit{p}')
og = orig[T].mean()
for k in cat_cols + num_cols:
    if k in orig.columns:
        c[f'{k}_org_mean'] = c[k].map(orig.groupby(k, observed=False)[T].mean()).fillna(og).astype(float)
n2c = []
for k in num_cols:
    c[f'{k}_cat'] = c[k].fillna('NaN').astype(str); n2c.append(f'{k}_cat')
all_cats = cat_cols + n2c
for k in all_cats:
    c[f'{k}_fe'] = c[k].map(c[k].value_counts(normalize=True)).astype(float).fillna(0.0)
c['is_30k_spike'] = (c['Annual_Income_USD'] == 30000.0).astype('int8')
c['is_millionaire_cliff'] = (c['Annual_Income_USD'] >= 170537.0).astype('int8')
c['is_dead_zone'] = ((c['Annual_Income_USD'] >= 38000.0) & (c['Annual_Income_USD'] <= 42000.0)).astype('int8')
c['is_env_hater'] = (c['Environmental_Concern_Level'] == 1).astype('int8')
train = c[c.is_train == 1].drop(columns=['is_train']); test = c[c.is_train == 0].drop(columns=['is_train', T])
ev = [k for k in train.columns if k not in ['id', T] and pd.api.types.is_numeric_dtype(train[k])]
cm = train[ev].corr().abs(); up = cm.where(np.triu(np.ones(cm.shape), k=1).astype(bool))
drop = set(k for k in up.columns if any(up[k] == 1.0)) | set(k for k in train.columns if train[k].nunique() == 1) \
       | set(k for k in test.columns if test[k].nunique() == 1)
drop = [k for k in drop if k not in ['id', T]]
train = train.drop(columns=drop); test = test.drop(columns=drop)
FEAT = [k for k in test.columns if k != 'id']; TEC = [k for k in all_cats if k not in drop]
print('features %d, target-encoded cats %d, dropped %d' % (len(FEAT), len(TEC), len(drop)), flush=True)
X, y, Xt = train[FEAT], train[T].astype(int), test[FEAT]

P = dict(n_estimators=40000, learning_rate=0.005, max_depth=5, num_leaves=32, min_child_samples=10, subsample=0.8,
         colsample_bytree=0.3, reg_alpha=0.071, reg_lambda=2.0, max_bin=1024, random_state=SEED_M,
         feature_pre_filter=False, metric='auc', n_jobs=-1, verbose=-1)
P.update(OVERRIDE)
if OVERRIDE: print('param override:', OVERRIDE, flush=True)

def encode(Xa, ya, others):
    """Dual TE fit on Xa (train rows get cross-fitted values), applied to each frame in others."""
    ta = TargetEncoder(shuffle=True, cv=5, smooth='auto', random_state=SEED)
    t10 = TargetEncoder(shuffle=True, cv=5, smooth=10.0, random_state=SEED)
    A = ta.fit_transform(Xa[TEC], ya); B = t10.fit_transform(Xa[TEC], ya)
    def build(F, ea, eb):
        F = F.drop(columns=TEC).copy()
        for i, k in enumerate(TEC):
            F[f'{k}_TE_auto'] = ea[:, i].astype('float32'); F[f'{k}_TE_10'] = eb[:, i].astype('float32')
        return F
    return build(Xa, A, B), [build(F, ta.transform(F[TEC]), t10.transform(F[TEC])) for F in others]

f_oof = OOF / f'oof_{TAG}.npy'
done = f_oof.exists() and (np.load(f_oof) == 0).mean() < 0.01
t0 = time.time(); oof, tp, iters = np.zeros(len(X)), np.zeros(len(Xt)), []
if done:
    iters = list(np.load(OOF / f'iters_{TAG}.npy')); print('%s CV already complete, skip' % TAG, flush=True)
for f, (itr, iva) in enumerate([] if done else StratifiedKFold(NFOLDS, shuffle=True, random_state=SEED).split(X, y)):
    Xtr, ytr = X.iloc[itr], y.iloc[itr]
    ia, ib = train_test_split(np.arange(len(itr)), test_size=0.1, stratify=ytr, random_state=SEED_M)  # inner ES split
    Xa, (Xb, Xva, Xte_f) = encode(Xtr.iloc[ia], ytr.iloc[ia], [Xtr.iloc[ib], X.iloc[iva], Xt])
    m = lgb.LGBMClassifier(**P).fit(Xa, ytr.iloc[ia], eval_set=[(Xb, ytr.iloc[ib])],
                                    callbacks=[lgb.early_stopping(700, verbose=False)])
    oof[iva] = m.predict_proba(Xva)[:, 1]; tp += m.predict_proba(Xte_f)[:, 1] / NFOLDS; iters.append(m.best_iteration_)
    np.save(OOF / f'oof_{TAG}.npy', oof); np.save(TEST / f'test_{TAG}.npy', tp); np.save(OOF / f'iters_{TAG}.npy', np.array(iters))
    print('%s fold%d=%.5f iters=%d (%.0fs)' % (TAG, f, roc_auc_score(y.iloc[iva], oof[iva]), iters[-1], time.time() - t0), flush=True)
if not done: print('%s oof=%.5f  mean_iters=%d' % (TAG, roc_auc_score(y, oof), np.mean(iters)), flush=True)
# full-data refit: TE on all train rows, fixed rounds = 1.1 x mean early-stopped iters (10% more data)
t1 = time.time(); Xa, (Xte_f,) = encode(X, y, [Xt])
m = lgb.LGBMClassifier(**dict(P, n_estimators=int(np.mean(iters) * 1.1))).fit(Xa, y)
np.save(TEST / f'test_{TAG}_full.npy', m.predict_proba(Xte_f)[:, 1])
print('%s full refit done rounds=%d (%.0fs)' % (TAG, int(np.mean(iters) * 1.1), time.time() - t1), flush=True)
