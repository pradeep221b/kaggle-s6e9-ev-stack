"""Honest port of najiama Pure LGBM **V3** (author CV 0.94606 / LB 0.94637). = naji_stream.py (V1 port) plus the two V3
additions, nothing else:  (1) 4 "smooth key" string categories: floor(income), floor(income/100), floor(income/1000),
floor(commute);  (2) a THIRD in-fold TargetEncoder with smooth=100.0 on every TE column.
Same folds (seed 42), same inner-10% early stop, same params (lr .005, depth 5, colsample .3, max_bin 1024).
Writes outputs/oof/oof_naji3_s<seed>.npy, outputs/test/test_naji3_s<seed>.npy(+_full), outputs/oof/iters_naji3_s<seed>.npy.
Usage: py experiments/naji3_stream.py [seed]        env SMOKE=1 -> 40k rows, 300 rounds, fold 0 only (selftest)
       env NJOBS=4 -> lightgbm threads (default -1)"""
import os, sys, time, numpy as np, pandas as pd, lightgbm as lgb, xgboost as xgb, catboost as cb
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import TargetEncoder
from sklearn.metrics import roc_auc_score
from cv import DATA, OOF, TEST, SEED, NFOLDS

SEED_M = int(sys.argv[1]) if len(sys.argv) > 1 else 42
SMOKE = os.environ.get('SMOKE') == '1'
MODEL = os.environ.get('MODEL', 'lgb')          # lgb (V3 as published) | cb | xgb  (kospintr-style learners on the same V3 features)
VAR = os.environ.get('VAR', '')                # '' (V3) | keys (wider key ladder) | pair (income x RA, age x RA TE) | ladder (TE smooth 1,1000 added) | te<N> (label only; set TESEED=N)
TESEED = int(os.environ.get('TESEED', SEED))    # random_state of the in-fold TargetEncoders (V3 = 42)
TAG = 'naji3%s%s_s%d' % ('' if MODEL == 'lgb' else MODEL, VAR, SEED_M)
T = 'Will_Buy_EV'
train = pd.read_csv(DATA / 'train.csv'); test = pd.read_csv(DATA / 'test.csv')
if SMOKE: train = train.sample(40000, random_state=0).reset_index(drop=True); test = test.head(5000)
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
c['is_30k_spike'] = (c['Annual_Income_USD'] == 30000.0).astype('int8')
c['is_millionaire_cliff'] = (c['Annual_Income_USD'] >= 170537.0).astype('int8')
c['is_dead_zone'] = ((c['Annual_Income_USD'] >= 38000.0) & (c['Annual_Income_USD'] <= 42000.0)).astype('int8')
c['is_env_hater'] = (c['Environmental_Concern_Level'] == 1).astype('int8')
# --- V3 addition 1: smooth keys (Markus.JM) -> TE + freq like every other category
c['income_exact_int'] = np.floor(c['Annual_Income_USD']).astype(str)
c['income100_floor'] = np.floor(c['Annual_Income_USD'] / 100.0).astype(str)
c['income1000_floor'] = np.floor(c['Annual_Income_USD'] / 1000.0).astype(str)
c['commute_integer'] = np.floor(c['Daily_Commute_km']).astype(str)
SMOOTH = ['income_exact_int', 'income100_floor', 'income1000_floor', 'commute_integer']
if VAR == 'keys':   # 09-09 probe 2: more resolutions of the same axis (lucifer19 / megayak ladders)
    for d in (10, 500, 5000, 10000):
        c[f'income{d}_floor'] = np.floor(c['Annual_Income_USD'] / d).astype(str); SMOOTH.append(f'income{d}_floor')
    c['commute5_floor'] = np.floor(c['Daily_Commute_km'] / 5).astype(str); SMOOTH.append('commute5_floor')
if VAR == 'pair':   # 09-09 probe 3: yekenot pair keys as TE categories
    c['inc_x_ra'] = c['Annual_Income_USD'].astype(str) + '|' + c['Range_Anxiety_Level'].astype(str)
    c['age_x_ra'] = c['Age'].astype(str) + '|' + c['Range_Anxiety_Level'].astype(str)
    SMOOTH += ['inc_x_ra', 'age_x_ra']
if VAR == 'recipe':   # 09-09 probe 5: Deotte's generator scores as numeric features (worry = latent behind anxiety, buy = latent behind label)
    c['worry_score'] = c['Daily_Commute_km'] - 5 * c['Charging_Stations_Near_Home'] - 5 * c['Charging_Stations_Near_Work'] - 150 * (c['Home_Charging_Possible'] == 'Yes')
    c['buy_score'] = 1.2 * c['Annual_Income_USD'] / 1e5 + 0.6 * c['Environmental_Concern_Level'] + 2.0 * (c['Subsidy_Available'] == 'Yes')                      - 1.0 * (c['Range_Anxiety_Level'] == 'Medium') - 3.0 * (c['Range_Anxiety_Level'] == 'High')
all_cats += SMOOTH
for k in all_cats:
    c[f'{k}_fe'] = c[k].map(c[k].value_counts(normalize=True)).astype(float).fillna(0.0)
train = c[c.is_train == 1].drop(columns=['is_train']); test = c[c.is_train == 0].drop(columns=['is_train', T])
ev = [k for k in train.columns if k not in ['id', T] and pd.api.types.is_numeric_dtype(train[k])]
cm = train[ev].corr().abs(); up = cm.where(np.triu(np.ones(cm.shape), k=1).astype(bool))
drop = set(k for k in up.columns if any(up[k] == 1.0)) | set(k for k in train.columns if train[k].nunique() == 1) \
       | set(k for k in test.columns if test[k].nunique() == 1)
drop = [k for k in drop if k not in ['id', T]]
train = train.drop(columns=drop); test = test.drop(columns=drop)
FEAT = [k for k in test.columns if k != 'id']; TEC = [k for k in all_cats if k not in drop]
assert all(k in TEC for k in SMOOTH), 'smooth keys must survive the drop filter'
print('features %d, target-encoded cats %d (incl. 4 smooth keys), dropped %d' % (len(FEAT), len(TEC), len(drop)), flush=True)
X, y, Xt = train[FEAT], train[T].astype(int), test[FEAT]

P = dict(n_estimators=300 if SMOKE else 40000, learning_rate=0.005, max_depth=5, num_leaves=32, min_child_samples=10,
         subsample=0.8, colsample_bytree=0.3, reg_alpha=0.071, reg_lambda=2.0, max_bin=1024, random_state=SEED_M,
         feature_pre_filter=False, metric='auc', n_jobs=int(os.environ.get('NJOBS', -1)), verbose=-1)

# kospintr (LB 0.94603 single CatBoost) learners: depth 5, lr .025, max_bin 1152, balanced class weights; XGB depth 6.
P_CB = dict(loss_function='Logloss', eval_metric='AUC', learning_rate=0.025, depth=5, max_bin=16384 if VAR == 'cb16k' else 1152, l2_leaf_reg=3.0,
            auto_class_weights='Balanced', iterations=300 if SMOKE else 40000, task_type='GPU', verbose=0, random_seed=SEED_M)
if os.environ.get('CBP'):   # 09-15 Optuna winner override, json dict of CatBoost params (see experiments/optuna_cb.py); use with VAR=t
    import json; P_CB.update(json.loads(os.environ['CBP'])); P_CB.pop('auto_class_weights', None) if P_CB.get('auto_class_weights') in (None, 'None') else None
# najiama "XGBoost Triple-TE" (LB 0.94639) params, credited to Tilii: depth 4 lossguide 16 leaves, gamma 3.67, lr .01, ES 500. MODEL=xgbn
P_XGBN = dict(objective='binary:logistic', eval_metric='auc', tree_method='hist', device='cuda', learning_rate=0.01, max_depth=4,
              max_leaves=16, grow_policy='lossguide', gamma=3.673225596759869, min_child_weight=4.532387806880492, subsample=0.7400402414525654,
              colsample_bytree=0.5695776529558766, reg_alpha=0.7523885021652775, reg_lambda=0.6189949691705282, max_bin=1024,
              n_estimators=300 if SMOKE else 100000, random_state=SEED_M, n_jobs=int(os.environ.get('NJOBS', -1)))
P_XGB = dict(objective='binary:logistic', eval_metric='auc', tree_method='hist', device='cuda', learning_rate=0.01,
             max_depth=6, min_child_weight=30, subsample=0.8, colsample_bytree=0.5, reg_lambda=2.0, max_bin=1024,
             n_estimators=300 if SMOKE else 40000, random_state=SEED_M, n_jobs=int(os.environ.get('NJOBS', -1)))

def fit(Xa, ya, Xb=None, yb=None, rounds=None):
    """rounds=None -> early stop on the inner split (Xb, yb); rounds=int -> fixed length (full refit)."""
    if MODEL == 'lgb':
        m = lgb.LGBMClassifier(**dict(P, **({'n_estimators': rounds} if rounds else {}))).fit(
            Xa, ya, eval_set=None if rounds else [(Xb, yb)], callbacks=None if rounds else [lgb.early_stopping(700, verbose=False)])
        return m, rounds or m.best_iteration_
    if MODEL == 'cb':
        m = cb.CatBoostClassifier(**dict(P_CB, **({'iterations': rounds} if rounds else {}))).fit(
            Xa, ya, eval_set=None if rounds else (Xb, yb), early_stopping_rounds=None if rounds else 150)
        return m, rounds or m.get_best_iteration()
    p = dict(P_XGBN if MODEL == 'xgbn' else P_XGB, **({'n_estimators': rounds} if rounds else {'early_stopping_rounds': 500 if MODEL == 'xgbn' else 200}))
    m = xgb.XGBClassifier(**p).fit(Xa, ya, eval_set=None if rounds else [(Xb, yb)], verbose=False)
    return m, rounds or m.best_iteration

def encode(Xa, ya, others):
    """Triple TE fit on Xa (train rows cross-fitted), applied to each frame in others. V3 addition 2 = smooth 100."""
    encs = [('auto', TargetEncoder(shuffle=True, cv=5, smooth='auto', random_state=TESEED)),
            ('10', TargetEncoder(shuffle=True, cv=5, smooth=10.0, random_state=TESEED)),
            ('100', TargetEncoder(shuffle=True, cv=5, smooth=100.0, random_state=TESEED))]
    if VAR == 'ladder':   # 09-09 probe 4: sharper + flatter prior on the same keys
        encs += [('1', TargetEncoder(shuffle=True, cv=5, smooth=1.0, random_state=TESEED)),
                 ('1000', TargetEncoder(shuffle=True, cv=5, smooth=1000.0, random_state=TESEED))]
    if VAR == 'teavg':   # 09-09 probe 3: average each TE over 3 inner-fold splits -> less noise per 8-row income value
        reps = [[TargetEncoder(shuffle=True, cv=5, smooth=e.smooth, random_state=TESEED + r) for _, e in encs] for r in range(3)]
        A = [np.mean([rep[i].fit_transform(Xa[TEC], ya) for rep in reps], axis=0) for i in range(len(encs))]
        tr = lambda F: [np.mean([rep[i].transform(F[TEC]) for rep in reps], axis=0) for i in range(len(encs))]
    else:
        A = [e.fit_transform(Xa[TEC], ya) for _, e in encs]
        tr = lambda F: [e.transform(F[TEC]) for _, e in encs]
    loc = local_rate(Xa, ya, others) if VAR == 'local' else None
    def build(F, mats, j):
        F = F.drop(columns=TEC).copy()
        for (lab, _), M in zip(encs, mats):
            for i, k in enumerate(TEC): F[f'{k}_TE_{lab}'] = M[:, i].astype('float32')
        if loc is not None:
            for k, v in loc[j].items(): F[k] = v
        return F
    return build(Xa, A, 0), [build(F, tr(F), j + 1) for j, F in enumerate(others)]

# 09-09 probe 1: senihbayankulu cross-fitted local rate P(y | x within +-h), prior weight 20; +0.00013 on a naji-style LGBM in his hands
BW = {'Annual_Income_USD': [8., 20., 60., 200., 800., 3000., 12000.], 'Daily_Commute_km': [0.1, 0.4, 1.5, 6.0], 'Age': [0.5, 2.5]}
def _lfit(v, yy):
    o = np.argsort(v, kind='mergesort'); return v[o], np.concatenate([[0.], np.cumsum(yy[o].astype(np.float64))])
def _lapply(st, q, h, g, w=20.0):
    v, cs = st; lo = np.searchsorted(v, q - h, 'left'); hi = np.searchsorted(v, q + h, 'right')
    return ((cs[hi] - cs[lo] + w * g) / ((hi - lo) + w)).astype('float32')
def local_rate(Xa, ya, others):
    """returns [dict for Xa (inner 5-fold cross-fit), dict for each of others (fit on all of Xa)]"""
    ya = np.asarray(ya); g = float(ya.mean()); out = [{} for _ in range(1 + len(others))]
    for c, hs in BW.items():
        va = Xa[c].to_numpy(float)
        for h in hs: out[0][f'loc_{c}_{h}'] = np.zeros(len(Xa), 'float32')
        for a, b in StratifiedKFold(5, shuffle=True, random_state=TESEED).split(va, ya):
            st = _lfit(va[a], ya[a]); gi = float(ya[a].mean())
            for h in hs: out[0][f'loc_{c}_{h}'][b] = _lapply(st, va[b], h, gi)
        st = _lfit(va, ya)
        for j, F in enumerate(others):
            for h in hs: out[j + 1][f'loc_{c}_{h}'] = _lapply(st, F[c].to_numpy(float), h, g)
    return out

f_oof = OOF / f'oof_{TAG}.npy'
done = (not SMOKE) and f_oof.exists() and (np.load(f_oof) == 0).mean() < 0.01
t0 = time.time(); oof, tp, iters = np.zeros(len(X)), np.zeros(len(Xt)), []
if done:
    iters = list(np.load(OOF / f'iters_{TAG}.npy')); print('%s CV already complete, skip' % TAG, flush=True)
folds = [] if done else list(StratifiedKFold(NFOLDS, shuffle=True, random_state=SEED).split(X, y))
for f, (itr, iva) in enumerate(folds[:1] if SMOKE else folds):
    Xtr, ytr = X.iloc[itr], y.iloc[itr]
    ia, ib = train_test_split(np.arange(len(itr)), test_size=0.1, stratify=ytr, random_state=SEED_M)  # inner ES split
    Xa, (Xb, Xva, Xte_f) = encode(Xtr.iloc[ia], ytr.iloc[ia], [Xtr.iloc[ib], X.iloc[iva], Xt])
    assert f'{SMOOTH[0]}_TE_100' in Xa.columns and Xa.shape[1] == len(FEAT) - len(TEC) + (5 if VAR == 'ladder' else 3) * len(TEC) + (13 if VAR == 'local' else 0)
    m, it = fit(Xa, ytr.iloc[ia], Xb, ytr.iloc[ib])
    oof[iva] = m.predict_proba(Xva)[:, 1]; tp += m.predict_proba(Xte_f)[:, 1] / NFOLDS; iters.append(it)
    if not SMOKE:
        np.save(OOF / f'oof_{TAG}.npy', oof); np.save(TEST / f'test_{TAG}.npy', tp); np.save(OOF / f'iters_{TAG}.npy', np.array(iters))
    print('%s fold%d=%.5f iters=%d (%.0fs)' % (TAG, f, roc_auc_score(y.iloc[iva], oof[iva]), iters[-1], time.time() - t0), flush=True)
if SMOKE:
    assert roc_auc_score(y.iloc[iva], oof[iva]) > 0.93, 'smoke AUC too low'; print('SMOKE OK'); sys.exit(0)
if not done: print('%s oof=%.5f  mean_iters=%d' % (TAG, roc_auc_score(y, oof), np.mean(iters)), flush=True)
t1 = time.time(); Xa, (Xte_f,) = encode(X, y, [Xt])
m, _ = fit(Xa, y, rounds=int(np.mean(iters) * 1.1))
np.save(TEST / f'test_{TAG}_full.npy', m.predict_proba(Xte_f)[:, 1])
print('%s full refit done rounds=%d (%.0fs)' % (TAG, int(np.mean(iters) * 1.1), time.time() - t1), flush=True)
