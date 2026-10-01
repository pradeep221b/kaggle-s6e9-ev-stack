"""Verify-then-gate for ridge (Paul Bryan Elefante) 'KPS6E09 Logistic Regression Sample' published OOF/TEST
(10-fold, LB 0.94637). Read-only mirror of stack_check_naji3.py: same members + one extra column 'ridge'.
Prints single AUC on our y, corr with our streams, paired gate of stack(+ridge) vs stack(without).
--write: saves outputs/oof/oof_ridge_s42.npy + test_ridge_s42_full.npy and submissions/sub_ridge_stack_<cv>.csv"""
import sys, re, glob, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from scipy.special import logit, expit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from cv import load, paired, OOF, TEST, SUBS, SEED, NFOLDS

X, y, Xte, te_id = load()
L = lambda p: logit(np.clip(p, 1e-6, 1 - 1e-6))
tr_id = pd.read_csv('data/train.csv', usecols=['id'])['id'].values
ho = pd.read_parquet(glob.glob('research/heuljax_ridge/out/oof/*.parquet')[0]).set_index('id')
ht = pd.read_parquet(glob.glob('research/heuljax_ridge/out/test_preds/*.parquet')[0]).set_index('id')
pcol = [c for c in ht.columns if 'pred' in c][0]
h_oof = ho.loc[tr_id, 'oof_pred'].values; h_test = ht.loc[te_id, pcol].values
assert np.array_equal(ho.loc[tr_id, 'y_true'].values, y.values), 'y mismatch'
print('ridge single oof=%.5f (author pooled 0.94640, 10-fold)' % roc_auc_score(y, h_oof))

OK = r'(lgb|xgb|cb|tabm|ftt|realmlp|tabr|autogluon|naji|najit58|zoom|direct|naji3[a-z0-9]*|sergey|yeke|lucxgb|heuljax|aryan)_s\d+'
pairs = {'gbdt_additive': ('oof_final.npy', 'test_final.npy'), 'realmlp_e48x3': ('oof_R_e48.npy', 'test_nn_avg.npy')}
for f in sorted(OOF.glob('oof_*_s*.npy')):
    tag = f.stem[4:]
    if re.fullmatch(OK, tag) and (TEST / f'test_{tag}.npy').exists():
        full = TEST / f'test_{tag}_full.npy'
        pairs[tag] = (f.name, full.name if full.exists() else f'test_{tag}.npy')
oofs, tests = {}, {}
for k, (fo, ft) in pairs.items():
    o = np.load(OOF / fo)
    if (o == 0).mean() > 0.01: continue
    oofs[k], tests[k] = o, np.load(TEST / ft)
fam = {}
for n in list(oofs):
    m = re.fullmatch(r'([a-z0-9]+)_s\d+', n); fam.setdefault(m.group(1) if m else n, []).append(n)
for f, names in fam.items():
    if len(names) > 1:
        oofs[f + '_bag%d' % len(names)] = expit(np.mean([L(oofs[n]) for n in names], axis=0))
        tests[f + '_bag%d' % len(names)] = expit(np.mean([L(tests[n]) for n in names], axis=0))
        for n in names: oofs.pop(n); tests.pop(n)
for k in oofs:
    if k.startswith(('naji3cb', 'naji3_', 'gbdt', 'naji3xgbn', 'zoom')):
        print('  corr(ridge, %s)=%.4f  oof=%.5f' % (k, np.corrcoef(h_oof, oofs[k])[0, 1], roc_auc_score(y, oofs[k])))
paired(h_oof, oofs['gbdt_additive'], y, 'ridge single vs additive tree')

def fit_stack(keys):
    Z = np.column_stack([L(oofs[k]) for k in keys]); Zt = np.column_stack([L(tests[k]) for k in keys])
    s = np.zeros(len(y))
    for itr, iva in StratifiedKFold(NFOLDS, shuffle=True, random_state=SEED).split(Z, y):
        s[iva] = LogisticRegression(C=0.1, max_iter=2000).fit(Z[itr], y.iloc[itr]).predict_proba(Z[iva])[:, 1]
    m = LogisticRegression(C=0.1, max_iter=2000).fit(Z, y)
    return s, m.predict_proba(Zt)[:, 1], dict(zip(keys, np.round(m.coef_[0], 3)))

base = list(oofs); s0, p0, w0 = fit_stack(base)
print('stack WITHOUT ridge oof=%.5f (%d members)' % (roc_auc_score(y, s0), len(base)))
oofs['ridge'], tests['ridge'] = h_oof, h_test
s1, p1, w1 = fit_stack(base + ['ridge'])
print('stack WITH ridge oof=%.5f  weight ridge=%.3f' % (roc_auc_score(y, s1), w1['ridge']))
print('weights:', w1)
d = paired(s1, s0, y, 'stack(+ridge) vs stack(without)')
ok = d.mean() > 2 * d.std(ddof=1) and (d > 0).all()
print('GATE', 'PASS' if ok else 'FAIL', '(paired mean %.5f, sd %.5f, folds>0 %d/5)' % (d.mean(), d.std(ddof=1), (d > 0).sum()))
if ok and '--write' in sys.argv:
    np.save(OOF / 'oof_ridge_s42.npy', h_oof); np.save(TEST / 'test_ridge_s42_full.npy', h_test); np.save(TEST / 'test_ridge_s42.npy', h_test)
    out = SUBS / ('sub_ridge_stack_cv%05d.csv' % round(roc_auc_score(y, s1) * 1e5))
    pd.DataFrame({'id': te_id, 'Will_Buy_EV': p1}).to_csv(out, index=False); print('wrote', out.name)
