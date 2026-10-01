"""Final assembly: every base model with an (oof, test) pair -> logit stacker, cross-fitted on the
fixed folds, gated by paired() against the previous final (0.75 additive GBDT + 0.25 RealMLP rank blend).
Discovers streams automatically: outputs/oof/oof_<tag>.npy + outputs/test/test_<tag>.npy;
prefers test_<tag>_full.npy (full-data refit) when present. Incomplete OOFs (crashed runs) are skipped.
Writes submissions/sub_stack.csv only if the stack passes. `--push` submits.
"""
import sys, re, numpy as np, pandas as pd
from scipy.stats import rankdata
from scipy.special import logit, expit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from cv import load, paired, OOF, TEST, SUBS, SEED, NFOLDS

X, y, Xte, te_id = load()
rk = lambda a: rankdata(a) / len(a)
L = lambda p: logit(np.clip(p, 1e-6, 1 - 1e-6))

# The two legacy streams have irregular names; everything else follows oof_<model>_s<seed>.npy.
LEGACY = {'gbdt_additive': ('oof_final.npy', 'test_final.npy'), 'realmlp_e48x3': ('oof_R_e48.npy', 'test_nn_avg.npy')}
pairs = dict(LEGACY)
for f in sorted(OOF.glob('oof_*_s*.npy')):
    tag = f.name[4:-4]
    if re.fullmatch(r'(lgb|xgb|cb|tabm|ftt|realmlp|tabr|autogluon|naji|najit58|zoom|direct)_s\d+', tag) and (TEST / ('test_%s.npy' % tag)).exists():
        pairs[tag] = (f.name, 'test_%s.npy' % tag)

oofs, tests = {}, {}
for n, (fo, ft) in pairs.items():
    o = np.load(OOF / fo)
    if (o == 0).mean() > 0.01: print('skip %s (incomplete oof)' % n); continue
    full = TEST / ft.replace('.npy', '_full.npy')
    oofs[n], tests[n] = o, np.load(full if full.exists() else TEST / ft)
    print('  %-16s oof=%.5f  corr_additive=%.4f  test=%s' % (n, roc_auc_score(y, o),
          np.corrcoef(o, oofs['gbdt_additive'])[0, 1], 'FULL refit' if full.exists() else 'fold-avg'))

# Seed-bag: average all seeds of one model family in logit space -> one stream per family.
# Near-duplicate seed columns otherwise let the stacker fit noise (+/- weights on the same model).
fam = {}
for n in list(oofs):
    m = re.fullmatch(r'([a-z]+)_s\d+', n)
    fam.setdefault(m.group(1) if m else n, []).append(n)
for f, names in fam.items():
    if len(names) > 1:
        oofs[f + '_bag%d' % len(names)] = expit(np.mean([L(oofs[n]) for n in names], axis=0))
        tests[f + '_bag%d' % len(names)] = expit(np.mean([L(tests[n]) for n in names], axis=0))
        for n in names: oofs.pop(n); tests.pop(n)
        print('  bagged %d seeds -> %-12s oof=%.5f' % (len(names), f, roc_auc_score(y, oofs[f + '_bag%d' % len(names)])))

BASE = 0.75 * rk(oofs['gbdt_additive']) + 0.25 * rk(oofs['realmlp_e48x3'])
print('previous final blend oof=%.5f   (%d base models)' % (roc_auc_score(y, BASE), len(oofs)))
paired(np.mean([rk(o) for o in oofs.values()], axis=0), BASE, y, 'equal rank avg all')

Z = np.column_stack([L(o) for o in oofs.values()]); Zt = np.column_stack([L(t) for t in tests.values()])
stack = np.zeros(len(y))
for itr, iva in StratifiedKFold(NFOLDS, shuffle=True, random_state=SEED).split(Z, y):
    stack[iva] = LogisticRegression(C=0.1, max_iter=2000).fit(Z[itr], y.iloc[itr]).predict_proba(Z[iva])[:, 1]
paired(stack, BASE, y, 'logit stack vs 09-03 blend')
# Incumbent gate: the best previously FROZEN stack OOF (outputs/oof/stacks/stack_*.npy). A new stack must beat the
# incumbent, not the obsolete 09-03 blend -- otherwise a regression could PASS and be submitted (Codex judgment 09-06).
STACKS = OOF / 'stacks'; STACKS.mkdir(exist_ok=True)
prev = sorted(STACKS.glob('stack_*.npy'), key=lambda f: roc_auc_score(y, np.load(f)))
INC = np.load(prev[-1]) if prev else BASE
print('incumbent: %s oof=%.5f' % (prev[-1].name if prev else '09-03 blend', roc_auc_score(y, INC)))
d = paired(stack, INC, y, 'logit stack vs incumbent')
m = LogisticRegression(C=0.1, max_iter=2000).fit(Z, y)
print('stack weights:', {k: round(float(w), 3) for k, w in zip(oofs, m.coef_[0])})

if d.mean() > 2 * d.std(ddof=1) and (d > 0).all():
    p = m.predict_proba(Zt)[:, 1]
    cv = roc_auc_score(y, stack); tag = 'cv%05d_%s' % (round(cv * 1e5), pd.Timestamp.now().strftime('%m%d_%H%M'))
    OUT = SUBS / ('sub_stack_%s.csv' % tag)          # versioned; never overwrite a submitted file
    pd.DataFrame({'id': te_id, 'Will_Buy_EV': p}).to_csv(OUT, index=False)
    np.save(STACKS / ('stack_%s.npy' % tag), stack)   # freeze the OOF -> becomes the next incumbent
    pd.Series({k: float(w) for k, w in zip(oofs, m.coef_[0])}).to_json(STACKS / ('stack_%s_weights.json' % tag))
    print('PASS -> wrote %s  (oof %.5f vs incumbent %.5f)' % (OUT.name, cv, roc_auc_score(y, INC)))
    if '--push' in sys.argv:
        import kaggle
        kaggle.api.competition_submit(str(OUT), 'logit stack of %d models, oof %.5f' % (len(oofs), roc_auc_score(y, stack)),
                                      'playground-series-s6e9'); print('submitted')
else:
    print('FAIL -> stack does not beat the incumbent; nothing written')
