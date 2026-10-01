"""Final additive LGBM: slow lr, fixed tree count (no ES on val -> clean OOF). Refit full -> sub."""
import time, numpy as np, pandas as pd, lightgbm as lgb
from cv import load, run_cv, paired, SEED, OOF, TEST, SUBS
from feats import additive

X, y, Xte, te_id = load()
# fold0 ES at lr .02 plateaued at 8092 trees -> scale to lr .006
LR, N = 0.006, 27000
P = dict(n_estimators=N, learning_rate=LR, num_leaves=31, max_bin=13214,
         interaction_constraints=additive(X.shape[1]), verbose=-1, random_state=SEED)

t = time.time()
oof, s = run_cv(lambda: lgb.LGBMClassifier(**P), X, y, 'FINAL lr%.3f n%d' % (LR, N))
paired(oof, np.load(OOF / 'oof_C_additive.npy'), y, 'vs C (0.94525)')
paired(oof, np.load(OOF / 'oof_F.npy'), y, 'vs F (0.94540)')
print('   cv %.0fs' % (time.time() - t), flush=True)
np.save(OOF / 'oof_final.npy', oof)

t = time.time()
m = lgb.LGBMClassifier(**dict(P, n_estimators=int(N * 1.1))).fit(X, y)   # full data -> ~10% more trees
p = m.predict_proba(Xte)[:, 1]
np.save(TEST / 'test_final.npy', p)
pd.DataFrame({'id': te_id, 'Will_Buy_EV': p}).to_csv(SUBS / 'sub_final_lgb.csv', index=False)
print('wrote sub_final_lgb.csv  mean=%.5f  refit %.0fs' % (p.mean(), time.time() - t), flush=True)
