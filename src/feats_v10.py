"""Port of the public 0.94608 pipeline's feature matrix (lamhuy8904 'Version 10'), made leak-free:
target encodings are fitted on the TRAIN part of each outer fold only (nested), never on the val rows.
build(tr, te) -> (Xtr_df, Xte_df, te_specs)   where te_specs drive add_te() per fold.
"""
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold

CATS = ['Gender', 'City_Type', 'Current_Car_Type', 'Home_Charging_Possible', 'Subsidy_Available', 'Range_Anxiety_Level']
TE_NUM = [('te_inc_raw', 'Annual_Income_USD', 1), ('te_inc_100', 'Annual_Income_USD', 100),
          ('te_inc_1000', 'Annual_Income_USD', 1000), ('te_km', 'Daily_Commute_km', 1)]
JOINT = {'joint_env_subsidy': ['Environmental_Concern_Level', 'Subsidy_Available'],
         'joint_env_anxiety': ['Environmental_Concern_Level', 'Range_Anxiety_Level'],
         'joint_home_city': ['Home_Charging_Possible', 'City_Type'],
         'joint_inc_env': ['inc_div_1000', 'Environmental_Concern_Level'],
         'joint_km_anxiety': ['km_div_10', 'Range_Anxiety_Level']}


def _tz(a):
    z = np.zeros(len(a), dtype=np.int8)
    for k, m in enumerate([10, 100, 1000, 10000], start=1):
        z[a % m == 0] = k
    return z


def build(tr, te):
    """Static features (no target). Returns frames with model columns + joint-key columns (strings)."""
    out = []
    for df in (tr, te):
        d = pd.DataFrame(index=df.index)
        for c in CATS:
            d['LE_' + c] = df[c].cat.codes.astype('int8')
        for c in ['Age', 'Annual_Income_USD', 'Daily_Commute_km', 'Charging_Stations_Near_Home',
                  'Charging_Stations_Near_Work', 'Environmental_Concern_Level']:
            d[c] = df[c].astype('float32')
        inc = df['Annual_Income_USD'].to_numpy(np.int64)
        km = np.round(df['Daily_Commute_km'].to_numpy(float) * 10).astype(np.int64)
        d['inc_at_floor'] = (inc == 30000); d['inc_round_100'] = (inc % 100 == 0); d['inc_round_1000'] = (inc % 1000 == 0)
        d['inc_tz'] = _tz(inc); d['inc_last_digit'] = inc % 10; d['inc_mod_100'] = inc % 100; d['inc_mod_1000'] = inc % 1000
        d['km_at_floor'] = (km == 50); d['km_round_1'] = (km % 10 == 0); d['km_last_digit'] = km % 10; d['km_mod_100'] = km % 100
        for dv in [100, 250, 500, 1000, 2000, 5000]:
            d[f'inc_div_{dv}'] = inc // dv
        for dv in [5, 10, 25, 50]:
            d[f'km_div_{dv}'] = km // dv
        sub, ra = d['LE_Subsidy_Available'], d['LE_Range_Anxiety_Level']
        d['Env_x_Subsidy'] = d['Environmental_Concern_Level'] * sub
        d['Env_x_Income'] = d['Environmental_Concern_Level'] * d['Annual_Income_USD']
        d['Subsidy_x_Income'] = sub * d['Annual_Income_USD']
        d['RA_x_Commute'] = ra * d['Daily_Commute_km']
        d['Charging_Total'] = d['Charging_Stations_Near_Home'] + d['Charging_Stations_Near_Work']
        d['Charging_Total_x_Commute'] = d['Charging_Total'] * d['Daily_Commute_km']
        for name, (a, b) in JOINT.items():  # string keys, used only for TE
            d[name] = df[a].astype(str).values + '_' + df[b].astype(str).values if a in df else \
                      d[a].astype(str).values + '_' + df[b].astype(str).values
        for c in d.columns:
            if d[c].dtype == bool: d[c] = d[c].astype('int8')
        out.append(d)
    return out[0], out[1]


def _smooth(key_tr, y_tr, key_apply, prior, m):
    s = pd.DataFrame({'k': key_tr, 'y': y_tr}).groupby('k').y.agg(['sum', 'count'])
    rate = (s['sum'] + m * prior) / (s['count'] + m)
    return pd.Series(key_apply).map(rate).fillna(prior).values.astype('float32')


def _keys(d):
    ks = {n: (d[c].to_numpy(np.int64) // dv if dv > 1 else d[c].to_numpy(np.int64)) for n, c, dv in TE_NUM}
    ks.update({'te_' + j: d[j].values for j in JOINT})
    return ks


def add_te(Dtr, ytr, Dapply_list, seed=42):
    """Fit TE on (Dtr, ytr). Train rows get inner-5-fold OOF values; each frame in Dapply_list gets
    full-fit values. Returns (Xtr, [Xapply...]) with joint string columns dropped."""
    prior = float(np.mean(ytr)); ytr = np.asarray(ytr)
    ktr = _keys(Dtr)
    Xtr = Dtr.drop(columns=list(JOINT)).copy()
    for n in ktr:
        m = 100.0 if n.startswith('te_inc') or n == 'te_km' else 50.0
        col = np.zeros(len(Dtr), dtype='float32')
        for itr, iva in StratifiedKFold(5, shuffle=True, random_state=seed).split(Dtr, ytr):
            col[iva] = _smooth(ktr[n][itr], ytr[itr], ktr[n][iva], prior, m)
        Xtr[n] = col
    outs = []
    for D in Dapply_list:
        X = D.drop(columns=list(JOINT)).copy(); ka = _keys(D)
        for n in ktr:
            m = 100.0 if n.startswith('te_inc') or n == 'te_km' else 50.0
            X[n] = _smooth(ktr[n], ytr, ka[n], prior, m)
        outs.append(X)
    return Xtr, outs
