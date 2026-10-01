"""Honest port of jazivxt/single-model-zoom-zoom (public LB 0.94607 single LGBM, 9 min): raw + digits + frequency
+ recipe features + dual TE on exact keys + income grid stats (16384 bins, left/right/kernel smoothing). Only change:
early stopping on an inner 10% split. Our folds (seed 42). Usage: py experiments/zoom_stream.py [seed]"""
import hashlib
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder

from lightgbm import LGBMClassifier
import lightgbm as lgb
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

MODEL = "lightgbm"       # "lightgbm", "xgboost", or "catboost"
N_FOLDS = 5
OUTER_SEED = 42
INNER_SEED = 17
MODEL_SEED = 60
USE_FABLE_FEATURES = True
FOLDS_TO_RUN = None
VERBOSE_EVAL = 0

# Change this one path. Keep the trailing slash.
DATA_PATH = str(__import__("pathlib").Path(__file__).resolve().parents[1] / "data") + "/"
TRAIN_PATH = DATA_PATH + "train.csv"
TEST_PATH = DATA_PATH + "test.csv"

TARGET = "Will_Buy_EV"
ID = "id"
CONTINUOUS_COLS = [
    "Age", "Annual_Income_USD", "Daily_Commute_km",
    "Charging_Stations_Near_Home", "Charging_Stations_Near_Work",
    "Environmental_Concern_Level",
]
RAW_COLS = [
    "Age", "Annual_Income_USD", "Daily_Commute_km", "Number_of_Cars_Owned",
    "Charging_Stations_Near_Home", "Charging_Stations_Near_Work",
    "Environmental_Concern_Level", "Gender", "City_Type", "Current_Car_Type",
    "Home_Charging_Possible", "Subsidy_Available", "Range_Anxiety_Level",
]

def preprocess(frame):
    frame = frame.copy()
    mappings = {
        "Gender": {"Male": 0, "Female": 1, "Other": 2},
        "City_Type": {"Urban": 0, "Suburban": 1, "Rural": 2},
        "Current_Car_Type": {"Sedan": 0, "SUV": 1, "Hatchback": 2, "Truck": 3},
        "Home_Charging_Possible": {"Yes": 0, "No": 1},
        "Subsidy_Available": {"Yes": 0, "No": 1},
        "Range_Anxiety_Level": {"Low": 0, "Medium": 1, "High": 2},
    }
    for column, mapping in mappings.items():
        if not pd.api.types.is_numeric_dtype(frame[column]):
            frame[column] = frame[column].map(mapping)
    if TARGET in frame and not pd.api.types.is_numeric_dtype(frame[TARGET]):
        frame[TARGET] = frame[TARGET].map({"No": 0, "Yes": 1})
    return frame

train = preprocess(pd.read_csv(TRAIN_PATH))
test = preprocess(pd.read_csv(TEST_PATH))
y = train[TARGET].to_numpy(np.uint8)
train_ids = train[ID].to_numpy(np.int64)
test_ids = test[ID].to_numpy(np.int64)


# %% ---- cell ----
def target_free_features(train, test):
    tr_raw, te_raw = train[RAW_COLS].copy(), test[RAW_COLS].copy()
    tr_blocks = [tr_raw.to_numpy(np.float32)]
    te_blocks = [te_raw.to_numpy(np.float32)]
    names = list(RAW_COLS)

    for column in RAW_COLS:
        tr_v = tr_raw[column].fillna(0).to_numpy(np.float64)
        te_v = te_raw[column].fillna(0).to_numpy(np.float64)
        tr_digits, te_digits = [], []
        for power in range(-4, 4):
            scale = 10.0 ** power
            tr_digits.append(np.floor_divide(tr_v, scale) % 10)
            te_digits.append(np.floor_divide(te_v, scale) % 10)
            names.append(f"{column}__digit_{power}")
        tr_blocks.append(np.column_stack(tr_digits).astype(np.float32))
        te_blocks.append(np.column_stack(te_digits).astype(np.float32))

    tr_keys = tr_raw.astype("object").fillna("__NA__").astype(str)
    te_keys = te_raw.astype("object").fillna("__NA__").astype(str)
    tr_freq, te_freq = [], []
    for column in RAW_COLS:
        frequency = pd.concat([tr_keys[column], te_keys[column]], ignore_index=True).value_counts(normalize=True)
        tr_freq.append(tr_keys[column].map(frequency).to_numpy(np.float32))
        te_freq.append(te_keys[column].map(frequency).to_numpy(np.float32))
        names.append(f"{column}__frequency")
    tr_blocks.append(np.column_stack(tr_freq).astype(np.float32))
    te_blocks.append(np.column_stack(te_freq).astype(np.float32))

    return np.column_stack(tr_blocks).astype(np.float32), np.column_stack(te_blocks).astype(np.float32), tr_keys, te_keys, names


def add_parent_digit_channels(train, test, tr_keys, te_keys):
    tr_freq, te_freq, names = [], [], []
    tr_keys, te_keys = tr_keys.copy(), te_keys.copy()
    for column in CONTINUOUS_COLS:
        tr_v = train[column].fillna(0).to_numpy(np.float64)
        te_v = test[column].fillna(0).to_numpy(np.float64)
        for power in range(-4, 4):
            scale = 10.0 ** power
            key_name = f"{column}_digit{power}"
            tr_key = pd.Series((np.floor_divide(tr_v, scale) % 10).astype(np.int8)).astype(str)
            te_key = pd.Series((np.floor_divide(te_v, scale) % 10).astype(np.int8)).astype(str)
            tr_keys[key_name], te_keys[key_name] = tr_key.to_numpy(), te_key.to_numpy()
            frequency = pd.concat([tr_key, te_key], ignore_index=True).value_counts(normalize=True)
            tr_freq.append(tr_key.map(frequency).to_numpy(np.float32))
            te_freq.append(te_key.map(frequency).to_numpy(np.float32))
            names.append(f"{key_name}__frequency")
    return np.column_stack(tr_freq), np.column_stack(te_freq), tr_keys, te_keys, names


def fable_features(frame):
    home_yes = (frame["Home_Charging_Possible"].to_numpy() == 0).astype(np.float64)
    subsidy_yes = (frame["Subsidy_Available"].to_numpy() == 0).astype(np.float64)
    income = frame["Annual_Income_USD"].to_numpy(np.float64)
    concern = frame["Environmental_Concern_Level"].to_numpy(np.float64)
    anxiety = frame["Range_Anxiety_Level"].to_numpy()
    recipe = 1.2 * income / 1e5 + 0.6 * concern + 2.0 * subsidy_yes - (anxiety == 1) - 3.0 * (anxiety == 2)
    values = np.column_stack([
        frame["Daily_Commute_km"].to_numpy(np.float64)
        - 5.0 * frame["Charging_Stations_Near_Home"].to_numpy(np.float64)
        - 5.0 * frame["Charging_Stations_Near_Work"].to_numpy(np.float64)
        - 150.0 * home_yes,
        frame["Charging_Stations_Near_Home"].to_numpy(np.float64) + frame["Charging_Stations_Near_Work"].to_numpy(np.float64),
        income / 1e5 * subsidy_yes,
        concern * subsidy_yes,
        recipe,
    ]).astype(np.float32)
    names = ["fable_worry", "fable_chargers_total", "fable_income_x_subsidy", "fable_concern_x_subsidy", "fable_recipe"]
    return values, names

# %% ---- cell ----
def drop_duplicate_arrays(tr, te, names):
    seen, keep, duplicates = {}, [], []
    for j, name in enumerate(names):
        digest = hashlib.sha256()
        digest.update(np.ascontiguousarray(tr[:, j]).tobytes())
        digest.update(np.ascontiguousarray(te[:, j]).tobytes())
        key = digest.digest()
        prior = seen.get(key)
        if prior is not None and np.array_equal(tr[:, j], tr[:, prior], equal_nan=True) and np.array_equal(te[:, j], te[:, prior], equal_nan=True):
            duplicates.append({"dropped": name, "kept": names[prior]})
        else:
            seen[key] = j
            keep.append(j)
    return tr[:, keep], te[:, keep], [names[j] for j in keep], duplicates


def drop_duplicate_keys(tr_keys, te_keys):
    seen, keep, duplicates = {}, [], []
    for column in tr_keys.columns:
        digest = hashlib.sha256()
        digest.update(pd.util.hash_pandas_object(tr_keys[column], index=False).to_numpy().tobytes())
        digest.update(pd.util.hash_pandas_object(te_keys[column], index=False).to_numpy().tobytes())
        key = digest.digest()
        prior = seen.get(key)
        if prior is not None and tr_keys[column].equals(tr_keys[prior]) and te_keys[column].equals(te_keys[prior]):
            duplicates.append({"dropped": column, "kept": prior})
        else:
            seen[key] = column
            keep.append(column)
    return tr_keys[keep].copy(), te_keys[keep].copy(), duplicates


base_train, base_test, train_keys, test_keys, feature_names = target_free_features(train, test)
digit_train, digit_test, train_keys, test_keys, digit_names = add_parent_digit_channels(train, test, train_keys, test_keys)
base_train = np.column_stack([base_train, digit_train]).astype(np.float32)
base_test = np.column_stack([base_test, digit_test]).astype(np.float32)
feature_names += digit_names

if USE_FABLE_FEATURES:
    f_tr, f_names = fable_features(train)
    f_te, _ = fable_features(test)
    base_train = np.column_stack([base_train, f_tr]).astype(np.float32)
    base_test = np.column_stack([base_test, f_te]).astype(np.float32)
    feature_names += f_names

before_numeric, before_keys = base_train.shape[1], train_keys.shape[1]
base_train, base_test, feature_names, numeric_duplicates = drop_duplicate_arrays(base_train, base_test, feature_names)
train_keys, test_keys, key_duplicates = drop_duplicate_keys(train_keys, test_keys)

duplicate_summary = {
    "numeric_or_frequency_removed": len(numeric_duplicates),
    "exact_key_sets_removed": len(key_duplicates),
    "modeled_columns_removed": len(numeric_duplicates) + 2 * len(key_duplicates),
    "base_before": before_numeric,
    "base_after": base_train.shape[1],
    "keys_before": before_keys,
    "keys_after": train_keys.shape[1],
}

# %% ---- cell ----
def bin_statistics(codes, y_fit, n_bins, prior, smooth):
    sums = np.bincount(codes, weights=y_fit, minlength=n_bins).astype(np.float64)
    counts = np.bincount(codes, minlength=n_bins).astype(np.float64)
    central = (sums + smooth * prior) / (counts + smooth)
    left_sums, left_counts = np.r_[0.0, sums[:-1]], np.r_[0.0, counts[:-1]]
    right_sums, right_counts = np.r_[sums[1:], 0.0], np.r_[counts[1:], 0.0]
    left = (left_sums + smooth * prior) / (left_counts + smooth)
    right = (right_sums + smooth * prior) / (right_counts + smooth)
    kernel = np.exp(-0.5 * (np.arange(-1, 2) / 0.8) ** 2)
    neighbor_sums = np.convolve(sums, kernel, mode="same")
    neighbor_counts = np.convolve(counts, kernel, mode="same")
    symmetric = (neighbor_sums + smooth * kernel.sum() * prior) / (neighbor_counts + smooth * kernel.sum())
    slope = right - left
    curvature = central - 0.5 * (left + right)
    return np.column_stack([central, symmetric, left, right, slope, curvature, np.log1p(counts)]).astype(np.float32)


def income_asymmetric_features(fit_income, fit_y, valid_income, test_income, q=16384, smooth=10.0, seed=17):
    fit_income = np.asarray(fit_income, np.float64)
    valid_income = np.asarray(valid_income, np.float64)
    test_income = np.asarray(test_income, np.float64)
    fit_y = np.asarray(fit_y, np.uint8)
    prior = float(fit_y.mean())
    edges = np.linspace(float(fit_income.min()), float(fit_income.max()), q + 1)
    fit_codes = np.searchsorted(edges[1:-1], fit_income)
    valid_codes = np.searchsorted(edges[1:-1], valid_income)
    test_codes = np.searchsorted(edges[1:-1], test_income)
    n_bins = len(edges)
    position_scale = max(n_bins - 2, 1)
    fit_features = np.zeros((len(fit_income), 8), np.float32)
    inner = StratifiedKFold(5, shuffle=True, random_state=seed)
    for inner_fit, inner_valid in inner.split(fit_codes, fit_y):
        stats = bin_statistics(fit_codes[inner_fit], fit_y[inner_fit], n_bins, prior, smooth)
        fit_features[inner_valid, 0] = fit_codes[inner_valid] / position_scale
        fit_features[inner_valid, 1:] = stats[fit_codes[inner_valid]]
    stats = bin_statistics(fit_codes, fit_y, n_bins, prior, smooth)
    valid_features = np.column_stack([valid_codes / position_scale, stats[valid_codes]]).astype(np.float32)
    test_features = np.column_stack([test_codes / position_scale, stats[test_codes]]).astype(np.float32)
    return fit_features, valid_features, test_features

# %% ---- cell ----
def make_model(name, fold):
    if name == "lightgbm":
        return LGBMClassifier(
            n_estimators=3500, learning_rate=0.02, max_depth=5, num_leaves=31,
            min_child_samples=10, subsample=0.812763123433567, subsample_freq=1,
            colsample_bytree=0.3029300829885024, reg_alpha=0.07094285437903122,
            reg_lambda=2.0330390977032425, max_bin=1024,
            random_state=MODEL_SEED, n_jobs=-1, verbosity=-1,
        )
    if name == "xgboost":
        return XGBClassifier(
            n_estimators=2400, max_depth=6, learning_rate=0.03,
            min_child_weight=12.0, subsample=0.82, colsample_bytree=0.55,
            reg_alpha=0.08, reg_lambda=3.0, max_bin=512,
            objective="binary:logistic", eval_metric="auc", tree_method="hist",
            early_stopping_rounds=120, random_state=MODEL_SEED + fold, n_jobs=-1,
        )
    return CatBoostClassifier(
        iterations=600, depth=6, learning_rate=0.07, l2_leaf_reg=5.0, rsm=0.8,
        loss_function="Logloss", eval_metric="AUC", random_seed=MODEL_SEED + fold,
        random_strength=0.35, bootstrap_type="Bayesian", bagging_temperature=0.45,
        od_type="Iter", od_wait=140, thread_count=-1, verbose=False,
        allow_writing_files=False,
    )


def fit_model(model, name, x_fit, y_fit, x_valid, y_valid):
    if name == "lightgbm":
        model.fit(x_fit, y_fit, eval_set=[(x_valid, y_valid)], eval_metric="auc",
                  callbacks=[lgb.early_stopping(120, verbose=True), lgb.log_evaluation(VERBOSE_EVAL)])
    elif name == "xgboost":
        model.fit(x_fit, y_fit, eval_set=[(x_valid, y_valid)], verbose=VERBOSE_EVAL)
    else:
        model.fit(x_fit, y_fit, eval_set=(x_valid, y_valid), use_best_model=True, verbose=VERBOSE_EVAL)
    return model


def best_iteration(model, name):
    if name == "lightgbm":
        return int(model.best_iteration_)
    if name == "xgboost":
        return int(getattr(model, "best_iteration", -1))
    return int(model.get_best_iteration())

# %% ---- cell ----
splits = list(StratifiedKFold(
    N_FOLDS, shuffle=True, random_state=OUTER_SEED
).split(base_train, y))

folds_to_run = list(range(N_FOLDS)) if FOLDS_TO_RUN is None else list(FOLDS_TO_RUN)
oof = np.full(len(train), np.nan, dtype=np.float32)
test_folds, fold_rows = [], []
started = time.time()

print(f"Model: {MODEL}")
print(f"Metric: ROC AUC | folds: {folds_to_run}")
print(f"Rows: {len(train):,} train / {len(test):,} test")
print(f"Base features: {base_train.shape[1]} | exact keys: {train_keys.shape[1]}")


# ---------------- HONEST fold loop (ours): early stopping on an inner 10% of the fit rows, never on the scoring fold ----
import sys as _sys
from sklearn.model_selection import train_test_split
_sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from cv import OOF, TEST
SEED_M = int(_sys.argv[1]) if len(_sys.argv) > 1 else 42
TAG = "zoom_s%d" % SEED_M
oof = np.zeros(len(train)); tpred = np.zeros(len(test)); iters = []
t0 = time.time()
for fold, (fit_idx, valid_idx) in enumerate(splits):
    ia, ib = train_test_split(fit_idx, test_size=0.1, stratify=y[fit_idx], random_state=SEED_M)
    enc_a, enc_b, enc_v, enc_t = [], [], [], []
    for smoothing in (10, "auto"):
        encoder = TargetEncoder(cv=5, smooth=smoothing, shuffle=True, random_state=OUTER_SEED, target_type="binary")
        enc_a.append(np.asarray(encoder.fit_transform(train_keys.iloc[ia], y[ia]), dtype=np.float32))
        enc_b.append(np.asarray(encoder.transform(train_keys.iloc[ib]), dtype=np.float32))
        enc_v.append(np.asarray(encoder.transform(train_keys.iloc[valid_idx]), dtype=np.float32))
        enc_t.append(np.asarray(encoder.transform(test_keys), dtype=np.float32))
    inc_all = train["Annual_Income_USD"].to_numpy()
    inc_a, inc_v, inc_t = income_asymmetric_features(inc_all[ia], y[ia], inc_all[valid_idx], test["Annual_Income_USD"].to_numpy(), seed=INNER_SEED)
    _, inc_b, _ = income_asymmetric_features(inc_all[ia], y[ia], inc_all[ib], test["Annual_Income_USD"].to_numpy()[:10], seed=INNER_SEED)
    x_a = np.column_stack([base_train[ia], *enc_a, inc_a]).astype(np.float32)
    x_b = np.column_stack([base_train[ib], *enc_b, inc_b]).astype(np.float32)
    x_v = np.column_stack([base_train[valid_idx], *enc_v, inc_v]).astype(np.float32)
    x_t = np.column_stack([base_test, *enc_t, inc_t]).astype(np.float32)
    model = make_model(MODEL, fold); model.set_params(random_state=SEED_M)
    model = fit_model(model, MODEL, x_a, y[ia], x_b, y[ib])
    oof[valid_idx] = model.predict_proba(x_v)[:, 1]; tpred += model.predict_proba(x_t)[:, 1] / N_FOLDS
    iters.append(best_iteration(model, MODEL))
    np.save(OOF / f"oof_{TAG}.npy", oof); np.save(TEST / f"test_{TAG}.npy", tpred); np.save(OOF / f"iters_{TAG}.npy", np.array(iters))
    print("%s fold%d=%.5f iters=%d (%.0fs)" % (TAG, fold, roc_auc_score(y[valid_idx], oof[valid_idx]), iters[-1], time.time() - t0), flush=True)
print("%s oof=%.5f mean_iters=%d" % (TAG, roc_auc_score(y, oof), np.mean(iters)), flush=True)
