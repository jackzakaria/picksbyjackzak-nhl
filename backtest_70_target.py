"""All-games NHL accuracy backtest.

This script deliberately does NOT filter out games. Every game in the test
window receives a pick.  It is designed to test whether xG + goalie + nonlinear
interactions + model stacking can materially improve the current ~58.9% baseline.

No claim of 70% is made unless the identical walk-forward test actually produces it.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
from sklearn.ensemble import HistGradientBoostingClassifier
import lightgbm as lgb

ROOT = Path(__file__).resolve().parent
F_PATH = ROOT / "features.parquet"
XG_PATH = ROOT / "xg_pregame.parquet"
OUT = ROOT / "backtest_70_target_summary.json"

BASE = ["elo_diff","d_pts_pct","d_gdiff_season","d_rest","h_b2b","a_b2b","d_travel","d_shot_share","d_pdo","d_win_form"]
GOALIE = ["dg_gsax","dg_sv","dg_shots","dg_gp","hg_b2b","ag_b2b","dg_rest","hg_gp","ag_gp"]
TEAM = [
    "elo_diff","d_rest","h_b2b","a_b2b","d_g_last7","d_g_last3","d_travel","a_travel","d_tz_shift",
    "d_gf","d_ga","d_sf","d_sa","d_sh_pct","d_sv_pct","d_pdo","d_ppg","d_pim","d_fo","d_win_form",
    "d_shot_share","d_pts_pct","d_gdiff_season","d_streak_loc","h_gp_season","h_n_form"
]
XG = ["d_xg_xgf","d_xg_xga","d_xg_diff","d_xg_share","d_hd_xgf","d_hd_xga","d_hd_xg_diff"]


def add_interactions(df):
    df = df.copy()
    pairs = [
        ("elo_diff","d_xg_diff"),("elo_diff","d_shot_share"),("d_xg_diff","d_shot_share"),
        ("dg_gsax","elo_diff"),("dg_gsax","d_xg_diff"),("dg_sv","d_xg_xga"),
        ("d_pdo","d_xg_diff"),("d_rest","d_xg_diff"),("h_b2b","d_xg_diff"),
        ("d_gdiff_season","d_xg_diff"),("d_pts_pct","d_xg_diff")
    ]
    for a,b in pairs:
        if a in df and b in df:
            df[f"ix_{a}_{b}"] = df[a].astype(float) * df[b].astype(float)
    # Smooth nonlinear versions of the strongest rating variables.
    if "elo_diff" in df: df["elo_sq"] = np.sign(df.elo_diff) * np.log1p(np.abs(df.elo_diff))
    if "d_xg_diff" in df: df["xg_abs"] = np.log1p(np.abs(df.d_xg_diff)) * np.sign(df.d_xg_diff)
    return df


def fit_lr(X, y):
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=4000)).fit(X, y)


def fit_lgb(X, y):
    return lgb.LGBMClassifier(
        n_estimators=500, learning_rate=0.015, num_leaves=7, max_depth=4,
        min_child_samples=100, subsample=0.85, subsample_freq=1,
        colsample_bytree=0.85, reg_lambda=12.0, reg_alpha=0.25, verbose=-1
    ).fit(X, y)


def fit_hgb(X, y):
    return HistGradientBoostingClassifier(
        learning_rate=0.04, max_iter=300, max_leaf_nodes=7,
        min_samples_leaf=100, l2_regularization=2.0, random_state=7
    ).fit(X, y)


def make_features():
    f = pd.read_parquet(F_PATH)
    f = f[f.game_type == "R"].copy()
    if XG_PATH.exists():
        xg = pd.read_parquet(XG_PATH)
        f = f.merge(xg, on=["game_id","season"], how="left")
        for c in XG:
            f[c] = f[c].fillna(0.0)
        return add_interactions(f), True
    return add_interactions(f), False


def calibration_slope(p, y):
    # Regress outcome on model log-odds. A slope near 1 means the probabilities
    # are appropriately spread; <1 is overconfident and >1 is underconfident.
    eps = 1e-6
    z = np.log(np.clip(p, eps, 1-eps) / np.clip(1-p, eps, 1-eps)).reshape(-1,1)
    m = LogisticRegression(C=1e6, max_iter=4000).fit(z, y)
    return float(m.coef_[0,0])


def score(p, y):
    pick = (p >= 0.5).astype(int)
    return {
        "n": int(len(y)),
        "acc": float(accuracy_score(y, pick)),
        "logloss": float(log_loss(y, p, labels=[0,1])),
        "brier": float(brier_score_loss(y, p)),
        "calibration_slope": calibration_slope(p, y),
        "pick_home_rate": float(pick.mean())
    }


def season_threshold(oof_p, y):
    # Optional decision-threshold optimization, but every game remains included.
    cand = np.arange(0.40, 0.601, 0.01)
    best_t, best_a = 0.5, -1
    for t in cand:
        a = accuracy_score(y, (oof_p >= t).astype(int))
        if a > best_a:
            best_a, best_t = a, float(t)
    return best_t


def train_base(train, test, cols):
    Xtr = train[cols].fillna(train[cols].median(numeric_only=True)).astype(float)
    Xte = test[cols].fillna(train[cols].median(numeric_only=True)).astype(float)
    y = train.home_win.astype(int).values
    out = []
    for fn in (fit_lr, fit_lgb, fit_hgb):
        m = fn(Xtr, y)
        out.append(m.predict_proba(Xte)[:,1])
    return np.column_stack(out)


def walk_forward(f, cols, start_season=2018):
    seasons = sorted(int(s) for s in f.season.unique() if int(s) >= start_season)
    all_rows = []
    models = []
    for s in seasons:
        train = f[f.season < s].copy()
        test = f[f.season == s].copy()
        if train.empty or test.empty: continue
        # Base predictions on the test season.
        P_test = train_base(train, test, cols)
        # Learn stacking weights from prior-season OOF predictions.
        oof_parts = []
        prior_seasons = sorted(int(q) for q in train.season.unique())
        for q in prior_seasons[1:]:
            tr2 = train[train.season < q]
            te2 = train[train.season == q]
            if len(tr2) < 1000 or te2.empty: continue
            Pq = train_base(tr2, te2, cols)
            oof_parts.append(pd.DataFrame({"y": te2.home_win.astype(int).values,
                                           "p0": Pq[:,0],"p1":Pq[:,1],"p2":Pq[:,2]}))
        if oof_parts:
            oof = pd.concat(oof_parts, ignore_index=True)
            meta = LogisticRegression(C=0.3, max_iter=2000).fit(oof[["p0","p1","p2"]], oof.y)
            ptest = meta.predict_proba(P_test)[:,1]
            t = season_threshold(meta.predict_proba(oof[["p0","p1","p2"]])[:,1], oof.y)
        else:
            ptest = P_test.mean(axis=1)
            t = 0.5
        all_rows.append(pd.DataFrame({"season":s, "game_id":test.game_id.values,
                                      "y":test.home_win.astype(int).values,
                                      "p":ptest, "threshold":t}))
    return pd.concat(all_rows, ignore_index=True)


def main():
    f, has_xg = make_features()
    # Identical test window to the proposed xG comparison.
    test_start = 2018
    # Compare compact baseline vs compact+goalie vs xG vs all.
    feature_sets = {
        "current_10": BASE,
        "current_plus_goalie": BASE + GOALIE,
        "current_plus_xg": BASE + XG,
        "current_plus_xg_goalie": BASE + XG + GOALIE,
        "xg_goalie_interactions": BASE + XG + GOALIE + [c for c in f.columns if c.startswith("ix_") or c in {"elo_sq","xg_abs"}],
        "all": sorted(set(TEAM + GOALIE + XG + [c for c in f.columns if c.startswith("ix_") or c in {"elo_sq","xg_abs"}]))
    }
    results = {}
    for name, cols in feature_sets.items():
        cols = [c for c in cols if c in f.columns]
        print("testing", name, "features", len(cols))
        P = walk_forward(f, cols, start_season=test_start)
        # Mandatory all-game metrics: NO filtering whatsoever.
        base = score(P.p.values, P.y.values)
        p_adj = np.where(P.p >= P.threshold, P.p, P.p)  # decision threshold affects picks only; probability stays honest.
        picks = (P.p >= P.threshold).astype(int)
        base["acc_thresholded"] = float(accuracy_score(P.y, picks))
        base["avg_threshold"] = float(P.threshold.mean())
        results[name] = base
        print(name, base)
    payload = {"test_start": test_start, "has_moneypuck_xg": has_xg, "results": results}
    OUT.write_text(json.dumps(payload, indent=2))
    print("wrote", OUT)

if __name__ == "__main__":
    main()
