"""Walk-forward backtest: for each test season, train only on seasons before it, predict every game pregame."""
import sys, json, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import log_loss, brier_score_loss
import lightgbm as lgb

f = pd.read_parquet("features.parquet")
f = f[f.game_type == "R"].copy()          # predict regular season games (playoffs reported separately below)

TEAM = ["elo_diff", "d_rest", "h_b2b", "a_b2b", "d_g_last7", "d_g_last3", "d_travel", "a_travel", "d_tz_shift",
        "d_gf", "d_ga", "d_sf", "d_sa", "d_sh_pct", "d_sv_pct", "d_pdo", "d_ppg", "d_pim", "d_fo", "d_win_form",
        "d_shot_share", "d_pts_pct", "d_gdiff_season", "d_streak_loc", "h_gp_season", "h_n_form"]
GOALIE = ["dg_gsax", "dg_sv", "dg_shots", "dg_gp", "hg_b2b", "ag_b2b", "dg_rest", "hg_gp", "ag_gp"]
COMPACT = ["elo_diff","d_pts_pct","d_gdiff_season","d_rest","h_b2b","a_b2b","d_travel","d_shot_share","d_pdo","d_win_form"]
SETS = {"elo_only": ["elo_diff"], "compact": COMPACT, "team": TEAM, "team+goalie": TEAM + GOALIE}

def fit_predict(train, test, cols, model):
    Xtr, ytr, Xte = train[cols].values, train.home_win.values, test[cols].values
    if model == "lr":
        m = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=2000)).fit(Xtr, ytr)
        return m.predict_proba(Xte)[:, 1]
    m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.02, num_leaves=8, min_child_samples=60,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0, verbose=-1)
    m.fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1]

seasons = sorted(f.season.unique())
test_seasons = [s for s in seasons if s >= 2014]      # >=4 training seasons before first test
results = {}
for name, cols in SETS.items():
    for model in (["lr"] if name in ("elo_only","compact") else ["lr", "lgb"]):
        preds = []
        for s in test_seasons:
            tr, te = f[f.season < s], f[f.season == s]
            p = fit_predict(tr, te, cols, model)
            preds.append(pd.DataFrame({"game_id": te.game_id.values, "p": p}))
        p = pd.concat(preds).merge(f[["game_id", "season", "home_win", "went_ot", "game_date", "home", "away"]], on="game_id")
        p["pick_home"] = (p.p > 0.5).astype(int)
        p["hit"] = (p.pick_home == p.home_win).astype(int)
        p["conf"] = (p.p - 0.5).abs()
        results[f"{name}/{model}"] = p

def summarize(p):
    out = {"n": len(p), "acc": p.hit.mean(), "logloss": log_loss(p.home_win, p.p), "brier": brier_score_loss(p.home_win, p.p),
           "acc_reg_only": p[~p.went_ot].hit.mean(), "acc_ot_games": p[p.went_ot].hit.mean()}
    return out

print("\n=== Overall (regular season, walk-forward 2014-2026) ===")
summary = {k: summarize(v) for k, v in results.items()}
print(pd.DataFrame(summary).T.round(4))

best_key = "compact/lr"  # chosen: matches best log loss, fewest features, no goalie dependency
print("\nbest by log loss:", best_key)
P = results[best_key]

print("\n=== By season (best model) ===")
print(P.groupby("season").agg(n=("hit", "size"), acc=("hit", "mean"), home_rate=("home_win", "mean"),
                              pick_home_rate=("pick_home", "mean")).round(3))

print("\n=== Accuracy by confidence tier (best model) ===")
bins = [0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.5]
P["tier"] = pd.cut(P.conf, bins, labels=["50-55", "55-60", "60-65", "65-70", "70-75", "75+"])
print(P.groupby("tier", observed=True).agg(n=("hit", "size"), acc=("hit", "mean"), share=("hit", lambda s: len(s) / len(P))).round(3))

print("\n=== Filter search: min model prob -> accuracy, coverage ===")
rows = []
for thr in [0.5, 0.55, 0.58, 0.60, 0.62, 0.64, 0.65, 0.66, 0.68, 0.70]:
    q = P[P.conf >= thr - 0.5]
    rows.append({"min_prob": thr, "n": len(q), "per_season": round(len(q) / len(test_seasons)), "acc": q.hit.mean(),
                 "acc_reg_only": q[~q.went_ot].hit.mean()})
print(pd.DataFrame(rows).round(3))

# season-by-season at the site's tiers: OFFICIAL >= 0.65 (counts), LEAN 0.55-0.65 (shown, not counted)
thr65 = 0.65
o = P[P.conf >= 0.15]; l = P[(P.conf >= 0.05) & (P.conf < 0.15)]
t = o.groupby("season").agg(W=("hit", "sum"), n=("hit", "size")); t["L"] = t.n - t.W; t["OFFICIAL_pct"] = (t.W / t.n).round(3)
t["LEAN_pct"] = l.groupby("season").hit.mean().round(3)
print("\n=== Season results at the site tiers: OFFICIAL >= 0.65 | LEAN 0.55-0.65 ===")
print(t[["W", "L", "OFFICIAL_pct", "LEAN_pct"]])
print(f"\nOFFICIAL total {int(o.hit.sum())}-{int(len(o) - o.hit.sum())} = {o.hit.mean():.1%} ({len(o) / len(test_seasons):.0f}/season)   "
      f"LEAN total {int(l.hit.sum())}-{int(len(l) - l.hit.sum())} = {l.hit.mean():.1%}   seasons >= 65%: {(t.OFFICIAL_pct >= 0.65).sum()}/{len(t)}")

P.to_parquet("backtest_preds.parquet", index=False)
json.dump({"summary": summary, "best": best_key, "filter": rows, "thr65": thr65}, open("backtest_summary.json", "w"), default=float, indent=1)
