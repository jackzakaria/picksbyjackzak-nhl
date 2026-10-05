"""Build point-in-time MoneyPuck xG features from shot-level ZIP/CSV files.

Expected input directory:
  moneypuck_shots/
    shots_2015.zip ... shots_2024.zip
    shots_2007-2025.zip (also supported)
    shots_2025.zip / shots_2026.zip when available

No game outcome is used to construct a game's pregame xG feature.  Each team's
state is updated only after the completed game.

The output is xg_pregame.parquet, keyed by game_id, with home/away EWMA xGF/xGA,
xG share, xG differential, and optional high-danger xG when a suitable distance
column is present.  The high-danger definition is deliberately configurable and
is not presented as an official MoneyPuck category.
"""
from __future__ import annotations
import argparse, glob, math, os, re, zipfile
from collections import defaultdict
import numpy as np
import pandas as pd

try:
    import pyarrow  # noqa: F401
except Exception as e:
    raise SystemExit("This script needs pyarrow/pandas parquet support: " + str(e))

FRANCHISE = {"ATL": "WPG", "PHX": "ARI", "UTA": "ARI"}

class EWMA:
    def __init__(self, halflife=12.0):
        self.a = 1 - 0.5 ** (1 / halflife)
        self.v = None
    def add(self, x: float):
        self.v = float(x) if self.v is None else self.v + self.a * (float(x) - self.v)
    def get(self, default: float = 0.0) -> float:
        return default if self.v is None else float(self.v)


def find_csvs(shot_dir: str):
    return sorted(glob.glob(os.path.join(shot_dir, "*.zip")) + glob.glob(os.path.join(shot_dir, "*.csv")))


def normalize_cols(df: pd.DataFrame):
    ren = {}
    for c in df.columns:
        lc = c.lower()
        if lc == "game_id": ren[c] = "game_id"
        elif lc in {"teamcode", "team_code", "team"}: ren[c] = "teamCode"
        elif lc in {"xgoal", "xgoals", "xg"}: ren[c] = "xGoal"
        elif lc in {"shotdistance", "shot_distance", "distance"}: ren[c] = "shotDistance"
        elif lc in {"goal", "isgoal", "is_goal"}: ren[c] = "isGoal"
        elif lc in {"event", "event_type"}: ren[c] = "event"
    return df.rename(columns=ren)


def read_zip_csv(path: str, chunksize: int = 250_000):
    with zipfile.ZipFile(path) as z:
        csvs = [n for n in z.namelist() if n.lower().endswith(".csv")]
        if not csvs:
            return
        # A season ZIP normally contains one CSV. If multiple are present, read all.
        for name in csvs:
            with z.open(name) as f:
                yield from pd.read_csv(f, chunksize=chunksize, low_memory=False)


def read_shots(path: str):
    if path.lower().endswith(".zip"):
        return read_zip_csv(path)
    return pd.read_csv(path, chunksize=250_000, low_memory=False)


def aggregate_shots(shot_dir: str, high_danger_ft: float = 25.0):
    acc = []
    for path in find_csvs(shot_dir):
        print("reading", os.path.basename(path))
        for chunk in read_shots(path):
            chunk = normalize_cols(chunk)
            required = {"game_id", "teamCode", "xGoal"}
            if not required.issubset(chunk.columns):
                raise ValueError(f"Missing {required - set(chunk.columns)} in {path}; columns={list(chunk.columns)[:20]}")
            chunk = chunk[[c for c in ["game_id", "teamCode", "xGoal", "shotDistance", "isGoal"] if c in chunk.columns]].copy()
            chunk["game_id"] = pd.to_numeric(chunk["game_id"], errors="coerce")
            chunk["xGoal"] = pd.to_numeric(chunk["xGoal"], errors="coerce").fillna(0.0)
            chunk = chunk.dropna(subset=["game_id", "teamCode"])
            if "shotDistance" in chunk:
                chunk["shotDistance"] = pd.to_numeric(chunk["shotDistance"], errors="coerce")
                chunk["hd_xg"] = np.where(chunk["shotDistance"] <= high_danger_ft, chunk["xGoal"], 0.0)
            else:
                chunk["hd_xg"] = np.nan
            chunk["teamCode"] = chunk["teamCode"].astype(str).map(lambda x: FRANCHISE.get(x, x))
            g = chunk.groupby(["game_id", "teamCode"], as_index=False).agg(
                xgf=("xGoal", "sum"),
                hd_xgf=("hd_xg", "sum"),
                shots=("xGoal", "size"),
            )
            acc.append(g)
    if not acc:
        raise FileNotFoundError("No MoneyPuck shot ZIP/CSV files found")
    out = pd.concat(acc, ignore_index=True)
    out = out.groupby(["game_id", "teamCode"], as_index=False).agg(
        xgf=("xgf", "sum"), hd_xgf=("hd_xgf", "sum"), shots=("shots", "sum")
    )
    return out


def build_pregame(games_path: str, shot_dir: str, out: str, halflife: float, high_danger_ft: float):
    games = pd.read_parquet(games_path).copy()
    games = games.sort_values(["game_date", "game_time", "game_id"]).reset_index(drop=True)
    shot = aggregate_shots(shot_dir, high_danger_ft=high_danger_ft)
    shot["teamCode"] = shot["teamCode"].map(lambda x: FRANCHISE.get(x, x))
    shot_map = {(int(r.game_id), r.teamCode): r for r in shot.itertuples(index=False)}

    state = defaultdict(lambda: {
        "xgf": EWMA(halflife), "xga": EWMA(halflife), "hd_xgf": EWMA(halflife),
        "hd_xga": EWMA(halflife), "games": 0
    })
    rows = []
    prev_season = None
    for r in games.itertuples(index=False):
        season = int(r.season)
        if prev_season is not None and season != prev_season:
            # Do not reset the state: carryover is known information. EWMA itself
            # naturally de-weights older observations; season effects can be learned
            # by the downstream model.
            pass
        prev_season = season
        h = r.home_fr; a = r.away_fr
        hs = state[h]; ats = state[a]

        def vals(tstate):
            return {
                "xgf": tstate["xgf"].get(2.9), "xga": tstate["xga"].get(2.9),
                "hd_xgf": tstate["hd_xgf"].get(0.0), "hd_xga": tstate["hd_xga"].get(0.0),
                "games": tstate["games"],
            }
        hv, av = vals(hs), vals(ats)
        row = {"game_id": int(r.game_id), "season": season}
        for k,v in hv.items(): row[f"h_xg_{k}"] = v
        for k,v in av.items(): row[f"a_xg_{k}"] = v
        row["d_xg_xgf"] = hv["xgf"] - av["xgf"]
        row["d_xg_xga"] = hv["xga"] - av["xga"]
        row["d_xg_diff"] = (hv["xgf"] - hv["xga"]) - (av["xgf"] - av["xga"])
        h_den = hv["xgf"] + hv["xga"]; a_den = av["xgf"] + av["xga"]
        row["h_xg_share"] = hv["xgf"] / h_den if h_den else 0.5
        row["a_xg_share"] = av["xgf"] / a_den if a_den else 0.5
        row["d_xg_share"] = row["h_xg_share"] - row["a_xg_share"]
        row["d_hd_xgf"] = hv["hd_xgf"] - av["hd_xgf"]
        row["d_hd_xga"] = hv["hd_xga"] - av["hd_xga"]
        row["d_hd_xg_diff"] = (hv["hd_xgf"] - hv["hd_xga"]) - (av["hd_xgf"] - av["hd_xga"])
        rows.append(row)

        # Only completed games update state.
        if pd.isna(r.home_score) or pd.isna(r.away_score):
            continue
        hg = shot_map.get((int(r.game_id), h))
        ag = shot_map.get((int(r.game_id), a))
        hx = float(hg.xgf) if hg is not None else 0.0
        ax = float(ag.xgf) if ag is not None else 0.0
        hhd = float(hg.hd_xgf) if hg is not None and pd.notna(hg.hd_xgf) else 0.0
        ahd = float(ag.hd_xgf) if ag is not None and pd.notna(ag.hd_xgf) else 0.0
        hs["xgf"].add(hx); hs["xga"].add(ax); hs["hd_xgf"].add(hhd); hs["hd_xga"].add(ahd); hs["games"] += 1
        ats["xgf"].add(ax); ats["xga"].add(hx); ats["hd_xgf"].add(ahd); ats["hd_xga"].add(hhd); ats["games"] += 1

    outdf = pd.DataFrame(rows)
    outdf.to_parquet(out, index=False)
    print("wrote", out, outdf.shape)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", default="games.parquet")
    ap.add_argument("--shots-dir", default="moneypuck_shots")
    ap.add_argument("--out", default="xg_pregame.parquet")
    ap.add_argument("--halflife", type=float, default=12.0)
    ap.add_argument("--high-danger-ft", type=float, default=25.0)
    args = ap.parse_args()
    build_pregame(args.games, args.shots_dir, args.out, args.halflife, args.high_danger_ft)
