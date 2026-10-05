"""Merge fastRhockey per-season parquets into one game-level table (one row per game)."""
import glob, pandas as pd, numpy as np

D = "data"
FRANCHISE = {"ATL": "WPG", "PHX": "ARI", "UTA": "ARI"}  # franchise continuity for ratings

def load(kind):
    return pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{D}/{kind}/parquet/*.parquet"))], ignore_index=True)

gi = load("game_info")
gi = gi[gi.game_type.isin(["R", "P"]) & gi.game_state.isin(["OFF", "FINAL"])].copy()
gi["game_id"] = gi.game_id.astype(int)
gi = gi.drop_duplicates("game_id")
gi["season"] = gi.game_id.astype(str).str[:4].astype(int) + 1   # season end year (2014-15 -> 2015)
gi["game_date"] = pd.to_datetime(gi.game_date)

sched = load("schedules")[["game_id", "game_time", "playoff_round"]].drop_duplicates("game_id")
sched["game_id"] = sched.game_id.astype(int)
sched["game_time"] = pd.to_datetime(sched.game_time, utc=True, errors="coerce")

box = load("team_box"); box["game_id"] = box.game_id.astype(int)
box = box.drop_duplicates(["game_id", "home_away"])
ls = load("linescore"); ls["game_id"] = ls.game_id.astype(int); ls = ls.drop_duplicates("game_id")

gb = load("goalie_box"); gb["game_id"] = gb.game_id.astype(int)
def toi_sec(s):
    try: m, sec = str(s).split(":"); return int(m) * 60 + int(sec)
    except Exception: return 0
gb["toi_s"] = gb.toi.map(toi_sec)
# starter = goalie flagged starter; fall back to most TOI
gb["st"] = gb.starter.fillna(False).astype(bool).astype(int)
gb = gb.sort_values(["st", "toi_s"], ascending=False).drop_duplicates(["game_id", "home_away"])
gb = gb[["game_id", "home_away", "player_id", "player_name", "shots_against", "saves", "goals_against", "toi_s", "decision"]]
gb["game_id"] = gb.game_id.astype(int)
# OT indicator: any goalie decision 'O' in the game (OT/SO loss) -> game went past regulation
ot_ids = set(load("goalie_box").query("decision == 'O'").game_id.astype(int))

def side(df, ha, prefix, drop=("home_away", "team_id", "team_name", "season", "game_date")):
    d = df[df.home_away == ha].drop(columns=[c for c in drop if c in df.columns])
    return d.rename(columns={c: f"{prefix}_{c}" for c in d.columns if c != "game_id"})

g = gi.merge(sched, on="game_id", how="left")
g = g.merge(side(box, "home", "h"), on="game_id", how="left").merge(side(box, "away", "a"), on="game_id", how="left")
g = g.merge(side(gb, "home", "hg"), on="game_id", how="left").merge(side(gb, "away", "ag"), on="game_id", how="left")
g = g.merge(ls[["game_id", "has_shootout"]], on="game_id", how="left")
g["went_ot"] = g.game_id.isin(ot_ids)
g["home_win"] = (g.home_score > g.away_score).astype(int)
g["home_fr"] = g.home_team_abbr.map(lambda a: FRANCHISE.get(a, a))
g["away_fr"] = g.away_team_abbr.map(lambda a: FRANCHISE.get(a, a))
g = g[g.home_score != g.away_score]  # drop any unfinished/odd rows
g = g.sort_values(["game_date", "game_time", "game_id"]).reset_index(drop=True)
g.to_parquet("games.parquet", index=False)

print(g.shape, g.game_date.min().date(), g.game_date.max().date())
print(g.groupby(["season", "game_type"]).size().unstack())
r = g[g.game_type == "R"]
print("home win rate (reg):", r.home_win.mean().round(4), " OT/SO share:", r.went_ot.mean().round(3))
print("coverage  box:", g.h_shots_on_goal.notna().mean().round(3), " goalie:", g.hg_player_id.notna().mean().round(3))
