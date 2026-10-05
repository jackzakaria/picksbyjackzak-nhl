"""Point-in-time feature engine. Every feature for a game uses only games that finished before it.

Produces features.parquet with one row per game (home perspective).
"""
import math, pandas as pd, numpy as np
from collections import defaultdict, deque

FRANCHISE = {"ATL": "WPG", "PHX": "ARI", "UTA": "ARI"}

# arena coordinates (lat, lon) by franchise code
ARENA = {
 "ANA": (33.81, -117.88), "ARI": (33.53, -112.26), "BOS": (42.37, -71.06), "BUF": (42.88, -78.88),
 "CGY": (51.04, -114.05), "CAR": (35.80, -78.72), "CHI": (41.88, -87.67), "COL": (39.75, -105.01),
 "CBJ": (39.97, -83.01), "DAL": (32.79, -96.81), "DET": (42.34, -83.06), "EDM": (53.55, -113.50),
 "FLA": (26.16, -80.33), "LAK": (34.04, -118.27), "MIN": (44.94, -93.10), "MTL": (45.50, -73.57),
 "NSH": (36.16, -86.78), "NJD": (40.73, -74.17), "NYI": (40.72, -73.59), "NYR": (40.75, -73.99),
 "OTT": (45.30, -75.93), "PHI": (39.90, -75.17), "PIT": (40.44, -79.99), "SJS": (37.33, -121.90),
 "SEA": (47.62, -122.35), "STL": (38.63, -90.20), "TBL": (27.94, -82.45), "TOR": (43.64, -79.38),
 "VAN": (49.28, -123.11), "VGK": (36.10, -115.18), "WSH": (38.90, -77.02), "WPG": (49.89, -97.14),
}
TZ = {k: (-8 if v[1] < -114 else -7 if v[1] < -102 else -6 if v[1] < -88 else -5) for k, v in ARENA.items()}
TZ["ARI"] = -7

def dist_km(a, b):
    (la1, lo1), (la2, lo2) = ARENA[a], ARENA[b]
    p = math.pi / 180
    h = 0.5 - math.cos((la2 - la1) * p) / 2 + math.cos(la1 * p) * math.cos(la2 * p) * (1 - math.cos((lo2 - lo1) * p)) / 2
    return 12742 * math.asin(math.sqrt(h))

# ---------------- Elo ----------------
K, HOME_ADV, REGRESS = 6.0, 50.0, 1 / 3

def elo_expect(dr):
    return 1 / (1 + 10 ** (-dr / 400))

def mov_mult(margin, dr_winner):
    return 0.6686 * math.log(max(abs(margin), 1)) + 0.8048  # 538-style NHL MOV multiplier

class EWMA:
    """Exponentially weighted mean with count, halflife in games."""
    def __init__(self, halflife=12.0):
        self.a = 1 - 0.5 ** (1 / halflife); self.v = None; self.n = 0
    def add(self, x):
        self.n += 1
        self.v = x if self.v is None else self.v + self.a * (x - self.v)
    def get(self, default):
        return default if self.v is None else self.v

def build(games_path="games.parquet", out="features.parquet", games=None):
    g = pd.read_parquet(games_path) if games is None else games
    g = g.sort_values(["game_date", "game_time", "game_id"]).reset_index(drop=True)
    rows = []
    elo = defaultdict(lambda: 1500.0)
    last_season = {}
    # team state
    t_last_date, t_last_loc, t_dates = {}, {}, defaultdict(deque)
    t_home_streak = defaultdict(int)  # +n = n consecutive home games, -n = road
    t_stats = defaultdict(lambda: {k: EWMA(12) for k in
                ["gf", "ga", "sf", "sa", "sh_pct", "sv_pct", "ppg", "pim", "fo", "hits", "blk", "win", "cf_proxy"]})
    t_season = defaultdict(lambda: {"gp": 0, "pts": 0, "gf": 0, "ga": 0})
    # goalie state
    LEAGUE_SV = 0.910
    g_state = defaultdict(lambda: {"shots": 0.0, "saves": 0.0, "gsax_ewm": EWMA(15), "sv_ewm": EWMA(15), "last": None, "gp": 0})
    league_sv_ewm = EWMA(200)

    prev_season = None
    for r in g.itertuples(index=False):
        h, a, d, season = r.home_fr, r.away_fr, r.game_date, r.season
        if prev_season is not None and season != prev_season:
            for t in list(elo):  # regress between seasons
                elo[t] = 1500 + (elo[t] - 1500) * (1 - REGRESS)
            t_season.clear()
            for t in t_stats:  # soften carried-over form
                for e in t_stats[t].values():
                    if e.v is not None: e.n = min(e.n, 5)
        prev_season = season
        lsv = league_sv_ewm.get(LEAGUE_SV)

        def team_feats(t, opp, is_home):
            f = {}
            ld = t_last_date.get(t)
            rest = (d - ld).days if ld is not None else 7
            f["rest"] = min(rest, 7)
            f["b2b"] = int(rest == 1)
            dq = t_dates[t]
            f["g_last7"] = sum(1 for x in dq if (d - x).days <= 7)
            f["g_last3"] = sum(1 for x in dq if (d - x).days <= 3)
            loc_now = h  # game location is home team's arena
            ll = t_last_loc.get(t)
            f["travel"] = dist_km(ll, loc_now) if ll else 0.0
            f["tz_shift"] = (TZ[loc_now] - TZ[ll]) if ll else 0
            st = t_stats[t]
            f["n_form"] = st["gf"].n
            f["gf"] = st["gf"].get(2.9); f["ga"] = st["ga"].get(2.9)
            f["sf"] = st["sf"].get(30.0); f["sa"] = st["sa"].get(30.0)
            f["sh_pct"] = st["sh_pct"].get(0.095); f["sv_pct"] = st["sv_pct"].get(0.905)
            f["pdo"] = f["sh_pct"] + f["sv_pct"]
            f["ppg"] = st["ppg"].get(0.6); f["pim"] = st["pim"].get(9.0); f["fo"] = st["fo"].get(0.5)
            f["win_form"] = st["win"].get(0.5)
            f["shot_share"] = f["sf"] / (f["sf"] + f["sa"])
            ss = t_season[t]
            f["pts_pct"] = ss["pts"] / (2 * ss["gp"]) if ss["gp"] else 0.55
            f["gp_season"] = ss["gp"]
            f["gdiff_season"] = (ss["gf"] - ss["ga"]) / ss["gp"] if ss["gp"] else 0.0
            hs = t_home_streak[t]
            f["streak_loc"] = hs if is_home else -hs
            return f

        def goalie_feats(pid, sa_hist_key):
            gs = g_state[pid] if pid is not None and not (isinstance(pid, float) and np.isnan(pid)) else None
            f = {}
            if gs is None or gs["gp"] == 0:
                f.update(g_gsax=0.0, g_sv=lsv, g_shots=0.0, g_gp=0, g_b2b=0, g_rest=7)
                return f
            prior_shots = 1500.0  # shrink toward league average
            f["g_sv"] = (gs["saves"] + lsv * prior_shots) / (gs["shots"] + prior_shots)
            f["g_gsax"] = gs["gsax_ewm"].get(0.0)  # goals saved above expected per game, recent
            f["g_shots"] = min(gs["shots"], 20000) / 1000
            f["g_gp"] = gs["gp"]
            gl = gs["last"]
            f["g_b2b"] = int(gl is not None and (d - gl).days == 1)
            f["g_rest"] = min((d - gl).days, 10) if gl is not None else 7
            return f

        fh, fa = team_feats(h, a, True), team_feats(a, h, False)
        gh, ga_ = goalie_feats(r.hg_player_id, None), goalie_feats(r.ag_player_id, None)
        eh, ea = elo[h], elo[a]
        pending = pd.isna(r.home_score)
        row = {"game_id": r.game_id, "season": season, "game_type": r.game_type, "game_date": d,
               "home": r.home_team_abbr, "away": r.away_team_abbr, "home_win": (np.nan if pending else r.home_win), "went_ot": r.went_ot,
               "home_score": r.home_score, "away_score": r.away_score,
               "elo_h": eh, "elo_a": ea, "elo_diff": eh - ea + HOME_ADV, "elo_prob": elo_expect(eh - ea + HOME_ADV),
               "playoff": int(r.game_type == "P"),
               "hg_name": r.hg_player_name, "ag_name": r.ag_player_name}
        for k, v in fh.items(): row[f"h_{k}"] = v
        for k, v in fa.items(): row[f"a_{k}"] = v
        for k, v in gh.items(): row[f"h{k}"] = v
        for k, v in ga_.items(): row[f"a{k}"] = v
        rows.append(row)
        if pending:
            continue  # scheduled game: features emitted, no state update

        # ---------- update state with this game's result ----------
        margin = int(r.home_score - r.away_score)
        res_h = 1.0 if margin > 0 else 0.0
        exp_h = elo_expect(eh - ea + HOME_ADV)
        shift = K * mov_mult(margin, 0) * (res_h - exp_h)
        elo[h] += shift; elo[a] -= shift

        for t, opp, gf, ga, sf, sa, ppg, pim, fo, won, is_home in (
            (h, a, r.home_score, r.away_score, r.h_shots_on_goal, r.a_shots_on_goal, r.h_power_play_goals, r.h_pim, r.h_faceoff_win_pctg, res_h, True),
            (a, h, r.away_score, r.home_score, r.a_shots_on_goal, r.h_shots_on_goal, r.a_power_play_goals, r.a_pim, r.a_faceoff_win_pctg, 1 - res_h, False)):
            st = t_stats[t]
            sf = sf if pd.notna(sf) else 30; sa = sa if pd.notna(sa) else 30
            st["gf"].add(gf); st["ga"].add(ga); st["sf"].add(sf); st["sa"].add(sa)
            st["sh_pct"].add(gf / max(sf, 1)); st["sv_pct"].add(1 - ga / max(sa, 1))
            st["ppg"].add(ppg if pd.notna(ppg) else 0); st["pim"].add(pim if pd.notna(pim) else 8)
            st["fo"].add(fo if pd.notna(fo) else 0.5); st["win"].add(won)
            t_last_date[t] = d; t_last_loc[t] = h
            dq = t_dates[t]; dq.append(d)
            while dq and (d - dq[0]).days > 10: dq.popleft()
            if r.game_type == "R":
                ss = t_season[t]; ss["gp"] += 1; ss["gf"] += gf; ss["ga"] += ga
                ss["pts"] += 2 if won else (1 if r.went_ot else 0)
            hs = t_home_streak[t]
            t_home_streak[t] = (hs + 1 if hs > 0 else 1) if is_home else (hs - 1 if hs < 0 else -1)

        for pid, sa, sv in ((r.hg_player_id, r.hg_shots_against, r.hg_saves), (r.ag_player_id, r.ag_shots_against, r.ag_saves)):
            if pid is None or (isinstance(pid, float) and np.isnan(pid)) or pd.isna(sa): continue
            gs = g_state[pid]
            gs["shots"] += sa; gs["saves"] += sv; gs["gp"] += 1; gs["last"] = d
            gs["gsax_ewm"].add(sv - lsv * sa)
            gs["sv_ewm"].add(sv / sa if sa else lsv)
            league_sv_ewm.add(sv / sa if sa else lsv)

    f = pd.DataFrame(rows)
    # differential features (home minus away)
    for k in ["rest", "b2b", "g_last7", "g_last3", "travel", "tz_shift", "gf", "ga", "sf", "sa", "sh_pct", "sv_pct", "pdo",
              "ppg", "pim", "fo", "win_form", "shot_share", "pts_pct", "gdiff_season", "streak_loc"]:
        f[f"d_{k}"] = f[f"h_{k}"] - f[f"a_{k}"]
    for k in ["g_gsax", "g_sv", "g_shots", "g_gp", "g_b2b", "g_rest"]:
        f[f"d{k}"] = f[f"h{k}"] - f[f"a{k}"]
    if out: f.to_parquet(out, index=False)
    return f

if __name__ == "__main__":
    f = build()
    print(f.shape)
    print(f[["season", "elo_prob", "home_win"]].groupby("season").agg(n=("home_win", "size"), hw=("home_win", "mean"),
          elo_acc=("elo_prob", lambda s: ((s > 0.5).astype(int) == f.loc[s.index, "home_win"]).mean())).round(3))
