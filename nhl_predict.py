"""NHL game picks: refresh data, then predict a slate.

  python3 nhl_predict.py refresh                 # pull latest completed games (fastRhockey daily parquets)
  python3 nhl_predict.py predict                 # today's slate from the NHL API (api-web.nhle.com)
  python3 nhl_predict.py predict --date 2026-10-08
  python3 nhl_predict.py predict --games TOR@MTL,BOS@NYR --date 2026-10-08   # manual slate, no API needed

Model: walk-forward validated logistic regression on 10 pregame features (Elo, points%, season goal diff,
rest/back-to-back, travel, shot share, PDO, recent form). No goalie inputs, so picks can be locked the night before.
Backtest 2014-2026 regular season: 58.9% all games; OFFICIAL tier (prob >= 0.65) 2043-879 = 69.9%, ~225/season, >=65% every season.

Ledger rules (same as the NFL side): a pick is written once, at first run, with a locked_at stamp. Re-running a date never
changes an existing pick. Games that already started are never back-filled. `grade` scores locked picks against results.
"""
import sys, json, argparse, datetime as dt, urllib.request, subprocess, pandas as pd, numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

HERE = Path(__file__).resolve().parent
RAW = "https://raw.githubusercontent.com/sportsdataverse/fastRhockey-nhl-data/main/nhl"
KINDS = {"game_info": "game_info", "team_box": "team_box", "goalie_box": "goalie_box", "linescore": "linescore", "schedules": "nhl_schedule"}
FEATURES = ["elo_diff", "d_pts_pct", "d_gdiff_season", "d_rest", "h_b2b", "a_b2b", "d_travel", "d_shot_share", "d_pdo", "d_win_form"]
OFFICIAL, LEAN = 0.65, 0.55   # OFFICIAL = counts toward the record (backtest 69.9%, ~225/season); LEAN = shown, not counted (59.5%); below LEAN = no pick

# ---- NO-EDGE INVERSE TRIAL, owner decision 2026-10-02, recorded as his ------
# For these game dates only, a PASS entry is written with the OTHER team as the
# pick and its probability inverted to match. OFFICIAL and LEAN never flip.
#
# Hard-coded dates and nothing else, so the trial ends by itself: from Oct 5 the
# set no longer matches and `predict` is the model again with no manual step and
# nothing to remember to undo.
#
# Every new entry carries `pick_rule`, so the ledger says which rule produced it
# and the trial can be separated from the model at grading time rather than
# inferred later. A second tell, free: a trial entry's `p_pick` is <= 0.50, which
# the model rule can never produce, because it always stores the probability of
# the side it picked.
#
# The measurement this is testing against: over 15,972 walk-forward games the
# 50-55% band hit 52.1% and its inverse 47.9%, and the only sub-slice where the
# inverse won (50.0-51.0%, 49.1%) has a 95% interval of 46.1-52.0% and went the
# other way in 6 of 13 seasons. Three days of PASS games cannot settle that; the
# trial is recorded here with its own expectation so the result is read honestly.
NOEDGE_INVERSE_DATES = frozenset({"2026-10-02", "2026-10-03", "2026-10-04"})


def season_end_year(date):
    return date.year + 1 if date.month >= 9 else date.year


def refresh():
    y = season_end_year(dt.date.today())
    for kind, prefix in KINDS.items():
        for yy in (y - 1, y):
            url = f"{RAW}/{kind}/parquet/{prefix}_{yy}.parquet"
            dest = HERE / "data" / kind / "parquet" / f"{prefix}_{yy}.parquet"
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                urllib.request.urlretrieve(url, dest); print("ok ", url.split("/nhl/")[1])
            except Exception as e:
                print("skip", url.split("/nhl/")[1], "->", type(e).__name__)
    subprocess.run([sys.executable, str(HERE / "build_dataset.py")], cwd=HERE, check=True)


def fetch_slate(date):
    url = f"https://api-web.nhle.com/v1/schedule/{date.isoformat()}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/128 Safari/537.36",
                                              "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=20) as r:
        j = json.load(r)
    out = []
    for day in j.get("gameWeek", []):
        if day["date"] != date.isoformat():
            continue
        for gm in day["games"]:
            if gm.get("gameType") not in (2, 3):  # 2 = regular season, 3 = playoffs
                continue
            out.append({"game_id": gm["id"], "home": gm["homeTeam"]["abbrev"], "away": gm["awayTeam"]["abbrev"],
                        "game_type": "R" if gm["gameType"] == 2 else "P", "start_utc": gm.get("startTimeUTC")})
    return out


def predict(date, manual=None):
    import features as F
    games = pd.read_parquet(HERE / "games.parquet")
    games = games[games.game_date < pd.Timestamp(date)]
    if manual:
        slate = [{"game_id": 9_000_000_000 + i, "home": h, "away": a, "game_type": "R", "start_utc": None}
                 for i, (a, h) in enumerate(x.upper().split("@") for x in manual.split(","))]
    else:
        slate = fetch_slate(date)
    if not slate:
        print("no games on", date); return
    season = season_end_year(date)
    fr = lambda t: F.FRANCHISE.get(t, t)
    add = pd.DataFrame([{"game_id": s["game_id"], "season": season, "game_type": s["game_type"], "game_date": pd.Timestamp(date),
                         "game_time": pd.Timestamp(s["start_utc"]) if s["start_utc"] else pd.Timestamp(date, tz="UTC"),
                         "home_team_abbr": s["home"], "away_team_abbr": s["away"], "home_fr": fr(s["home"]), "away_fr": fr(s["away"]),
                         "home_score": np.nan, "away_score": np.nan, "home_win": np.nan, "went_ot": False} for s in slate])
    allg = pd.concat([games, add], ignore_index=True)
    feats = F.build(games=allg, out=None)
    train = feats[(feats.game_type == "R") & feats.home_win.notna()]
    test = feats[feats.game_id.isin(add.game_id)]
    m = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=3000)).fit(train[FEATURES], train.home_win.astype(int))
    p = m.predict_proba(test[FEATURES])[:, 1]
    res = test[["game_id", "game_date", "away", "home", "elo_a", "elo_h", "a_rest", "h_rest", "a_b2b", "h_b2b"]].copy()
    res["p_home"] = p.round(3)
    res["pick"] = np.where(p >= 0.5, res.home, res.away)
    res["p_pick"] = np.where(p >= 0.5, p, 1 - p).round(3)
    res["tier"] = np.select([res.p_pick >= OFFICIAL, res.p_pick >= LEAN], ["OFFICIAL", "LEAN"], "PASS")
    res["game_date"] = res.game_date.dt.strftime("%Y-%m-%d")

    # No-edge inverse trial. Stamped on every new entry, flipped only inside the
    # window and only for PASS. `other` is taken from the model's own pick before
    # anything is overwritten, and `p_pick` is inverted so the stored probability
    # is still the probability of the stored pick.
    res["pick_rule"] = "model"
    _trial = res.game_date.isin(NOEDGE_INVERSE_DATES) & (res.tier == "PASS")
    if _trial.any():
        _other = np.where(res.pick == res.home, res.away, res.home)
        _inverted = (1 - res.p_pick).round(3)
        res.loc[_trial, "pick"] = _other[_trial.values]
        res.loc[_trial, "p_pick"] = _inverted[_trial]
        res.loc[_trial, "pick_rule"] = "noedge_inverse_trial"

    res["start_utc"] = res.game_id.map({s_["game_id"]: s_["start_utc"] for s_ in slate})
    res = res.sort_values("p_pick", ascending=False)

    # ---- ledger: history-first, freeze at first write, no back-fill after puck drop ----
    out = HERE / "picks" / f"{date}.json"; out.parent.mkdir(exist_ok=True)
    existing = {r["game_id"]: r for r in json.load(open(out))} if out.exists() else {}
    now = dt.datetime.now(dt.timezone.utc)
    final = []
    for r in res.round({"elo_a": 0, "elo_h": 0}).to_dict("records"):
        if r["game_id"] in existing:
            final.append(existing[r["game_id"]]); continue           # frozen: never recomputed
        st = r.get("start_utc")
        if st and pd.Timestamp(st) <= pd.Timestamp(now):
            r.update(pick=None, p_pick=None, tier="NO PICK", locked_at=None, note="game already started at first run")
        else:
            r["locked_at"] = now.strftime("%Y-%m-%dT%H:%M:%SZ")
        r["result"] = None
        final.append(r)
    json.dump(final, open(out, "w"), indent=1, default=str)
    show = pd.DataFrame(final)
    cols = ["away", "home", "pick", "p_pick", "tier", "locked_at", "a_rest", "h_rest", "elo_a", "elo_h"]
    print(f"\nNHL picks for {date}  (trained on {len(train):,} games through {games.game_date.max().date()})\n")
    print(show[[c for c in cols if c in show]].to_string(index=False))
    n_new = sum(1 for r in final if r["game_id"] not in existing)
    print(f"\n{n_new} new picks locked, {len(final) - n_new} already frozen -> {out}")


FINAL_STATES = ("OFF", "FINAL")   # api-web gameState of a finished game; FUT/PRE/LIVE/CRIT are not


def fetch_scores(date):
    """Raw api-web score payload for one date (YYYY-MM-DD)."""
    req = urllib.request.Request(f"https://api-web.nhle.com/v1/score/{date}",
                                 headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/128 Safari/537.36",
                                          "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.load(r)


def alert(msg, log):
    """One ALERT line into cron.log, where the runbook says to look (same convention as nhl_verify_additive.py)."""
    line = f"ALERT {dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')} nhl_predict.py grade: {msg}"
    with open(log, "a") as fh:
        fh.write(line + "\n")
    if sys.stderr.isatty():
        print(line, file=sys.stderr)


def _parquet_lookup(root):
    games = pd.read_parquet(root / "games.parquet")
    key = games.set_index(["game_date", "home_team_abbr", "away_team_abbr"])[["home_score", "away_score", "home_win"]]

    def lookup(r):
        k = (pd.Timestamp(r["game_date"]), r["home"], r["away"])
        if k not in key.index:
            return None
        g = key.loc[k]; winner = r["home"] if g.home_win == 1 else r["away"]
        return winner, f"{int(g.away_score)}-{int(g.home_score)}"
    return lookup


def _api_lookup(ledgers, fetch, log):
    """Finals from api-web for every date (up to today) that still has an ungraded pick, keyed on game_id."""
    today = dt.datetime.now(dt.timezone.utc).date().isoformat()
    dates = sorted({r["game_date"] for _, picks in ledgers for r in picks
                    if r.get("pick") and r.get("result") is None and r["game_date"] <= today})
    finals = {}
    for d in dates:
        try:
            payload = fetch(d)
        except Exception as e:
            print("skip", d, "->", type(e).__name__); continue
        for gm in payload.get("games", []):
            if gm.get("gameState") in FINAL_STATES:
                finals[gm["id"]] = gm

    def lookup(r):
        gm = finals.get(r["game_id"])
        if gm is None:
            return None
        a, h = gm["awayTeam"], gm["homeTeam"]
        if (a.get("abbrev"), h.get("abbrev")) != (r["away"], r["home"]) or a.get("score") is None or h.get("score") is None \
                or a["score"] == h["score"]:
            alert(f"game {r['game_id']} api final {a.get('abbrev')} {a.get('score')} @ {h.get('abbrev')} {h.get('score')} "
                  f"does not fit ledger {r['away']}@{r['home']}; not graded", log)
            return None
        winner = r["home"] if h["score"] > a["score"] else r["away"]
        return winner, f"{int(a['score'])}-{int(h['score'])}"
    return lookup


def grade(source="parquet", root=HERE, fetch=fetch_scores):
    """Score every locked pick that has a final result; print the record by tier and write ledger.csv.

      python3 nhl_predict.py grade                  # results from games.parquet (default; the reconciliation pass)
      python3 nhl_predict.py grade --source api     # finals from api-web.nhle.com/v1/score/<date>, gameState OFF/FINAL only

    Either source fills result and final only where result is still null. A source that disagrees with a result
    already recorded does not overwrite it: it writes ALERT to cron.log and leaves the entry as it is.
    """
    root = Path(root); log = root / "cron.log"
    ledgers = [(f, json.load(open(f))) for f in sorted((root / "picks").glob("*.json"))]
    lookup = _api_lookup(ledgers, fetch, log) if source == "api" else _parquet_lookup(root)
    rows = []
    for f, picks in ledgers:
        changed = False
        for r in picks:
            got = lookup(r) if r.get("pick") else None
            if got:
                winner, final = got; result = "W" if r["pick"] == winner else "L"
                if r.get("result") is None:
                    r["result"] = result; r["final"] = final; changed = True
                elif r["result"] != result:
                    alert(f"{f.name} game {r['game_id']} {r['away']}@{r['home']} is graded {r['result']} {r.get('final')}, "
                          f"{source} says {result} {final}; left as recorded", log)
            rows.append(r)
        if changed: json.dump(picks, open(f, "w"), indent=1, default=str)
    L = pd.DataFrame(rows)
    if L.empty: print("no picks yet"); return
    L.to_csv(root / "ledger.csv", index=False)
    done = L[L.result.notna()]
    for tier in ["OFFICIAL", "LEAN"]:
        t = done[done.tier == tier]; w = int((t.result == "W").sum()); l = len(t) - w
        print(f"{tier:9s} {w}-{l}" + (f"  {w/len(t):.1%}" if len(t) else ""))
    print(f"pending: {int(L.result.isna().sum() - (L.tier == 'NO PICK').sum())}  |  ledger.csv written")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["refresh", "predict", "grade"])
    ap.add_argument("--date", default=dt.date.today().isoformat())
    ap.add_argument("--games", help="manual slate AWAY@HOME,AWAY@HOME")
    ap.add_argument("--source", choices=["parquet", "api"], default="parquet", help="grade: where finals come from")
    a = ap.parse_args()
    if a.cmd == "refresh":
        refresh()
    elif a.cmd == "grade":
        grade(a.source)
    else:
        predict(dt.date.fromisoformat(a.date), a.games)
