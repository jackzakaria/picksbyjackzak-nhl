# NHL game model (Picks by JackZak)

Walk-forward validated logistic regression, 10 pregame features, no goalie or odds inputs.
Backtest 2013-14 .. 2025-26 regular season (15,972 games, trained only on prior seasons):
  OFFICIAL (prob >= 0.65) 2043-879 = 69.9%, ~225/season, cleared 65% in all 13 seasons (low 65.4%, high 76.2%)
  LEAN (0.55-0.65)        4572-3118 = 59.5%, ~590/season   |   below 0.55: PASS, no pick   |   every game: 58.9%

Tiers match the NBA page: OFFICIAL counts toward the record, LEAN is shown but not counted, PASS is not shown.

Files
  build_dataset.py   merges data/{game_info,team_box,goalie_box,linescore,schedules} -> games.parquet
  features.py        point-in-time features (Elo, EWMA form, rest/travel, goalies) -> features.parquet
  backtest.py        walk-forward backtest, tiers, filter curve -> backtest_preds.parquet, backtest_summary.json
  nhl_predict.py     refresh | predict | grade
                     predict -> picks/YYYY-MM-DD.json, one entry per game with pick, p_pick, tier, locked_at
                     LEDGER: a pick is written once and never recomputed; rerunning a date is a no-op for existing games;
                     a game already started at first run gets NO PICK (never back-filled)
                     grade   -> fills result W/L on locked picks from finished games, prints OFFICIAL and LEAN records, writes ledger.csv
  data/              per-season parquets from github.com/sportsdataverse/fastRhockey-nhl-data (updated daily)

Setup:  pip install pandas pyarrow scikit-learn lightgbm
Daily cron (server, ET):
  0 9  * * *  cd /root/nhl && .venv/bin/python nhl_predict.py refresh && .venv/bin/python nhl_predict.py grade && .venv/bin/python nhl_predict.py predict   # lock today's slate
  0 12 * * *  cd /root/nhl && .venv/bin/python nhl_predict.py predict    # catches games added late; existing picks stay frozen
No goalie inputs, so 9am lock is safe. The NHL API needs a browser User-Agent (already set).
Season file naming is by season END year (2026-27 = *_2027.parquet); refresh pulls current + previous year.
