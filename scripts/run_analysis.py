"""Walk-forward evaluation of the state-space model against Elo and the market.

    python scripts/run_analysis.py           # full run (~3-5 min on a laptop)
    python scripts/run_analysis.py --fast    # reuse committed yearly hyperparameters

Design
------
* 2002-2005  burn-in (ratings move, nothing is scored)
* 2006-2014  initial training window
* 2015-2025  test: for each season S, hyperparameters are re-estimated on
             2006..S-1 only, then the filter runs through S week by week.
Writes tables to results/ and figures to figures/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from nflstate import plots
from nflstate.elo import run_elo, tune_elo
from nflstate.evaluate import (BREAKEVEN_110, ats_by_disagreement, block_bootstrap_diff, brier,
                               calibration_table, log_loss, pointwise_log_loss)
from nflstate.kalman import Params, cover_probability, offsets, run_filter
from nflstate.market import fit_spread_sd, market_prob, overround
from nflstate.model import fit_params

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "games.parquet"
RES, FIG = ROOT / "results", ROOT / "figures"
TRAIN_START, TEST_START, LAST = 2006, 2015, 2025


def walk_forward(games: pd.DataFrame, fast: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Yearly re-fit of hyperparameters on past seasons; predict the next season."""
    cache = RES / "walkforward_params.csv"
    if fast and cache.exists():
        table = pd.read_csv(cache)
    else:
        rows, base = [], Params()
        for s in range(TEST_START, LAST + 1):
            p, info = fit_params(games, TRAIN_START, s - 1, base=base)
            base = p  # warm start next year
            rows.append({"test_season": s, **p.as_dict(), "train_loglik": info["loglik"],
                         "train_games": info["n_games"], "converged": info["converged"]})
            print(f"  {s}: sigma={p.sigma:.2f} q_week={p.q_week:.3f} rho={p.rho:.3f} "
                  f"q_season={p.q_season:.2f} q_hfa={p.q_hfa:.4f} beta_qb={p.beta_qb:.2f} "
                  f"beta_rest={p.beta_rest:.2f}")
        table = pd.DataFrame(rows)
    preds = []
    fields = list(Params.__dataclass_fields__)
    for row in table.itertuples(index=False):
        p = Params(**{f: getattr(row, f) for f in fields})
        out = run_filter(games[games["season"] <= row.test_season], p).games
        preds.append(out[out["season"] == row.test_season])
    return pd.concat(preds, ignore_index=True), table


def main(fast: bool = False) -> None:
    RES.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    games = pd.read_parquet(DATA)
    print(f"{len(games):,} games {games.season.min()}-{games.season.max()}")

    # ---------------- state-space model ----------------
    print("walk-forward hyperparameter fits")
    wf, wf_params = walk_forward(games, fast)
    wf_params.to_csv(RES / "walkforward_params.csv", index=False)

    # frozen variants (tuned once on 2006-2014) for an ablation
    p_frozen, _ = fit_params(games, TRAIN_START, TEST_START - 1, base=Params())
    p_noqb, _ = fit_params(games, TRAIN_START, TEST_START - 1, base=Params(),
                           free=["sigma", "q_week", "rho", "q_season", "q_hfa"])
    frozen = run_filter(games, p_frozen).games
    noqb = run_filter(games, p_noqb).games

    # ---------------- baselines ----------------
    elo_p = tune_elo(games, TRAIN_START, TEST_START - 1)
    elo = run_elo(games, elo_p)
    train = games[games["season"].between(TRAIN_START, TEST_START - 1) & (games["result"] != 0)]
    spread_sd = fit_spread_sd(train["spread_line"], train["result"] > 0)
    home_rate = float((train["result"] > 0).mean())
    mkt = market_prob(games, spread_sd)

    # ---------------- assemble test set ----------------
    key = ["game_id"]
    test = wf.merge(elo[key + ["elo_p_home", "elo_spread"]], on=key)
    test = test.merge(mkt[key + ["mkt_p_home", "mkt_source"]], on=key)
    test = test.merge(frozen[key + ["p_home", "pred_margin"]].rename(
        columns={"p_home": "frozen_p_home", "pred_margin": "frozen_margin"}), on=key)
    test = test.merge(noqb[key + ["p_home", "pred_margin"]].rename(
        columns={"p_home": "noqb_p_home", "pred_margin": "noqb_margin"}), on=key)
    test["p_cover"] = cover_probability(test["pred_margin"], test["pred_sd"], test["spread_line"])
    test.to_parquet(RES / "test_predictions.parquet", index=False)

    sc = test[test["result"] != 0]
    y = (sc["result"] > 0).to_numpy(int)
    forecasters = {
        "Home team, base rate": np.full(len(sc), home_rate),
        "Elo (538-style)": sc["elo_p_home"].to_numpy(),
        "State-space, no QB/rest": sc["noqb_p_home"].to_numpy(),
        "State-space, frozen 2006-14 params": sc["frozen_p_home"].to_numpy(),
        "State-space (walk-forward)": sc["p_home"].to_numpy(),
        "Market (no-vig closing)": sc["mkt_p_home"].to_numpy(),
    }
    spreads = {
        "Home team, base rate": np.full(len(sc), float(train["result"].mean())),
        "Elo (538-style)": sc["elo_spread"].to_numpy(),
        "State-space, no QB/rest": sc["noqb_margin"].to_numpy(),
        "State-space, frozen 2006-14 params": sc["frozen_margin"].to_numpy(),
        "State-space (walk-forward)": sc["pred_margin"].to_numpy(),
        "Market (no-vig closing)": sc["spread_line"].to_numpy(),
    }
    acc = {k: float(np.mean((v > 0.5) == y)) for k, v in forecasters.items()}
    metrics = pd.DataFrame({
        "forecaster": list(forecasters),
        "log_loss": [log_loss(y, p) for p in forecasters.values()],
        "brier": [brier(y, p) for p in forecasters.values()],
        "accuracy": [acc[k] for k in forecasters],
        "spread_mae_vs_result": [float(np.mean(np.abs(spreads[k] - sc["result"]))) for k in forecasters],
        "spread_mae_vs_line": [float(np.mean(np.abs(spreads[k] - sc["spread_line"]))) for k in forecasters],
    })
    metrics.to_csv(RES / "metrics.csv", index=False)
    print(metrics.round(4).to_string(index=False))

    weeks = (sc["season"] * 100 + sc["week"]).to_numpy()
    ll_model = pointwise_log_loss(y, sc["p_home"].to_numpy())
    ll_mkt = pointwise_log_loss(y, sc["mkt_p_home"].to_numpy())
    ll_elo = pointwise_log_loss(y, sc["elo_p_home"].to_numpy())
    d_mkt = block_bootstrap_diff(ll_model, ll_mkt, weeks)
    d_elo = block_bootstrap_diff(ll_model, ll_elo, weeks)

    # ---- against the spread ----
    ats = ats_by_disagreement(test)
    ats.to_csv(RES / "ats_by_disagreement.csv", index=False)
    nopush = test[test["result"] != test["spread_line"]]
    yc = (nopush["result"] > nopush["spread_line"]).to_numpy(int)
    cover_ll = log_loss(yc, nopush["p_cover"].to_numpy())
    allside = ats_by_disagreement(test, edges=(0, np.inf)).iloc[0]
    big = ats_by_disagreement(test, edges=(3, np.inf)).iloc[0]

    # blend check: does the model add information beyond the line? (descriptive OLS on test)
    X = np.c_[np.ones(len(sc)), sc["spread_line"], sc["pred_margin"] - sc["spread_line"]]
    beta, *_ = np.linalg.lstsq(X, sc["result"].to_numpy(float), rcond=None)
    resid = sc["result"].to_numpy(float) - X @ beta
    cov = np.linalg.inv(X.T @ X) * resid.var(ddof=3)
    edge_coef, edge_se = float(beta[2]), float(np.sqrt(cov[2, 2]))

    # what are the big disagreements? final regular-season week / backup QBs
    final_week = (test["game_type"] == "REG") & (test["week"] == test["season"].map(
        test[test["game_type"] == "REG"].groupby("season")["week"].max()))
    abs_edge = (test["pred_margin"] - test["spread_line"]).abs()
    big_mask = abs_edge >= 4
    anyqb = (test["home_qb_change"] + test["away_qb_change"]) > 0
    big_profile = {
        "n_edge_ge4": int(big_mask.sum()),
        "share_final_week_edge_ge4": float(final_week[big_mask].mean()),
        "share_final_week_all": float(final_week.mean()),
        "share_qb_change_edge_ge4": float(anyqb[big_mask].mean()),
        "share_qb_change_all": float(anyqb.mean()),
    }

    # biggest disagreements
    t = test.assign(edge=test["pred_margin"] - test["spread_line"])
    t = t.reindex(t["edge"].abs().sort_values(ascending=False).index).head(12)
    t["model_side"] = np.where(t["edge"] > 0, t["home_team"], t["away_team"])
    t["covered"] = np.where(t["result"] == t["spread_line"], "push",
                            np.where(np.sign(t["result"] - t["spread_line"]) == np.sign(t["edge"]), "yes", "no"))
    top = t[["season", "week", "away_team", "home_team", "away_qb_name", "home_qb_name",
             "spread_line", "pred_margin", "result", "model_side", "covered",
             "home_qb_change", "away_qb_change"]].round(1)
    top.to_csv(RES / "top_disagreements.csv", index=False)

    # ---------------- descriptive (full-sample) smoother ----------------
    p_all, _ = fit_params(games, TRAIN_START, LAST, base=wf_params.iloc[-1:].pipe(
        lambda d: Params(**{f: float(d[f].iloc[0]) for f in Params.__dataclass_fields__})))
    full = run_filter(games, p_all, store=True)
    m_s, sd_s = full.smooth()
    m_f, sd_f = full.filtered()
    steps = full.steps.copy()
    steps["t"] = steps["season"] + (steps["week"] - 1) / 22
    T = len(full.teams)
    z90 = 1.645
    hrow = []
    idx = {t: i for i, t in enumerate(full.teams)}
    step_of = {(a, b): k for k, (a, b) in enumerate(zip(steps["season"], steps["week"]))}
    g2 = full.games
    for s in range(TRAIN_START, LAST + 1):
        i_end = np.flatnonzero(steps["season"] == s)[-1]
        i_mid = np.flatnonzero(steps["season"] == s)[0]
        gs = g2[(g2["season"] == s) & (g2["neutral"] == 0)]
        # season-only estimate: residual of the margin after subtracting smoothed team strengths
        k = np.array([step_of[(a, b)] for a, b in zip(gs["season"], gs["week"])])
        diff = m_s[k, gs["home_team"].map(idx)] - m_s[k, gs["away_team"].map(idx)]
        r = gs["result"].to_numpy() - diff - offsets(gs, p_all)
        hrow.append({"season": s, "smooth": m_s[i_mid, T], "smooth_lo": m_s[i_mid, T] - z90 * sd_s[i_mid, T],
                     "smooth_hi": m_s[i_mid, T] + z90 * sd_s[i_mid, T], "filtered_end": m_f[i_end, T],
                     "raw": float(r.mean()), "raw_se": float(r.std(ddof=1) / np.sqrt(len(r))),
                     "home_win_rate": float((gs["result"] > 0).mean())})
    hfa_tab = pd.DataFrame(hrow)
    def pooled(lo, hi):
        d = hfa_tab[hfa_tab["season"].between(lo, hi)]
        w = 1 / d["raw_se"] ** 2
        return {"mean": float((w * d["raw"]).sum() / w.sum()), "se": float(np.sqrt(1 / w.sum()))}
    hfa_periods = {"2006-2018": pooled(2006, 2018), "2019": pooled(2019, 2019), "2020": pooled(2020, 2020),
                   "2021-2025": pooled(2021, 2025)}
    hfa_tab.to_csv(RES / "hfa_by_season.csv", index=False)

    traj = []
    for team in ("KC", "NE", "DET", "SF"):
        j = full.teams.index(team)
        sel = steps["season"] >= TRAIN_START
        traj.append(pd.DataFrame({"team": team, "t": steps.loc[sel, "t"], "mean": m_s[sel, j],
                                  "lo": m_s[sel, j] - z90 * sd_s[sel, j], "hi": m_s[sel, j] + z90 * sd_s[sel, j]}))
    traj = pd.concat(traj)
    final = pd.DataFrame({"team": full.teams, "rating": m_f[-1, :T], "sd": sd_f[-1, :T]}).sort_values(
        "rating", ascending=False)
    final.round(2).to_csv(RES / "ratings_end_2025.csv", index=False)

    # ---------------- figures ----------------
    plots.calibration({
        "State-space model": calibration_table(y, sc["p_home"].to_numpy()),
        "Market (no-vig)": calibration_table(y, sc["mkt_p_home"].to_numpy()),
        "Elo": calibration_table(y, sc["elo_p_home"].to_numpy()),
    }, FIG / "calibration.png")
    plots.hfa(hfa_tab, FIG / "hfa_over_time.png")
    plots.ratings(traj, ["KC", "NE", "DET", "SF"], FIG / "rating_trajectories.png")
    plots.disagreement(ats, BREAKEVEN_110, FIG / "disagreement_ats.png")

    ml = test[test["mkt_source"] == "moneyline"]
    summary = {
        "n_test_games": int(len(test)), "n_test_scored": int(len(sc)), "n_ties_excluded": int(len(test) - len(sc)),
        "test_share_moneyline": float((test["mkt_source"] == "moneyline").mean()),
        "median_overround": float(np.median(overround(ml["home_moneyline"], ml["away_moneyline"]))),
        "spread_sd_market": spread_sd, "home_rate_train": home_rate, "elo_params": vars(elo_p),
        "params_frozen": p_frozen.as_dict(), "params_noqb": p_noqb.as_dict(), "params_full_sample": p_all.as_dict(),
        "logloss_model_minus_market": d_mkt, "logloss_model_minus_elo": d_elo,
        "cover_logloss_model": cover_ll, "cover_logloss_coin": float(np.log(2)),
        "ats_all": {"n": int(allside["n"]), "win_rate": float(allside["win_rate"]),
                    "lo": float(allside["lo"]), "hi": float(allside["hi"])},
        "ats_edge_ge3": {"n": int(big["n"]), "win_rate": float(big["win_rate"]),
                         "lo": float(big["lo"]), "hi": float(big["hi"])},
        "edge_regression": {"coef": edge_coef, "se": edge_se, "line_coef": float(beta[1])},
        "hfa_periods": hfa_periods,
        "big_disagreement_profile": big_profile,
        "corr_model_line": float(np.corrcoef(sc["pred_margin"], sc["spread_line"])[0, 1]),
    }
    (RES / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps({k: summary[k] for k in ("hfa_periods", "big_disagreement_profile", "params_full_sample", "logloss_model_minus_market", "logloss_model_minus_elo",
                                              "ats_all", "ats_edge_ge3", "edge_regression",
                                              "cover_logloss_model")}, indent=1))
    print(hfa_tab.round(2).to_string(index=False))
    print(top.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="reuse results/walkforward_params.csv")
    main(ap.parse_args().fast)
