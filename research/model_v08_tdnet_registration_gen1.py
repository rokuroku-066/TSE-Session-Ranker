#!/usr/bin/env python3
"""Zero-based, title-only TDnet hypothesis screen (scratch artifact)."""

from __future__ import annotations

import gc
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

ROOT = Path("/workspace/scratch/8678b1d14f37")
for p in (ROOT / "src", ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from research.screen_feature_candidates_v06 import (  # noqa: E402
    _anchor_spec,
    _daily_net,
    evaluate_recipe,
    max_t_adjusted_uplifts,
)
from tse_session_ranker.data.common import normalize_expected_sessions  # noqa: E402
from tse_session_ranker.data.tdnet import collect_tdnet_dataset  # noqa: E402
from tse_session_ranker.profit import daily_portfolio_returns  # noqa: E402
from tse_session_ranker.research_candidates import (  # noqa: E402
    build_clean_tdnet_candidate_features,
)

OUT = Path("/tmp/zero_tdnet_text")
PRIMARY_COST_BPS = 20.0
BASE_PANEL = Path("/tmp/model_v05_broad_family_panel_r2.pkl")
TDNET_CACHE = ROOT / "research/.cache/model_v05_tdnet"


def net20(picks: pd.DataFrame) -> pd.Series:
    return daily_portfolio_returns(
        picks, top_k=2, cost_bps=PRIMARY_COST_BPS
    ).set_index("date")["net_return_pct"].sort_index()


def summary(picks: pd.DataFrame) -> dict:
    daily = net20(picks)
    gross = daily_portfolio_returns(
        picks, top_k=2, cost_bps=0.0
    ).set_index("date")["net_return_pct"].sort_index()
    month = daily.groupby(daily.index.to_period("M")).mean()
    trimmed = daily.drop(daily.nlargest(min(5, len(daily))).index)
    return {
        "days": int(len(daily)),
        "gross_mean_pct": float(gross.mean()),
        "net20_mean_pct": float(daily.mean()),
        "net20_median_pct": float(daily.median()),
        "net20_positive_day_fraction": float((daily > 0).mean()),
        "net20_positive_month_fraction": float((month > 0).mean()),
        "net20_months": {str(k): float(v) for k, v in month.items()},
        "net20_top5_removed_pct": float(trimmed.mean()) if len(trimmed) else None,
    }


def changed_slots(base: pd.DataFrame, cand: pd.DataFrame) -> dict:
    b = base[["date", "model_rank", "code"]].rename(columns={"code": "base"})
    c = cand[["date", "model_rank", "code"]].rename(columns={"code": "cand"})
    z = b.merge(c, on=["date", "model_rank"], validate="one_to_one")
    changed = z["base"].fillna("").ne(z["cand"].fillna(""))
    return {
        "slots": int(changed.sum()),
        "days": int(z.loc[changed, "date"].nunique()),
        "fraction": float(changed.mean()),
    }


def exposure(frame: pd.DataFrame, cols: list[str], start: str, end: str) -> dict:
    x = frame.loc[
        frame["date"].between(start, end)
        & frame["tdnet_source_complete"].eq(True),
        ["date", *cols],
    ].copy()
    numeric = x[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    active = numeric.abs().gt(1e-12).any(axis=1)
    return {
        "rows": int(active.sum()),
        "days": int(x.loc[active, "date"].nunique()),
    }


def add_derived(frame: pd.DataFrame) -> pd.DataFrame:
    # Every added name starts tdnet_clean_ so the frozen evaluator applies the
    # TDnet source-completeness mask.  Values are based only on title bundles
    # assigned at or before 08:58:59 JST and strictly prior price features.
    f = frame
    fresh = f["tdnet_v07_fresh_classified_economic_any"]
    follow = f["tdnet_v07_followup_family_count"].gt(0).astype(float)
    econ = f["tdnet_v07_economic_family_count"]
    f["tdnet_clean_z_fresh_only"] = fresh * (1.0 - follow)
    f["tdnet_clean_z_followup_only"] = follow * (1.0 - fresh)
    f["tdnet_clean_z_fresh_followup_both"] = fresh * follow
    f["tdnet_clean_z_econ_mixed2plus"] = econ.ge(2).astype(float)
    f["tdnet_clean_z_econ_mixed3plus"] = econ.ge(3).astype(float)
    width = f["tdnet_clean_bundle_width_minutes_log1p"]
    docs = f["tdnet_clean_document_count_log1p"]
    f["tdnet_clean_z_bundle_intensity"] = docs / (1.0 + width)
    f["tdnet_clean_z_repeat_pressure"] = (
        f["tdnet_clean_prior_bundle_count_60_log1p"]
        - np.log1p(60.0 / 252.0)
        - f["tdnet_clean_prior_bundle_count_252_log1p"]
    )
    momentum20 = f["xrank_close_momentum_20"]
    momentum60 = f["xrank_close_momentum_60"]
    f["tdnet_clean_z_fresh_x_mom20"] = fresh * momentum20
    f["tdnet_clean_z_fresh_x_mom60"] = fresh * momentum60
    f["tdnet_clean_z_followup_x_mom20"] = follow * momentum20
    f["tdnet_clean_z_followup_x_mom60"] = follow * momentum60
    f["tdnet_clean_z_fresh_x_negative_mom20"] = fresh * (-momentum20).clip(lower=0)
    f["tdnet_clean_z_fresh_x_positive_mom20"] = fresh * momentum20.clip(lower=0)
    f["tdnet_clean_z_fresh_x_premarket"] = fresh * f["tdnet_clean_premarket_count_log1p"]
    f["tdnet_clean_z_fresh_x_postclose"] = fresh * f["tdnet_clean_postclose_count_log1p"]
    f["tdnet_clean_z_fresh_x_age"] = fresh * f["tdnet_clean_latest_age_hours_log1p"]
    f["tdnet_clean_z_followup_x_age"] = follow * f["tdnet_clean_latest_age_hours_log1p"]
    return f


def main() -> None:
    manifest = json.loads((ROOT / "research/model_v05_panel_manifest.json").read_text())
    sessions = normalize_expected_sessions(manifest["sessions"])
    catalog = json.loads((ROOT / "research/model_v06_feature_catalog.json").read_text())
    protocol = json.loads((ROOT / "research/model_v06_feature_protocol.json").read_text())
    g0 = list(catalog["historical_screen"]["G0_price_core"]["features"])
    needed = list(dict.fromkeys([
        "date", "code", "name", "label", "oc_return_pct", "outcome_observed",
        "source_complete", "universe_source_complete", "price_eligible",
        "price_training_eligible", "tdnet_source_complete", *g0,
    ]))
    base = pd.read_pickle(BASE_PANEL)[needed].copy()
    dataset = collect_tdnet_dataset(TDNET_CACHE, allow_historical_provenance=True)
    feat = build_clean_tdnet_candidate_features(
        dataset.disclosures, sessions, decision_time="08:58:59"
    )
    panel = base.merge(feat, on=["date", "code"], how="left", validate="many_to_one")
    complete = panel["tdnet_source_complete"].eq(True)
    candidate_cols = [c for c in feat if c.startswith("tdnet_") and c != "tdnet_clean_feature_source_max_timestamp"]
    panel.loc[complete, candidate_cols] = panel.loc[complete, candidate_cols].fillna(0.0)
    panel.loc[~complete, candidate_cols] = np.nan
    panel = add_derived(panel)

    # Twelve outcome-blind semantic hypotheses.  Each is evaluated as a single
    # addition to the exact same G0/raked-logit anchor.
    hypotheses: dict[str, list[str]] = {
        "H01_stage_fresh_followup": [
            "tdnet_v07_fresh_classified_economic_any",
            "tdnet_v07_followup_family_count_log1p",
            "tdnet_v07_has_progress_stage",
        ],
        "H02_forecast_stage": [
            "tdnet_v07_has_forecast_initial",
            "tdnet_v07_has_forecast_revision",
        ],
        "H03_dividend_role": [
            "tdnet_v07_has_shareholder_dividend",
            "tdnet_v07_has_received_dividend",
            "tdnet_v07_has_intercompany_dividend",
            "tdnet_v07_has_subsidiary_dividend",
        ],
        "H04_capital_action_stage": [
            "tdnet_v07_has_fresh_buyback", "tdnet_v07_has_followup_buyback",
            "tdnet_v07_has_fresh_equity", "tdnet_v07_has_followup_equity",
            "tdnet_v07_has_fresh_share_cancellation",
            "tdnet_v07_has_followup_share_cancellation",
        ],
        "H05_ma_direction_stage": [
            "tdnet_v07_has_fresh_ma", "tdnet_v07_has_followup_ma",
            "tdnet_v07_has_ma_acquisition", "tdnet_v07_has_ma_divestiture",
            "tdnet_v07_has_ma_reorganization",
            "tdnet_v07_has_ma_internal_reorganization",
        ],
        "H06_economic_family_mix": [
            "tdnet_v07_economic_family_count_log1p",
            "tdnet_v07_single_economic_family",
            "tdnet_clean_z_econ_mixed2plus", "tdnet_clean_z_econ_mixed3plus",
        ],
        "H07_stage_cooccurrence": [
            "tdnet_clean_z_fresh_only", "tdnet_clean_z_followup_only",
            "tdnet_clean_z_fresh_followup_both",
        ],
        "H08_arrival_timing": [
            "tdnet_clean_premarket_count_log1p",
            "tdnet_clean_intraday_count_log1p",
            "tdnet_clean_postclose_count_log1p",
            "tdnet_clean_latest_age_hours_log1p", "tdnet_clean_weekend_age",
        ],
        "H09_bundle_density_cadence": [
            "tdnet_clean_document_count_log1p",
            "tdnet_clean_bundle_width_minutes_log1p",
            "tdnet_clean_z_bundle_intensity",
            "tdnet_clean_prior_bundle_count_60_log1p",
            "tdnet_clean_prior_bundle_count_252_log1p",
            "tdnet_clean_z_repeat_pressure",
        ],
        "H10_stage_linear_runup": [
            "tdnet_clean_z_fresh_x_mom20", "tdnet_clean_z_fresh_x_mom60",
            "tdnet_clean_z_followup_x_mom20", "tdnet_clean_z_followup_x_mom60",
        ],
        "H11_fresh_runup_hinges": [
            "tdnet_v07_fresh_classified_economic_any",
            "tdnet_clean_z_fresh_x_negative_mom20",
            "tdnet_clean_z_fresh_x_positive_mom20",
        ],
        "H12_stage_timing_interaction": [
            "tdnet_clean_z_fresh_x_premarket",
            "tdnet_clean_z_fresh_x_postclose",
            "tdnet_clean_z_fresh_x_age", "tdnet_clean_z_followup_x_age",
        ],
    }

    split = {
        "exploration": (pd.Timestamp("2024-07-01"), pd.Timestamp("2024-12-30")),
        "confirmation": (pd.Timestamp("2025-01-06"), pd.Timestamp("2025-03-31")),
    }
    spec = _anchor_spec(protocol)
    results: dict[str, dict] = {"hypotheses": hypotheses, "periods": {}}
    pick_store: dict[tuple[str, str], pd.DataFrame] = {}
    for period_name, (start, end) in split.items():
        period_results: dict[str, dict] = {}
        base_result, base_picks = evaluate_recipe(
            panel, sessions, spec, recipe_id="G0", columns=g0,
            start=start, end=end, train_start=pd.Timestamp("2024-01-04"),
            minimum_training_sessions=60,
        )
        pick_store[(period_name, "G0")] = base_picks
        period_results["G0"] = {"summary": summary(base_picks), "raw": base_result}
        candidate_picks = {}
        for hid, cols in hypotheses.items():
            result, picks = evaluate_recipe(
                panel, sessions, spec, recipe_id=f"G0+{hid}",
                columns=[*g0, *cols], start=start, end=end,
                train_start=pd.Timestamp("2024-01-04"), minimum_training_sessions=60,
            )
            pick_store[(period_name, hid)] = picks
            candidate_picks[hid] = picks
            ex = exposure(panel, cols, str(start.date()), str(end.date()))
            period_results[hid] = {
                "features": cols,
                "event_exposure": ex,
                "summary": summary(picks),
                "uplift_net20_pct": float((net20(picks) - net20(base_picks)).mean()),
                "changed": changed_slots(base_picks, picks),
                "raw": result,
            }
            print(period_name, hid, period_results[hid]["uplift_net20_pct"], flush=True)
        adjusted = max_t_adjusted_uplifts(base_picks, candidate_picks)
        for hid, vals in adjusted.items():
            period_results[hid]["max_t_adjusted"] = vals
        results["periods"][period_name] = period_results
        del candidate_picks
        gc.collect()

    # A condition was preregistered before outcomes: minimum 30 active event
    # days in both periods; positive uplift in both; positive absolute net20 in
    # confirmation; confirmation adjusted 80% lower bound >= 0; at least 50
    # changed slots in confirmation.  This deliberately prevents a tiny sparse
    # subgroup or unchanged model from looking successful.
    decisions = {}
    for hid in hypotheses:
        e = results["periods"]["exploration"][hid]
        c = results["periods"]["confirmation"][hid]
        checks = {
            "exploration_event_days_ge_30": e["event_exposure"]["days"] >= 30,
            "confirmation_event_days_ge_30": c["event_exposure"]["days"] >= 30,
            "exploration_uplift_positive": e["uplift_net20_pct"] > 0,
            "confirmation_uplift_positive": c["uplift_net20_pct"] > 0,
            "confirmation_absolute_net20_positive": c["summary"]["net20_mean_pct"] > 0,
            "confirmation_adjusted_lower_nonnegative": c["max_t_adjusted"]["adjusted_one_sided_80pct_lower_pct"] >= 0,
            "confirmation_changed_slots_ge_50": c["changed"]["slots"] >= 50,
        }
        decisions[hid] = {
            "checks": checks,
            "qualified": all(checks.values()),
        }
    results["decision_rule"] = {
        "minimum_event_days_each_period": 30,
        "minimum_confirmation_changed_slots": 50,
        "requires_positive_exploration_and_confirmation_uplift": True,
        "requires_positive_confirmation_absolute_net20": True,
        "requires_confirmation_max_t_adjusted_80pct_lower_nonnegative": True,
    }
    results["decisions"] = decisions
    results["qualified"] = [k for k, v in decisions.items() if v["qualified"]]
    results["limits"] = [
        "All dates are retrospectively outcome-known; confirmation is temporal but not sealed.",
        "TDnet inputs contain titles and timestamps only, not PDF body quantities.",
        "No historical board, PTS, exact indicative open, futures, volume or turnover is available.",
        "The screen changes only research features; production and frozen artifacts are untouched.",
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "result.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = []
    for hid in hypotheses:
        row = {"hypothesis": hid, "qualified": decisions[hid]["qualified"]}
        for period_name in split:
            r = results["periods"][period_name][hid]
            row.update({
                f"{period_name}_event_days": r["event_exposure"]["days"],
                f"{period_name}_net20": r["summary"]["net20_mean_pct"],
                f"{period_name}_uplift": r["uplift_net20_pct"],
                f"{period_name}_adjusted_lower": r["max_t_adjusted"]["adjusted_one_sided_80pct_lower_pct"],
                f"{period_name}_changed_slots": r["changed"]["slots"],
                f"{period_name}_positive_month_fraction": r["summary"]["net20_positive_month_fraction"],
                f"{period_name}_top5_removed": r["summary"]["net20_top5_removed_pct"],
            })
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT / "summary.csv", index=False)
    # Persist all candidate picks for individual error inspection.
    pick_rows = []
    for (period_name, recipe), picks in pick_store.items():
        x = picks.copy()
        x.insert(0, "period", period_name)
        x.insert(1, "hypothesis", recipe)
        pick_rows.append(x)
    pd.concat(pick_rows, ignore_index=True).to_csv(OUT / "picks.csv", index=False)
    print(json.dumps({"qualified": results["qualified"], "output": str(OUT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
