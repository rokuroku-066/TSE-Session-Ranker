from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research import model_v17_liquidity_reliability_audit as audit
from research import model_v17_liquidity_reliability_runner as runner


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research"
DAILY_FORMAT = "jpx_stock_quotations_auction_regular_way_domestic_ordinary"
EXPECTED_PROTOCOL_SHA256 = (
    "f7d2efa30c5f6ca03a95e1f6e84e0fb6de3f68e8f0d520183877e2f0ab4a416f"
)
EXPECTED_CANDIDATES = (
    "PAIR01_TOP2_EXACT_LIQ_RERANK",
    "TW02_TURNOVER_WEIGHTED_PRICE_MEMORY",
    "VW03_AGGREGATE_COST_BASIS",
    "RPY04_RETURN_PER_TURNOVER_REVERSAL",
    "AR05_SECURITY_ACTIVITY_REGIME_EXPERTS",
    "PS06_PERSISTENT_VS_ISOLATED_ACTIVITY",
    "VP07_VWAP_RANGE_PRESSURE_MEMORY",
    "RW08_EXACT_LOT_RELIABILITY_WEIGHT",
    "MR09_TURNOVER_WEIGHTED_MARKET_REGIME",
    "AT10_ATTENTION_MIGRATION",
)


def _protocol() -> dict[str, object]:
    value = json.loads(
        (RESEARCH / "model_v17_liquidity_reliability_protocol.json").read_text(
            encoding="utf-8"
        )
    )
    assert isinstance(value, dict)
    return value


def _daily_prices(
    *,
    periods: int = 34,
    codes: tuple[str, ...] = ("1001", "1002", "1003", "1004"),
) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    sessions = pd.bdate_range("2025-08-01", periods=periods)
    rows: list[dict[str, object]] = []
    for code_index, code in enumerate(codes):
        prior_close = 100.0 + 10.0 * code_index
        for date_index, date in enumerate(sessions):
            open_price = prior_close * (
                1.0 + 0.001 * np.sin((date_index + code_index) / 3.0)
            )
            close = open_price * (
                1.0 + 0.002 * np.cos((date_index + code_index) / 4.0)
            )
            volume = 100_000.0 + 2_000.0 * date_index + 11_000.0 * code_index
            vwap = (2.0 * open_price + close) / 3.0
            rows.append(
                {
                    "date": date,
                    "code": code,
                    "name": code,
                    "open": open_price,
                    "high": max(open_price, close) * 1.003,
                    "low": min(open_price, close) * 0.997,
                    "close": close,
                    "volume": volume,
                    "turnover": volume * vwap,
                    "vwap": vwap,
                    "trading_unit": 100.0,
                    "traded": True,
                    "partial_session": False,
                    "source_format": DAILY_FORMAT,
                }
            )
            prior_close = close
    return pd.DataFrame(rows), pd.DatetimeIndex(sessions)


def _target_row(
    features: pd.DataFrame, target: pd.Timestamp, code: str = "1001"
) -> pd.Series:
    rows = features.loc[
        features["date"].eq(target) & features["code"].eq(code)
    ]
    assert len(rows) == 1
    return rows.iloc[0]


def _score_ledger() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "candidate_id": ["B", "A", "A"],
            "date": pd.to_datetime(["2026-04-02", "2026-04-02", "2026-04-01"]),
            "model_rank": [1, 1, 1],
            "code": ["1003", "1002", "1001"],
            "name": ["c", "b", "a"],
            "model_score": [0.1, 0.2, 0.3],
            "pre_veto_code": ["1003", "1002", "1001"],
            "vetoed": [False, False, False],
            "source_rank": [1, 1, 1],
        }
    )


def test_exact_protocol_hash_candidate_registry_and_authority() -> None:
    protocol_path = RESEARCH / "model_v17_liquidity_reliability_protocol.json"
    observed_sha = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    protocol = _protocol()

    assert observed_sha == EXPECTED_PROTOCOL_SHA256
    assert observed_sha == runner.PROTOCOL_SHA256 == audit.PROTOCOL_SHA256
    assert protocol["protocol_id"] == runner.PROTOCOL_ID == audit.PROTOCOL_ID
    assert tuple(item["id"] for item in protocol["candidates"]) == (
        EXPECTED_CANDIDATES
    )
    assert runner.CANDIDATES == EXPECTED_CANDIDATES
    assert runner.MATCHED_CONTROL == audit.MATCHED_CONTROL == {
        candidate: (
            "PAIR00_PRICE_ONLY_TOP2_RERANK"
            if candidate.startswith("PAIR01_")
            else "C00_PRICE_RIDGE_TOP1"
        )
        for candidate in EXPECTED_CANDIDATES
    }
    for authority in (protocol["authority"],):
        assert authority["project_level_untouched"] is False
        assert authority["production_promotion_allowed"] is False
        assert authority["production_model_changed"] is False
        assert authority["orders_allowed"] is False
    validated, validated_sha = runner.validate_protocol(protocol_path)
    assert validated == protocol
    assert validated_sha == EXPECTED_PROTOCOL_SHA256


def test_exact_ten_hypothesis_formulas_and_models_are_registered() -> None:
    candidates = {item["id"]: item for item in _protocol()["candidates"]}

    assert candidates["PAIR01_TOP2_EXACT_LIQ_RERANK"]["features"] == (
        "C00 model-score difference plus differences of all seven "
        "exact_liquidity_rank_features"
    )
    assert candidates["PAIR01_TOP2_EXACT_LIQ_RERANK"]["model"] == (
        "LogisticRegression"
    )
    assert "equal-return dates are excluded identically" in candidates[
        "PAIR01_TOP2_EXACT_LIQ_RERANK"
    ]["training_pairs"]
    assert candidates["TW02_TURNOVER_WEIGHTED_PRICE_MEMORY"]["features"] == [
        "tw_mean_k = sum(turnover[t] * OC_return_pct[t]) / sum(turnover[t]) for k in {5,20}",
        "tw_minus_equal_k = tw_mean_k - mean(OC_return_pct[t]) for k in {5,20}",
    ]
    assert candidates["VW03_AGGREGATE_COST_BASIS"]["features"] == [
        "AVWAP_k = sum(turnover[t]) / sum(volume[t]) for k in {5,20}",
        "xrank_basis_gap20 = xrank(close[D-1] / AVWAP_20 - 1)",
        "xrank_basis_migration5_20 = xrank(AVWAP_5 / AVWAP_20 - 1)",
    ]
    assert candidates["RPY04_RETURN_PER_TURNOVER_REVERSAL"]["formula"] == (
        "-xrank(sum(OC_return_pct[t] / turnover[t]) over D-5 through D-1)"
    )
    assert candidates["RPY04_RETURN_PER_TURNOVER_REVERSAL"]["model"] == (
        "deterministic; no fit and no direction search"
    )
    assert candidates["AR05_SECURITY_ACTIVITY_REGIME_EXPERTS"]["state"] == (
        "xrank_liq_turnover_shock1 > 0 versus <= 0"
    )
    assert candidates["PS06_PERSISTENT_VS_ISOLATED_ACTIVITY"]["features"] == [
        "sustained = clip(log(median(turnover[D-3:D-1]) / median(turnover[D-20:D-4])), -3, 3)",
        "isolated = clip(log(turnover[D-1] / median(turnover[D-20:D-2])) - sustained, -3, 3)",
        "cross-sectional xrank of sustained and isolated",
        "each xrank multiplied by xrank_liq_close_vwap_dev1",
    ]
    assert candidates["VP07_VWAP_RANGE_PRESSURE_MEMORY"]["features"] == [
        "p[t] = clip((close[t] - vwap[t]) / (high[t] - low[t]), -1, 1)",
        "turnover-weighted p over trailing 5 and 20 sessions, then xrank each",
    ]
    assert candidates["RW08_EXACT_LOT_RELIABILITY_WEIGHT"]["reliability"] == (
        "median(volume / trading_unit) over D-20:D-1, ranked to a strictly "
        "positive within-date percentile"
    )
    assert candidates["RW08_EXACT_LOT_RELIABILITY_WEIGHT"]["score_universe"] == (
        "identical to C00_PRICE_RIDGE_TOP1"
    )
    assert candidates["MR09_TURNOVER_WEIGHTED_MARKET_REGIME"]["aggregates"] == [
        "pressure_breadth[D-1] = sum(turnover_i * sign(close_i - vwap_i)) / sum(turnover_i)",
        "turnover_HHI[D-1] = sum((turnover_i / sum(turnover_i))^2)",
    ]
    assert candidates["MR09_TURNOVER_WEIGHTED_MARKET_REGIME"]["features"] == (
        "pressure_breadth and turnover_HHI each multiplied by signed "
        "xrank_close_momentum_5 and signed xrank_close_momentum_20"
    )
    assert candidates["AT10_ATTENTION_MIGRATION"]["features"] == [
        "migration = turnover_xrank[D-1] - median(turnover_xrank[D-5:D-2])",
        "xrank_attention_migration = cross-sectional xrank(migration)",
        "xrank_attention_migration * xrank_liq_close_vwap_dev1",
    ]
    assert "All four issuer turnover ranks" in candidates[
        "AT10_ATTENTION_MIGRATION"
    ]["availability_rule"]
    assert "never imputed or proxied" in candidates[
        "AT10_ATTENTION_MIGRATION"
    ]["availability_rule"]


def test_protocol_runner_evaluation_contract_is_exact() -> None:
    protocol = _protocol()
    evaluation = protocol["evaluation"]
    assert evaluation["fixed_slots_per_model_date"] == 1
    assert evaluation["candidate_variants"] == 10
    assert evaluation["no_capacity_grid"] is True
    assert evaluation["costs_bps"] == [20, 40, 60]
    assert evaluation["primary_cost_bps"] == 40
    assert evaluation["selection_slices"] == {
        "replay_a": ["2026-04-01", "2026-05-29"],
        "replay_b": ["2026-06-01", "2026-07-27"],
    }
    assert evaluation["bootstrap"] == {
        "method": "paired moving block",
        "block_length": 5,
        "samples": 20000,
        "random_state": 20260804,
        "candidate_seed_rule": (
            "random_state + zero-based candidate order index, yielding "
            "20260804 through 20260813"
        ),
        "familywise_one_sided_confidence": 0.9,
        "bonferroni_individual_confidence": 0.99,
    }
    assert evaluation["selection_gate_all_required"] == {
        "net40_mean_positive": True,
        "net40_median_positive": True,
        "net60_mean_positive": True,
        "both_fixed_slices_net40_positive": True,
        "positive_months_net40_at_least": 3,
        "top4_days_removed_net40_positive": True,
        "top5_profit_codes_cash_net40_positive": True,
        "familywise_paired_lower_vs_control_nonnegative": True,
        "unique_codes_at_least": 40,
        "maximum_code_share_at_most": 0.05,
        "top10_code_share_at_most": 0.25,
        "executed_days_at_least": 72,
        "executed_slot_fraction_at_least": 0.8,
    }
    assert runner.SELECTION_SLICES == audit.SELECTION_SLICES
    assert runner.COSTS_BPS == audit.COSTS_BPS
    assert runner.BOOTSTRAP_SAMPLES == audit.BOOTSTRAP_SAMPLES
    assert runner.BONFERRONI_CONFIDENCE == audit.BOOTSTRAP_CONFIDENCE


def test_source_lock_tamper_fails_closed(tmp_path: Path) -> None:
    source = RESEARCH / "model_v17_replay_input_lock.json"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == runner.INPUT_LOCK_SHA256
    lock = json.loads(source.read_text(encoding="utf-8"))
    assert audit._locked_source_set_digest(lock["files"]) == (
        runner.INPUT_LOCK_SOURCE_SET_SHA256
    )

    tampered = copy.deepcopy(lock)
    tampered["files"][0]["bytes"] += 1
    path = tmp_path / "tampered-lock.json"
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="input-lock SHA-256 mismatch"):
        runner._expected_extension_sources(path)


def test_features_are_d_minus_one_and_same_day_invariant() -> None:
    prices, sessions = _daily_prices()
    target = sessions[26]
    original = runner.build_reliability_features(prices, sessions)
    original_target = original[original["date"].eq(target)].reset_index(drop=True)
    assert original_target["_history_complete20"].all()
    assert original_target["feature_source_max_date_v17"].eq(sessions[25]).all()

    mutated = prices.copy()
    target_rows = mutated["date"].eq(target)
    mutated.loc[
        target_rows,
        ["open", "high", "low", "close", "volume", "turnover", "vwap", "trading_unit"],
    ] *= np.asarray([1.4, 1.8, 0.6, 1.5, 8.0, 9.0, 1.7, 2.0])
    changed = runner.build_reliability_features(mutated, sessions)
    pd.testing.assert_frame_equal(
        original_target,
        changed[changed["date"].eq(target)].reset_index(drop=True),
        check_exact=True,
    )


def test_future_mutation_cannot_change_earlier_features() -> None:
    prices, sessions = _daily_prices()
    cutoff = sessions[26]
    original = runner.build_reliability_features(prices, sessions)
    mutated = prices.copy()
    mask = mutated["date"].eq(sessions[30])
    mutated.loc[mask, ["close", "volume", "turnover", "vwap"]] *= [2.0, 7.0, 9.0, 1.8]
    changed = runner.build_reliability_features(mutated, sessions)
    pd.testing.assert_frame_equal(
        original.loc[original["date"].le(cutoff)].reset_index(drop=True),
        changed.loc[changed["date"].le(cutoff)].reset_index(drop=True),
        check_exact=True,
    )


def test_row_order_is_stable_and_duplicate_fails_closed() -> None:
    prices, sessions = _daily_prices()
    original = runner.build_reliability_features(prices, sessions)
    shuffled = runner.build_reliability_features(
        prices.sample(frac=1.0, random_state=71), sessions
    )
    pd.testing.assert_frame_equal(original, shuffled, check_exact=True)

    duplicated = pd.concat([prices, prices.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        runner.build_reliability_features(duplicated, sessions)


def test_nonconsecutive_nonpositive_and_unit_change_fail_history() -> None:
    prices, sessions = _daily_prices()
    target = sessions[26]

    missing = prices.loc[
        ~(prices["date"].eq(sessions[12]) & prices["code"].eq("1001"))
    ].copy()
    assert not bool(
        _target_row(runner.build_reliability_features(missing, sessions), target)[
            "_history_complete20"
        ]
    )

    nonpositive = prices.copy()
    mask = nonpositive["date"].eq(sessions[12]) & nonpositive["code"].eq("1001")
    nonpositive.loc[mask, ["volume", "turnover"]] = 0.0
    assert not bool(
        _target_row(runner.build_reliability_features(nonpositive, sessions), target)[
            "_history_complete20"
        ]
    )

    changed_unit = prices.copy()
    changed_unit.loc[mask, "trading_unit"] = 200.0
    assert not bool(
        _target_row(runner.build_reliability_features(changed_unit, sessions), target)[
            "_history_complete20"
        ]
    )


def test_vp07_zero_range_rule_and_at10_observed_rank_rule() -> None:
    prices, sessions = _daily_prices()
    source = sessions[25]
    target = sessions[26]
    mask = prices["date"].eq(source) & prices["code"].eq("1001")

    valid = prices.copy()
    close = float(valid.loc[mask, "close"].iloc[0])
    valid.loc[mask, ["high", "low", "vwap"]] = close
    valid_row = _target_row(
        runner.build_reliability_features(valid, sessions), target
    )
    assert np.isfinite(valid_row["vp07_pressure_5"])
    assert np.isfinite(valid_row["vp07_pressure_20"])

    invalid = valid.copy()
    invalid.loc[mask, "vwap"] = close * 0.99
    invalid_row = _target_row(
        runner.build_reliability_features(invalid, sessions), target
    )
    assert np.isnan(invalid_row["vp07_pressure_5"])
    assert np.isnan(invalid_row["vp07_pressure_20"])

    complete = _target_row(
        runner.build_reliability_features(prices, sessions), target
    )
    assert np.isfinite(complete["at10_attention_migration"])
    missing_prior_rank = prices.loc[
        ~(prices["date"].eq(sessions[23]) & prices["code"].eq("1001"))
    ].copy()
    incomplete = _target_row(
        runner.build_reliability_features(missing_prior_rank, sessions), target
    )
    assert np.isnan(incomplete["at10_attention_migration"])
    assert np.isnan(incomplete["at10_xrank_attention_migration"])


def test_pairwise_selection_is_top2_only_and_incomplete_pair_is_cash() -> None:
    class StubModel:
        def decision_function(self, frame: pd.DataFrame) -> np.ndarray:
            return frame.iloc[:, 0].to_numpy(dtype=float)

    dates = pd.to_datetime(["2026-04-01", "2026-04-01", "2026-04-02"])
    top2 = pd.DataFrame(
        {
            "date": dates,
            "code": ["1001", "1002", "1003"],
            "name": ["a", "b", "c"],
            "source_rank": [1, 2, 1],
            "feature": [0.2, 0.8, 0.9],
        }
    )
    scheduled = pd.DatetimeIndex(
        [
            pd.Timestamp("2026-04-01"),
            pd.Timestamp("2026-04-02"),
            pd.Timestamp("2026-04-03"),
        ]
    )
    selected = runner.choose_pairwise_top_one(
        top2, StubModel(), ("feature",), scheduled, "PAIR_TEST"
    )
    first = selected.loc[selected["date"].eq(scheduled[0])].iloc[0]
    incomplete = selected.loc[selected["date"].eq(scheduled[1])].iloc[0]
    empty = selected.loc[selected["date"].eq(scheduled[2])].iloc[0]
    assert first["code"] in {"1001", "1002"}
    assert int(first["source_rank"]) in (1, 2)
    assert "9999" not in set(selected["code"].dropna())
    assert pd.isna(incomplete["code"])
    assert pd.isna(incomplete["source_rank"])
    assert pd.isna(empty["code"])
    assert pd.isna(empty["source_rank"])


def test_equal_return_pair_is_excluded_and_unequal_pair_is_symmetric() -> None:
    rows = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2026-03-02", "2026-03-02", "2026-03-03", "2026-03-03"]
            ),
            "code": ["1001", "1002", "1001", "1002"],
            "oc_return_pct": [0.5, 0.5, 0.7, -0.2],
            "feature": [0.3, -0.1, 0.8, 0.2],
        }
    )
    training = runner.build_pair_training_rows(
        rows, before=pd.Timestamp("2026-04-01"), feature_columns=("feature",)
    )
    assert training["date"].nunique() == 1
    assert training["date"].iloc[0] == pd.Timestamp("2026-03-03")
    assert training["target"].tolist() == [1, 0]
    np.testing.assert_allclose(training["feature"], [0.6, -0.6])


def test_rw08_weights_preserve_common_scoring_universe() -> None:
    dates = pd.to_datetime(["2026-04-01"] * 4)
    panel = pd.DataFrame(
        {
            "date": dates,
            "code": ["1001", "1002", "1003", "1004"],
            "common_score_eligible": [True, True, True, False],
            "rw08_traded_lots_med20": [10.0, 30.0, 20.0, np.nan],
        }
    )
    scheduled = pd.DatetimeIndex([pd.Timestamp("2026-04-01")])
    c00_universe = runner._candidate_scoring(panel, scheduled)
    rw_universe = runner._candidate_scoring(panel, scheduled)
    assert c00_universe[["date", "code"]].equals(rw_universe[["date", "code"]])
    assert runner._finite_feature_mask(
        rw_universe, ("rw08_traded_lots_med20",)
    ).all()

    training = c00_universe.assign(oc_return_pct=[0.1, -0.1, 0.0])
    weights = runner.reliability_weights(training)
    assert (weights > 0.0).all()
    assert weights.groupby(training["date"]).sum().iloc[0] == pytest.approx(1.0)


def test_score_hash_is_row_order_invariant_and_outcome_join_is_fail_closed() -> None:
    ledger = _score_ledger()
    assert not {"label", "oc_return_pct"} & set(ledger.columns)
    assert runner.semantic_score_hash(ledger) == runner.semantic_score_hash(
        ledger.sample(frac=1.0, random_state=7)
    )
    assert audit.semantic_score_hash(ledger) == runner.semantic_score_hash(ledger)

    outcomes = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-04-01", "2026-04-02", "2026-04-02"]),
            "code": ["1001", "1002", "1003"],
            "label": [1.0, 0.0, np.nan],
            "oc_return_pct": [0.5, -0.4, np.nan],
        }
    )
    joined = runner.attach_outcomes(ledger, outcomes)
    assert joined["label"].notna().sum() == 2
    with pytest.raises(ValueError, match="duplicate"):
        runner.attach_outcomes(ledger, pd.concat([outcomes, outcomes.iloc[[0]]]))
    with pytest.raises(ValueError, match="already contains outcomes"):
        runner.attach_outcomes(joined, outcomes)


def test_independent_audit_has_no_runner_or_project_metric_imports() -> None:
    source_path = RESEARCH / "model_v17_liquidity_reliability_audit.py"
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported.update(
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    )
    assert not any("model_v17_liquidity_reliability_runner" in name for name in imported)
    assert "tse_session_ranker.profit" not in imported
    assert "tse_session_ranker.validation" not in imported


def test_independent_cost_tail_bootstrap_and_gates_match_runner() -> None:
    lock = json.loads(
        (RESEARCH / "model_v17_replay_input_lock.json").read_text(
            encoding="utf-8"
        )
    )
    scheduled = pd.DatetimeIndex(
        pd.to_datetime([item["date"] for item in lock["files"]])
    )

    def picks(model_id: str, phase: float) -> pd.DataFrame:
        returns = 0.8 + 0.3 * np.sin(np.arange(len(scheduled)) / 4.0 + phase)
        return pd.DataFrame(
            {
                "candidate_id": model_id,
                "date": scheduled,
                "model_rank": 1,
                "code": [str(1000 + index) for index in range(len(scheduled))],
                "label": (returns > 0.0).astype(float),
                "oc_return_pct": returns,
            }
        )

    candidate = picks(EXPECTED_CANDIDATES[0], 0.0)
    control = picks(runner.PAIR00, 1.0)
    expected = runner._variant_metrics(
        candidate,
        control,
        scheduled,
        candidate_id=EXPECTED_CANDIDATES[0],
        control_id=runner.PAIR00,
        random_state=runner.BOOTSTRAP_RANDOM_STATE,
    )
    observed = audit.variant_metrics(
        candidate,
        control,
        scheduled,
        candidate_id=EXPECTED_CANDIDATES[0],
        control_id=runner.PAIR00,
        random_state=runner.BOOTSTRAP_RANDOM_STATE,
    )
    assert audit.json_safe(observed) == audit.json_safe(expected)


def test_saved_artifacts_pass_independent_audit_when_present() -> None:
    paths = {
        "result": RESEARCH / "model_v17_liquidity_reliability_result.json",
        "scores": RESEARCH / "model_v17_liquidity_reliability_scores.csv",
        "picks": RESEARCH / "model_v17_liquidity_reliability_picks.csv",
    }
    if not all(path.exists() for path in paths.values()):
        pytest.skip("v1.7 one-shot selection artifacts have not been generated yet")
    report = audit.audit(
        protocol_path=RESEARCH / "model_v17_liquidity_reliability_protocol.json",
        result_path=paths["result"],
        scores_path=paths["scores"],
        picks_path=paths["picks"],
        runner_path=RESEARCH / "model_v17_liquidity_reliability_runner.py",
    )
    assert report["status"] == "pass"
    assert report["discrepancies"] == []
    assert all(report["checks"].values())
    assert report["authority"]["production_model_changed"] is False
    assert report["authority"]["production_promotion_allowed"] is False
    assert report["authority"]["orders_allowed"] is False
