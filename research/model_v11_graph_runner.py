#!/usr/bin/env python3
"""Preregistered cross-stock graph and information-propagation experiment."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans


ROOT = Path(__file__).resolve().parents[1]
for import_root in (ROOT / "src", ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from tse_session_ranker.research_models import (  # noqa: E402
    ResearchModelSpec,
    fit_research_model,
)


PROTOCOL_PATH = ROOT / "research" / "model_v11_graph_protocol.json"
PROTOCOL_SHA_PATH = ROOT / "research" / "model_v11_graph_protocol.sha256"
EXPECTED_PROTOCOL_SHA256 = (
    "2f2382716ba9f7b151988470334b4da93bee5f68e30f7264beedcf16bf790244"
)
SCORE_START = pd.Timestamp("2024-07-01")
SCORE_END = pd.Timestamp("2025-07-31")
GRAPH_WINDOW = 120
MIN_OBSERVED = 60
NODE_CAPACITY = 1536
NEIGHBORS = 8
COMMUNITIES = 4
MIN_SIGN_OBSERVATIONS = 20
MIN_TDNET_EVENTS = 3
COSTS = (0, 20, 40, 60)
RANDOM_STATE = 20260723

G0_FEATURES = (
    "oc_last",
    "oc_mean_5",
    "oc_mean_20",
    "oc_mean_60",
    "oc_win_20",
    "oc_std_20",
    "overnight_last",
    "overnight_mean_20",
    "overnight_mean_60",
    "night_day_corr_60",
    "xrank_atr14_pct",
    "xrank_close_momentum_5",
    "xrank_close_momentum_20",
    "xrank_close_momentum_60",
    "xrank_prior_close_location_20",
)
L4_FEATURES = (*G0_FEATURES, "flat_oc_rate_20")
COMPARATOR_IDS = ("C00_daily_rank_ridge", "C01_L4_price_control")
GRAPH_IDS = (
    "G01_positive_oc_corr_message",
    "G02_signed_oc_corr_message",
    "G03_directed_lead_lag_message",
    "G04_overnight_corr_message",
    "G05_cojump_neighbor_pressure",
    "G06_tdnet_event_neighbor_spillover",
    "G07_neighbor_conditioned_empirical_uplift",
    "G08_dynamic_community_archetype",
    "G09_two_hop_oc_diffusion",
    "G10_graph_disagreement_cash",
)
ALL_IDS = (*COMPARATOR_IDS, *GRAPH_IDS)
PERIODS = {
    "discovery": (pd.Timestamp("2024-07-01"), pd.Timestamp("2024-10-31")),
    "confirmation_a": (pd.Timestamp("2024-11-01"), pd.Timestamp("2025-03-31")),
    "confirmation_b": (pd.Timestamp("2025-04-01"), pd.Timestamp("2025-07-31")),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_arrays(values: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(values):
        array = np.ascontiguousarray(values[name])
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(array.shape).encode("ascii"))
        digest.update(b"\0")
        digest.update(array.tobytes())
    return digest.hexdigest()


def load_protocol() -> tuple[dict[str, Any], dict[str, Any]]:
    actual = sha256_file(PROTOCOL_PATH)
    if actual != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("graph protocol changed after registration")
    if (
        PROTOCOL_SHA_PATH.read_text(encoding="utf-8").strip()
        != f"{EXPECTED_PROTOCOL_SHA256}  model_v11_graph_protocol.json"
    ):
        raise RuntimeError("graph protocol SHA sidecar mismatch")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol["protocol_id"] != "v11_cross_stock_graph_propagation_20260723":
        raise RuntimeError("unexpected graph protocol id")
    if tuple(item["id"] for item in protocol["registered_hypotheses"]) != GRAPH_IDS:
        raise RuntimeError("registered graph candidate identities changed")
    if protocol["hypothesis_family_size"] != len(GRAPH_IDS):
        raise RuntimeError("graph family size mismatch")
    if protocol["authority"]["production_promotion_allowed_from_this_run"]:
        raise RuntimeError("retrospective protocol cannot permit promotion")

    panel_spec = protocol["frozen_inputs"]["panel"]
    panel_path = Path(panel_spec["path"])
    manifest_path = Path(panel_spec["manifest_path"])
    if sha256_file(panel_path) != panel_spec["sha256"]:
        raise RuntimeError("panel SHA mismatch")
    if sha256_file(manifest_path) != panel_spec["manifest_sha256"]:
        raise RuntimeError("manifest SHA mismatch")
    reference_hashes = {}
    for name, spec in protocol["frozen_inputs"]["known_references"].items():
        path = ROOT / spec["path"]
        value = sha256_file(path)
        if value != spec["sha256"]:
            raise RuntimeError(f"known reference changed: {path}")
        reference_hashes[name] = value
    return protocol, {
        "protocol_sha256": actual,
        "panel_sha256": panel_spec["sha256"],
        "panel_manifest_sha256": panel_spec["manifest_sha256"],
        "known_reference_hashes": reference_hashes,
    }


def matrix_from_panel(
    panel: pd.DataFrame,
    dates: pd.DatetimeIndex,
    codes: Sequence[str],
    column: str,
    *,
    require_training_eligible: bool = False,
) -> np.ndarray:
    local = panel.loc[
        panel["date"].isin(dates) & panel["code"].isin(codes),
        ["date", "code", column, "price_training_eligible"],
    ].copy()
    if require_training_eligible:
        local.loc[~local["price_training_eligible"], column] = np.nan
    matrix = local.pivot(index="date", columns="code", values=column).reindex(
        index=dates, columns=list(codes)
    )
    return matrix.to_numpy(dtype=float)


def frame_matrix(
    frame: pd.DataFrame,
    dates: pd.DatetimeIndex,
    codes: Sequence[str],
    column: str,
) -> np.ndarray:
    matrix = frame.pivot(index="date", columns="code", values=column).reindex(
        index=dates, columns=list(codes)
    )
    return matrix.to_numpy(dtype=float)


def normalized_columns(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    observed = np.isfinite(values)
    count = observed.sum(axis=0)
    total = np.where(observed, values, 0.0).sum(axis=0)
    mean = np.divide(
        total,
        count,
        out=np.zeros(values.shape[1], dtype=np.float32),
        where=count > 0,
    )
    centered = np.where(observed, values - mean, 0.0)
    norm = np.sqrt(np.square(centered, dtype=np.float32).sum(axis=0))
    return np.divide(
        centered,
        norm,
        out=np.zeros_like(centered, dtype=np.float32),
        where=norm > 0,
    )


def similarity_matrix(
    target_values: np.ndarray, source_values: np.ndarray | None = None
) -> np.ndarray:
    target = normalized_columns(target_values)
    source = target if source_values is None else normalized_columns(source_values)
    result = target.T @ source
    return np.asarray(result, dtype=np.float32)


def adjacency_from_similarity(
    similarity: np.ndarray,
    codes: np.ndarray,
    *,
    signed: bool,
) -> sparse.csr_matrix:
    similarity = np.asarray(similarity, dtype=np.float32).copy()
    np.fill_diagonal(similarity, 0.0)
    rows: list[int] = []
    columns: list[int] = []
    data: list[float] = []
    for target in range(len(codes)):
        values = similarity[target]
        strength = np.abs(values) if signed else values
        valid = np.flatnonzero(
            np.isfinite(strength)
            & (strength > 0.0)
            & (np.arange(len(codes)) != target)
        )
        if not len(valid):
            continue
        take = min(NEIGHBORS, len(valid))
        if len(valid) > take:
            boundary = np.partition(strength[valid], len(valid) - take)[
                len(valid) - take
            ]
            pool = valid[strength[valid] >= boundary]
        else:
            pool = valid
        order = np.lexsort((codes[pool], -strength[pool]))
        selected = pool[order[:take]]
        selected_values = values[selected] if signed else strength[selected]
        denominator = float(np.abs(selected_values).sum())
        if denominator <= 0.0:
            continue
        rows.extend([target] * len(selected))
        columns.extend(selected.tolist())
        data.extend((selected_values / denominator).astype(float).tolist())
    return sparse.csr_matrix(
        (data, (rows, columns)), shape=similarity.shape, dtype=np.float64
    )


def row_normalize(matrix: sparse.csr_matrix) -> sparse.csr_matrix:
    result = matrix.tocsr(copy=True).astype(float)
    denominator = np.asarray(np.abs(result).sum(axis=1)).ravel()
    scale = np.divide(
        1.0,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0.0,
    )
    return sparse.diags(scale) @ result


def two_hop_adjacency(adjacency: sparse.csr_matrix) -> sparse.csr_matrix:
    result = (adjacency @ adjacency).tocsr()
    result.setdiag(0.0)
    result.eliminate_zeros()
    return row_normalize(result)


def spectral_communities(adjacency: sparse.csr_matrix) -> np.ndarray:
    symmetric = ((adjacency + adjacency.T) * 0.5).tocsr()
    degree = np.asarray(symmetric.sum(axis=1)).ravel()
    if np.count_nonzero(degree > 0.0) < COMMUNITIES:
        raise RuntimeError("insufficient connected nodes for communities")
    inverse = np.divide(
        1.0,
        np.sqrt(degree),
        out=np.zeros_like(degree),
        where=degree > 0.0,
    )
    normalized = sparse.diags(inverse) @ symmetric @ sparse.diags(inverse)
    _, vectors = eigsh(
        normalized,
        k=COMMUNITIES,
        which="LA",
        v0=np.ones(normalized.shape[0], dtype=float),
        tol=1e-8,
        maxiter=10_000,
    )
    row_norm = np.linalg.norm(vectors, axis=1)
    embedding = np.divide(
        vectors,
        row_norm[:, None],
        out=np.zeros_like(vectors),
        where=row_norm[:, None] > 0.0,
    )
    labels = KMeans(
        n_clusters=COMMUNITIES,
        random_state=RANDOM_STATE,
        n_init=10,
        algorithm="lloyd",
    ).fit_predict(embedding)
    if sorted(np.unique(labels).tolist()) != list(range(COMMUNITIES)):
        raise RuntimeError("spectral community fit did not occupy four groups")
    return labels.astype(np.int8)


def rowwise_percentile_message(values: np.ndarray) -> np.ndarray:
    frame = pd.DataFrame(values)
    rank = frame.rank(axis=1, method="average")
    count = frame.notna().sum(axis=1)
    output = rank.sub(0.5).div(count.replace(0, np.nan), axis=0).mul(2.0).sub(1.0)
    return output.to_numpy(dtype=float)


def select_graph_nodes(
    panel: pd.DataFrame,
    sessions: pd.DatetimeIndex,
    fold_start: pd.Timestamp,
) -> tuple[np.ndarray, pd.DatetimeIndex, dict[str, Any]]:
    prior_sessions = sessions[sessions < fold_start]
    graph_dates = prior_sessions[-GRAPH_WINDOW:]
    history = panel.loc[
        panel["date"].isin(graph_dates)
        & panel["price_training_eligible"]
        & panel["oc_return_pct"].notna(),
        ["date", "code", "oc_return_pct"],
    ]
    counts = history.groupby("code", sort=True)["oc_return_pct"].count()
    qualified = counts.loc[counts.ge(MIN_OBSERVED)].index.astype(str)
    state = panel.loc[
        panel["date"].lt(fold_start)
        & panel["code"].isin(qualified)
        & panel["no_trade_rate_20"].notna(),
        ["date", "code", "no_trade_rate_20"],
    ].sort_values(["date", "code"], kind="stable")
    latest = state.groupby("code", sort=True).tail(1).set_index("code")
    catalog = pd.DataFrame(index=pd.Index(qualified, dtype=object))
    catalog["history_count"] = counts.reindex(catalog.index).astype(int)
    catalog["latest_no_trade_rate_20"] = latest[
        "no_trade_rate_20"
    ].reindex(catalog.index)
    catalog["code_tie_break"] = catalog.index.astype(str)
    catalog = catalog.sort_values(
        ["latest_no_trade_rate_20", "history_count", "code_tie_break"],
        ascending=[True, False, True],
        na_position="last",
        kind="stable",
    )
    nodes = catalog.head(NODE_CAPACITY).index.astype(str).to_numpy()
    if len(nodes) < 100:
        raise RuntimeError(f"only {len(nodes)} graph nodes in {fold_start:%Y-%m}")
    return nodes, graph_dates, {
        "qualified_codes": int(len(catalog)),
        "selected_nodes": int(len(nodes)),
        "graph_window_first": str(graph_dates.min().date()),
        "graph_window_last": str(graph_dates.max().date()),
        "graph_window_sessions": int(len(graph_dates)),
        "minimum_selected_history_count": int(
            catalog.loc[nodes, "history_count"].min()
        ),
        "maximum_selected_no_trade_rate_20": float(
            catalog.loc[nodes, "latest_no_trade_rate_20"].max()
        ),
    }


def conditional_response(
    message: np.ndarray,
    target_returns: np.ndarray,
    current_message: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    nodes = message.shape[1]
    positive_mean = np.full(nodes, np.nan, dtype=float)
    negative_mean = np.full(nodes, np.nan, dtype=float)
    positive_count = np.zeros(nodes, dtype=int)
    negative_count = np.zeros(nodes, dtype=int)
    for node in range(nodes):
        observed = np.isfinite(target_returns[:, node])
        positive = observed & np.isfinite(message[:, node]) & (message[:, node] > 0)
        negative = observed & np.isfinite(message[:, node]) & (message[:, node] < 0)
        positive_count[node] = int(positive.sum())
        negative_count[node] = int(negative.sum())
        if positive_count[node] >= MIN_SIGN_OBSERVATIONS:
            positive_mean[node] = float(target_returns[positive, node].mean())
        if negative_count[node] >= MIN_SIGN_OBSERVATIONS:
            negative_mean[node] = float(target_returns[negative, node].mean())
    score = np.where(current_message > 0, positive_mean, negative_mean)
    eligible = (
        np.isfinite(current_message)
        & (current_message != 0.0)
        & np.isfinite(score)
    )
    return score, eligible, {
        "positive_sign_qualified_nodes": int(np.isfinite(positive_mean).sum()),
        "negative_sign_qualified_nodes": int(np.isfinite(negative_mean).sum()),
        "minimum_positive_count": int(positive_count.min()),
        "minimum_negative_count": int(negative_count.min()),
    }


def build_tdnet_reaction(
    adjacency: sparse.csr_matrix,
    events: np.ndarray,
    returns: np.ndarray,
    complete_dates: np.ndarray,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, dict[str, Any]]:
    adjacency = adjacency.tocsr()
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    presence: list[float] = []
    counts: list[int] = []
    for target in range(adjacency.shape[0]):
        sources = adjacency.indices[
            adjacency.indptr[target] : adjacency.indptr[target + 1]
        ]
        for source in sources:
            mask = (
                complete_dates
                & np.isfinite(returns[:, target])
                & np.isfinite(events[:, source])
                & (events[:, source] > 0.0)
            )
            count = int(mask.sum())
            if count < MIN_TDNET_EVENTS:
                continue
            rows.append(target)
            columns.append(int(source))
            values.append(float(returns[mask, target].mean()))
            presence.append(1.0)
            counts.append(count)
    shape = adjacency.shape
    reaction = sparse.csr_matrix((values, (rows, columns)), shape=shape)
    availability = sparse.csr_matrix((presence, (rows, columns)), shape=shape)
    return reaction, availability, {
        "qualified_directed_edges": int(len(values)),
        "minimum_event_dates_on_qualified_edge": min(counts) if counts else None,
        "maximum_event_dates_on_qualified_edge": max(counts) if counts else None,
    }


def build_fold_graph(
    panel: pd.DataFrame,
    sessions: pd.DatetimeIndex,
    fold_start: pd.Timestamp,
    nodes: np.ndarray,
    graph_dates: pd.DatetimeIndex,
) -> tuple[dict[str, Any], dict[str, Any]]:
    oc_window = matrix_from_panel(
        panel,
        graph_dates,
        nodes,
        "oc_return_pct",
        require_training_eligible=True,
    )
    overnight_window = matrix_from_panel(
        panel,
        graph_dates,
        nodes,
        "overnight_last",
        require_training_eligible=True,
    )
    positive_oc = adjacency_from_similarity(
        similarity_matrix(oc_window), nodes, signed=False
    )
    signed_oc = adjacency_from_similarity(
        similarity_matrix(oc_window), nodes, signed=True
    )
    lead_lag = adjacency_from_similarity(
        similarity_matrix(oc_window[1:], oc_window[:-1]),
        nodes,
        signed=True,
    )
    overnight = adjacency_from_similarity(
        similarity_matrix(overnight_window), nodes, signed=False
    )
    absolute = np.abs(oc_window)
    threshold = np.nanquantile(absolute, 0.90, axis=1)
    jumps = np.where(
        np.isfinite(oc_window) & (absolute >= threshold[:, None]),
        np.sign(oc_window),
        0.0,
    )
    cojump = adjacency_from_similarity(
        similarity_matrix(jumps), nodes, signed=False
    )
    two_hop = two_hop_adjacency(positive_oc)
    community = spectral_communities(positive_oc)

    history_dates = sessions[sessions < fold_start]
    history_returns = matrix_from_panel(
        panel,
        history_dates,
        nodes,
        "oc_return_pct",
        require_training_eligible=True,
    )
    history_oc_last = matrix_from_panel(
        panel,
        history_dates,
        nodes,
        "oc_last",
        require_training_eligible=True,
    )
    history_rank_message = rowwise_percentile_message(history_oc_last)
    lead_history_message = (
        lead_lag @ np.nan_to_num(history_rank_message, nan=0.0).T
    ).T

    tdnet_local = panel.loc[
        panel["date"].isin(history_dates) & panel["code"].isin(nodes),
        [
            "date",
            "code",
            "tdnet_clean_any",
            "tdnet_source_complete",
            "tdnet_clean_feature_source_max_timestamp",
        ],
    ].copy()
    event_rows = tdnet_local["tdnet_clean_any"].gt(0.0)
    if tdnet_local.loc[event_rows, "tdnet_clean_feature_source_max_timestamp"].isna().any():
        raise RuntimeError("historical TDnet event lacks source timestamp")
    if event_rows.any():
        cutoff = pd.DatetimeIndex(
            [
                pd.Timestamp.combine(
                    date.date(), pd.Timestamp("08:58:59").time()
                ).tz_localize("Asia/Tokyo")
                for date in tdnet_local.loc[event_rows, "date"]
            ]
        )
        source_time = pd.DatetimeIndex(
            tdnet_local.loc[
                event_rows, "tdnet_clean_feature_source_max_timestamp"
            ]
        )
        if (source_time > cutoff).any():
            raise RuntimeError("historical TDnet event exceeds decision cutoff")
    history_events = frame_matrix(
        tdnet_local, history_dates, nodes, "tdnet_clean_any"
    )
    complete_by_date = (
        tdnet_local.groupby("date", sort=True)["tdnet_source_complete"]
        .all()
        .reindex(history_dates, fill_value=False)
        .to_numpy(dtype=bool)
    )
    reaction, reaction_availability, reaction_report = build_tdnet_reaction(
        lead_lag, history_events, history_returns, complete_by_date
    )

    community_shock = np.full(
        (len(history_dates), COMMUNITIES), np.nan, dtype=float
    )
    for label in range(COMMUNITIES):
        with np.errstate(invalid="ignore"):
            community_shock[:, label] = np.nanmean(
                history_rank_message[:, community == label], axis=1
            )
    model = {
        "positive_oc": positive_oc,
        "signed_oc": signed_oc,
        "lead_lag": lead_lag,
        "overnight": overnight,
        "cojump": cojump,
        "two_hop": two_hop,
        "community": community,
        "history_returns": history_returns,
        "lead_history_message": lead_history_message,
        "community_history_shock": community_shock,
        "tdnet_reaction": reaction,
        "tdnet_reaction_availability": reaction_availability,
    }
    graph_report = {
        "strictly_prior_graph": bool(graph_dates.max() < fold_start),
        "strictly_prior_conditional_history": bool(history_dates.max() < fold_start),
        "history_first": str(history_dates.min().date()),
        "history_last": str(history_dates.max().date()),
        "history_sessions": int(len(history_dates)),
        "edge_counts": {
            "positive_oc": int(positive_oc.nnz),
            "signed_oc": int(signed_oc.nnz),
            "lead_lag": int(lead_lag.nnz),
            "overnight": int(overnight.nnz),
            "cojump": int(cojump.nnz),
            "two_hop": int(two_hop.nnz),
        },
        "community_counts": {
            str(label): int((community == label).sum())
            for label in range(COMMUNITIES)
        },
        "tdnet_reaction": reaction_report,
    }
    return model, graph_report


def map_node_values_to_score_rows(
    score_rows: pd.DataFrame,
    score_dates: pd.DatetimeIndex,
    nodes: np.ndarray,
    values: np.ndarray,
    eligible: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    date_position = {date: index for index, date in enumerate(score_dates)}
    code_position = {code: index for index, code in enumerate(nodes)}
    output = np.zeros(len(score_rows), dtype=float)
    output_eligible = np.zeros(len(score_rows), dtype=bool)
    for position, row in enumerate(score_rows[["date", "code"]].itertuples(index=False)):
        date_index = date_position.get(pd.Timestamp(row.date))
        code_index = code_position.get(str(row.code))
        if date_index is None or code_index is None:
            continue
        value = values[date_index, code_index]
        if eligible[date_index, code_index] and np.isfinite(value):
            output[position] = float(value)
            output_eligible[position] = True
    return output, output_eligible


def score_graph_fold(
    score_rows: pd.DataFrame,
    score_dates: pd.DatetimeIndex,
    nodes: np.ndarray,
    graph: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    node_frame = score_rows.loc[score_rows["code"].isin(nodes)].copy()
    oc_last = frame_matrix(node_frame, score_dates, nodes, "oc_last")
    overnight_last = frame_matrix(
        node_frame, score_dates, nodes, "overnight_last"
    )
    oc_message = rowwise_percentile_message(oc_last)
    overnight_message = rowwise_percentile_message(overnight_last)
    oc_input = np.nan_to_num(oc_message, nan=0.0)
    overnight_input = np.nan_to_num(overnight_message, nan=0.0)

    positive = (graph["positive_oc"] @ oc_input.T).T
    signed = (graph["signed_oc"] @ oc_input.T).T
    lead = (graph["lead_lag"] @ oc_input.T).T
    overnight = (graph["overnight"] @ overnight_input.T).T
    outer_shock = np.where(np.abs(oc_message) >= 0.80, np.sign(oc_message), 0.0)
    cojump = (graph["cojump"] @ np.nan_to_num(outer_shock, nan=0.0).T).T
    two_hop = (graph["two_hop"] @ oc_input.T).T

    node_available = np.isfinite(frame_matrix(
        node_frame.assign(_present=1.0), score_dates, nodes, "_present"
    ))
    base_degree = np.asarray(graph["positive_oc"].getnnz(axis=1)).ravel() > 0
    signed_degree = np.asarray(graph["signed_oc"].getnnz(axis=1)).ravel() > 0
    lead_degree = np.asarray(graph["lead_lag"].getnnz(axis=1)).ravel() > 0
    overnight_degree = np.asarray(graph["overnight"].getnnz(axis=1)).ravel() > 0
    cojump_degree = np.asarray(graph["cojump"].getnnz(axis=1)).ravel() > 0
    two_hop_degree = np.asarray(graph["two_hop"].getnnz(axis=1)).ravel() > 0

    node_scores: dict[str, np.ndarray] = {
        "G01_positive_oc_corr_message": positive,
        "G02_signed_oc_corr_message": signed,
        "G03_directed_lead_lag_message": lead,
        "G04_overnight_corr_message": overnight,
        "G05_cojump_neighbor_pressure": cojump,
        "G09_two_hop_oc_diffusion": two_hop,
    }
    node_eligible: dict[str, np.ndarray] = {
        "G01_positive_oc_corr_message": node_available & base_degree[None, :],
        "G02_signed_oc_corr_message": node_available & signed_degree[None, :],
        "G03_directed_lead_lag_message": node_available & lead_degree[None, :],
        "G04_overnight_corr_message": node_available
        & overnight_degree[None, :],
        "G05_cojump_neighbor_pressure": node_available
        & cojump_degree[None, :]
        & np.isfinite(cojump)
        & (cojump != 0.0),
        "G09_two_hop_oc_diffusion": node_available & two_hop_degree[None, :],
    }

    event_rows = node_frame["tdnet_clean_any"].gt(0.0)
    if node_frame.loc[
        event_rows, "tdnet_clean_feature_source_max_timestamp"
    ].isna().any():
        raise RuntimeError("score TDnet event lacks source timestamp")
    if event_rows.any():
        cutoff = pd.DatetimeIndex(
            [
                pd.Timestamp.combine(
                    date.date(), pd.Timestamp("08:58:59").time()
                ).tz_localize("Asia/Tokyo")
                for date in node_frame.loc[event_rows, "date"]
            ]
        )
        observed = pd.DatetimeIndex(
            node_frame.loc[
                event_rows, "tdnet_clean_feature_source_max_timestamp"
            ]
        )
        if (observed > cutoff).any():
            raise RuntimeError("score TDnet event exceeds decision cutoff")
    current_events = frame_matrix(
        node_frame, score_dates, nodes, "tdnet_clean_any"
    )
    current_event_binary = np.where(
        np.isfinite(current_events) & (current_events > 0.0), 1.0, 0.0
    )
    event_numerator = (
        graph["tdnet_reaction"] @ current_event_binary.T
    ).T
    event_denominator = (
        graph["tdnet_reaction_availability"] @ current_event_binary.T
    ).T
    event_score = np.divide(
        event_numerator,
        event_denominator,
        out=np.zeros_like(event_numerator),
        where=event_denominator > 0.0,
    )
    complete_dates = (
        node_frame.groupby("date", sort=True)["tdnet_source_complete"]
        .all()
        .reindex(score_dates, fill_value=False)
        .to_numpy(dtype=bool)
    )
    event_eligible = (
        node_available
        & complete_dates[:, None]
        & (event_denominator > 0.0)
    )
    node_scores["G06_tdnet_event_neighbor_spillover"] = event_score
    node_eligible["G06_tdnet_event_neighbor_spillover"] = event_eligible

    conditional_score = np.zeros_like(lead)
    conditional_eligible = np.zeros_like(lead, dtype=bool)
    conditional_reports = []
    for date_index in range(len(score_dates)):
        values, valid, report = conditional_response(
            graph["lead_history_message"],
            graph["history_returns"],
            lead[date_index],
        )
        conditional_score[date_index] = values
        conditional_eligible[date_index] = valid & node_available[date_index]
        conditional_reports.append(report)
    node_scores["G07_neighbor_conditioned_empirical_uplift"] = (
        conditional_score
    )
    node_eligible["G07_neighbor_conditioned_empirical_uplift"] = (
        conditional_eligible
    )

    community = graph["community"]
    current_community_shock = np.full(
        (len(score_dates), COMMUNITIES), np.nan, dtype=float
    )
    for label in range(COMMUNITIES):
        with np.errstate(invalid="ignore"):
            current_community_shock[:, label] = np.nanmean(
                oc_message[:, community == label], axis=1
            )
    community_score = np.zeros_like(lead)
    community_eligible = np.zeros_like(lead, dtype=bool)
    community_reports = []
    history_community = graph["community_history_shock"]
    for date_index in range(len(score_dates)):
        current_by_node = current_community_shock[
            date_index, community
        ]
        history_by_node = history_community[:, community]
        values, valid, report = conditional_response(
            history_by_node,
            graph["history_returns"],
            current_by_node,
        )
        community_score[date_index] = values
        community_eligible[date_index] = valid & node_available[date_index]
        community_reports.append(report)
    node_scores["G08_dynamic_community_archetype"] = community_score
    node_eligible["G08_dynamic_community_archetype"] = community_eligible

    lead_rank = rowwise_percentile_message(
        np.where(
            node_eligible["G03_directed_lead_lag_message"],
            lead,
            np.nan,
        )
    )
    overnight_rank = rowwise_percentile_message(
        np.where(
            node_eligible["G04_overnight_corr_message"],
            overnight,
            np.nan,
        )
    )
    agreement = (lead_rank + overnight_rank) / 2.0
    agreement_eligible = (
        node_available
        & np.isfinite(lead_rank)
        & np.isfinite(overnight_rank)
        & (lead_rank != 0.0)
        & (overnight_rank != 0.0)
        & (np.sign(lead_rank) == np.sign(overnight_rank))
    )
    node_scores["G10_graph_disagreement_cash"] = agreement
    node_eligible["G10_graph_disagreement_cash"] = agreement_eligible

    full_scores: dict[str, np.ndarray] = {}
    full_eligible: dict[str, np.ndarray] = {}
    for candidate_id in GRAPH_IDS:
        score_values, eligible_values = map_node_values_to_score_rows(
            score_rows,
            score_dates,
            nodes,
            node_scores[candidate_id],
            node_eligible[candidate_id],
        )
        full_scores[candidate_id] = score_values
        full_eligible[candidate_id] = eligible_values
    report = {
        "tdnet_complete_score_sessions": int(complete_dates.sum()),
        "tdnet_source_missing_score_sessions": int((~complete_dates).sum()),
        "candidate_available_rows": {
            name: int(values.sum()) for name, values in full_eligible.items()
        },
        "G07_sign_table": conditional_reports[0],
        "G08_sign_table": community_reports[0],
    }
    return full_scores, full_eligible, report


def fit_price_control(
    training: pd.DataFrame,
    columns: Sequence[str],
    candidate_id: str,
    period: pd.Period,
):
    return fit_research_model(
        ResearchModelSpec(
            name=f"v11_graph_{candidate_id}_{period}",
            family="ridge_daily_rank",
            objective="same_day_return_percentile",
            parameters={"alpha": 1.0},
        ),
        training,
        tuple(columns),
    )


def make_score_frame(
    rows: pd.DataFrame,
    score: Iterable[float],
    eligible: Iterable[bool],
    candidate_id: str,
) -> pd.DataFrame:
    output = rows[
        ["date", "code", "name", "label", "oc_return_pct"]
    ].copy()
    output["model_score"] = np.asarray(score, dtype=float)
    output["score_eligible"] = np.asarray(eligible, dtype=bool)
    output["candidate_id"] = candidate_id
    return output


def select_slots(
    scores: pd.DataFrame, sessions: pd.DatetimeIndex, top_k: int
) -> pd.DataFrame:
    ranked = (
        scores.loc[scores["score_eligible"]]
        .sort_values(
            ["date", "model_score", "code"],
            ascending=[True, False, True],
            kind="stable",
        )
        .groupby("date", sort=True)
        .head(top_k)
        .copy()
    )
    ranked["model_rank"] = ranked.groupby("date", sort=False).cumcount() + 1
    desired = pd.MultiIndex.from_product(
        [sessions, range(1, top_k + 1)], names=["date", "model_rank"]
    ).to_frame(index=False)
    selected = desired.merge(
        ranked[
            [
                "date",
                "model_rank",
                "code",
                "name",
                "model_score",
                "score_eligible",
                "label",
                "oc_return_pct",
                "candidate_id",
            ]
        ],
        on=["date", "model_rank"],
        how="left",
        validate="one_to_one",
    )
    selected["candidate_id"] = selected["candidate_id"].fillna(
        str(scores["candidate_id"].iloc[0])
    )
    selected["score_eligible"] = (
        selected["score_eligible"].fillna(False).astype(bool)
    )
    return selected


def daily_returns(
    picks: pd.DataFrame, top_k: int, cost_bps: int
) -> pd.Series:
    executed = picks["oc_return_pct"].notna()
    slot = (
        picks["oc_return_pct"].fillna(0.0)
        - executed.astype(float) * cost_bps / 100.0
    )
    return slot.groupby(picks["date"], sort=True).sum().div(top_k)


def metrics_for_picks(picks: pd.DataFrame, top_k: int) -> dict[str, Any]:
    daily_by_cost = {
        cost: daily_returns(picks, top_k, cost) for cost in COSTS
    }
    costs = {
        str(cost): {
            "mean_pct": float(values.mean()),
            "median_pct": float(values.median()),
            "win_rate": float(values.gt(0.0).mean()),
            "days": int(len(values)),
        }
        for cost, values in daily_by_cost.items()
    }
    periods = daily_by_cost[40].index.to_period("M")
    monthly = {}
    for period in periods.unique():
        mask = periods == period
        monthly[str(period)] = {
            f"net{cost}_mean_pct" if cost else "gross_mean_pct": float(
                values.loc[mask].mean()
            )
            for cost, values in daily_by_cost.items()
        }
    slices = {}
    for name, (start, end) in PERIODS.items():
        mask = (
            (daily_by_cost[40].index >= start)
            & (daily_by_cost[40].index <= end)
        )
        row: dict[str, Any] = {"days": int(mask.sum())}
        for cost, values in daily_by_cost.items():
            label = f"net{cost}_mean_pct" if cost else "gross_mean_pct"
            row[label] = float(values.loc[mask].mean())
        slices[name] = row

    daily20 = daily_by_cost[20]
    removed_dates = list(daily20.nlargest(min(20, len(daily20))).index)
    removed = daily20.drop(removed_dates)
    executed = picks["oc_return_pct"].notna()
    contribution = (
        picks["oc_return_pct"].fillna(0.0) - executed.astype(float) * 0.20
    ).div(top_k)
    by_code = contribution.groupby(picks["code"], dropna=True).sum().sort_values(
        ascending=False
    )
    top_profit_codes = [
        str(value) for value in by_code.loc[by_code.gt(0.0)].head(10).index
    ]
    cash_daily = (
        contribution.mask(picks["code"].isin(top_profit_codes), 0.0)
        .groupby(picks["date"], sort=True)
        .sum()
    )
    positive_total = float(by_code.loc[by_code.gt(0.0)].sum())
    largest_share = (
        float(by_code.iloc[0] / positive_total)
        if len(by_code) and by_code.iloc[0] > 0.0 and positive_total > 0.0
        else 0.0
    )
    selection_counts = picks["code"].dropna().astype(str).value_counts()
    top10_selection_share = (
        float(selection_counts.head(10).sum() / len(picks))
        if len(picks)
        else 0.0
    )
    return {
        "costs": costs,
        "monthly": monthly,
        "positive_months_net40": int(
            sum(row["net40_mean_pct"] > 0.0 for row in monthly.values())
        ),
        "months": int(len(monthly)),
        "slices": slices,
        "best20_session_removal": {
            "removed_dates": [
                str(pd.Timestamp(value).date()) for value in removed_dates
            ],
            "remaining_days": int(len(removed)),
            "net20_mean_pct": float(removed.mean()),
        },
        "top10_code_cash": {
            "codes": top_profit_codes,
            "net20_mean_pct": float(cash_daily.mean()),
        },
        "concentration": {
            "largest_positive_code_share": largest_share,
            "top10_selection_share": top10_selection_share,
            "unique_codes": int(picks["code"].nunique(dropna=True)),
        },
        "filled_slots": int(executed.sum()),
        "scheduled_slots": int(len(picks)),
        "cash_slots": int((~executed).sum()),
        "slot_fill_rate": float(executed.mean()),
        "days_with_at_least_one_filled_slot": int(
            executed.groupby(picks["date"], sort=True).any().sum()
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=ROOT / "research" / "model_v11_graph",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prefix = args.output_prefix.resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    protocol, integrity = load_protocol()
    panel_path = Path(protocol["frozen_inputs"]["panel"]["path"])
    projection = list(
        dict.fromkeys(
            [
                "date",
                "code",
                "name",
                "label",
                "oc_return_pct",
                "price_eligible",
                "price_training_eligible",
                "candidate_price_source_max_date",
                "no_trade_rate_20",
                "tdnet_source_complete",
                "tdnet_clean_feature_source_max_timestamp",
                "tdnet_clean_any",
                *L4_FEATURES,
            ]
        )
    )
    raw_panel = joblib.load(panel_path, mmap_mode="r")
    panel = raw_panel[projection].copy()
    del raw_panel
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    panel["code"] = panel["code"].astype(str)
    sessions = pd.DatetimeIndex(panel["date"].drop_duplicates().sort_values())
    score_sessions = sessions[
        (sessions >= SCORE_START) & (sessions <= SCORE_END)
    ]
    if len(score_sessions) != 266:
        raise RuntimeError(f"expected 266 score sessions, got {len(score_sessions)}")
    score_months = pd.PeriodIndex(
        score_sessions.to_period("M").unique()
    ).sort_values()
    if len(score_months) != 13:
        raise RuntimeError(f"expected 13 score months, got {len(score_months)}")

    score_parts: dict[str, list[pd.DataFrame]] = {
        candidate_id: [] for candidate_id in ALL_IDS
    }
    fold_reports: list[dict[str, Any]] = []
    mutation_reports: list[dict[str, Any]] = []

    for period in score_months:
        month_dates = score_sessions[score_sessions.to_period("M") == period]
        fold_start = month_dates.min()
        panel_train = panel.loc[
            panel["date"].lt(fold_start)
            & panel["price_training_eligible"]
            & panel["oc_return_pct"].notna()
        ].copy()
        panel_score = panel.loc[
            panel["date"].isin(month_dates) & panel["price_eligible"]
        ].copy()
        if panel_train.empty or panel_train["date"].max() >= fold_start:
            raise RuntimeError(f"non-PIT comparator training: {period}")
        price_source_violation = (
            panel_score["candidate_price_source_max_date"].notna()
            & panel_score["candidate_price_source_max_date"].ge(
                panel_score["date"]
            )
        )
        if price_source_violation.any():
            raise RuntimeError(f"score price source violation: {period}")

        c00_model = fit_price_control(
            panel_train, G0_FEATURES, "C00_daily_rank_ridge", period
        )
        c01_model = fit_price_control(
            panel_train, L4_FEATURES, "C01_L4_price_control", period
        )
        c00_score = np.asarray(c00_model.score(panel_score), dtype=float)
        c01_score = np.asarray(c01_model.score(panel_score), dtype=float)
        score_parts["C00_daily_rank_ridge"].append(
            make_score_frame(
                panel_score,
                c00_score,
                np.ones(len(panel_score), dtype=bool),
                "C00_daily_rank_ridge",
            )
        )
        score_parts["C01_L4_price_control"].append(
            make_score_frame(
                panel_score,
                c01_score,
                np.ones(len(panel_score), dtype=bool),
                "C01_L4_price_control",
            )
        )

        nodes, graph_dates, node_report = select_graph_nodes(
            panel, sessions, fold_start
        )
        graph, graph_report = build_fold_graph(
            panel, sessions, fold_start, nodes, graph_dates
        )
        graph_score, graph_eligible, score_report = score_graph_fold(
            panel_score, month_dates, nodes, graph
        )
        for candidate_id in GRAPH_IDS:
            score_parts[candidate_id].append(
                make_score_frame(
                    panel_score,
                    graph_score[candidate_id],
                    graph_eligible[candidate_id],
                    candidate_id,
                )
            )

        original_values = {
            "C00_daily_rank_ridge": c00_score,
            "C01_L4_price_control": c01_score,
            **graph_score,
            **{
                f"{name}__eligible": values.astype(np.uint8)
                for name, values in graph_eligible.items()
            },
        }
        original_digest = digest_arrays(original_values)
        mutated = panel_score.copy()
        mutated["oc_return_pct"] = (
            np.arange(len(mutated), dtype=float) * 1000.0 + 101.0
        )
        mutated["label"] = (
            np.arange(len(mutated), dtype=float) * -1000.0 - 303.0
        )
        mutated_c00 = np.asarray(c00_model.score(mutated), dtype=float)
        mutated_c01 = np.asarray(c01_model.score(mutated), dtype=float)
        mutated_graph_score, mutated_graph_eligible, _ = score_graph_fold(
            mutated, month_dates, nodes, graph
        )
        mutated_values = {
            "C00_daily_rank_ridge": mutated_c00,
            "C01_L4_price_control": mutated_c01,
            **mutated_graph_score,
            **{
                f"{name}__eligible": values.astype(np.uint8)
                for name, values in mutated_graph_eligible.items()
            },
        }
        mutated_digest = digest_arrays(mutated_values)
        if original_digest != mutated_digest:
            raise RuntimeError(f"target mutation changed scores: {period}")

        for candidate_id in ALL_IDS:
            if candidate_id == "C00_daily_rank_ridge":
                original_frame = make_score_frame(
                    panel_score,
                    c00_score,
                    np.ones(len(panel_score), dtype=bool),
                    candidate_id,
                )
                mutated_frame = make_score_frame(
                    mutated,
                    mutated_c00,
                    np.ones(len(mutated), dtype=bool),
                    candidate_id,
                )
            elif candidate_id == "C01_L4_price_control":
                original_frame = make_score_frame(
                    panel_score,
                    c01_score,
                    np.ones(len(panel_score), dtype=bool),
                    candidate_id,
                )
                mutated_frame = make_score_frame(
                    mutated,
                    mutated_c01,
                    np.ones(len(mutated), dtype=bool),
                    candidate_id,
                )
            else:
                original_frame = make_score_frame(
                    panel_score,
                    graph_score[candidate_id],
                    graph_eligible[candidate_id],
                    candidate_id,
                )
                mutated_frame = make_score_frame(
                    mutated,
                    mutated_graph_score[candidate_id],
                    mutated_graph_eligible[candidate_id],
                    candidate_id,
                )
            for top_k in (1, 2):
                original_keys = select_slots(
                    original_frame, month_dates, top_k
                )[["date", "model_rank", "code"]]
                mutated_keys = select_slots(
                    mutated_frame, month_dates, top_k
                )[["date", "model_rank", "code"]]
                if not original_keys.equals(mutated_keys):
                    raise RuntimeError(
                        f"target mutation changed selection: "
                        f"{period}/{candidate_id}/top{top_k}"
                    )
        mutation_reports.append(
            {
                "period": str(period),
                "original_score_and_fill_digest": original_digest,
                "mutated_score_and_fill_digest": mutated_digest,
                "all_top1_top2_selection_keys_exact": True,
                "passes": True,
            }
        )
        fold_reports.append(
            {
                "period": str(period),
                "score_session_first": str(month_dates.min().date()),
                "score_session_last": str(month_dates.max().date()),
                "score_sessions": int(len(month_dates)),
                "panel_train_rows": int(len(panel_train)),
                "panel_train_end": str(panel_train["date"].max().date()),
                "panel_score_rows": int(len(panel_score)),
                "price_source_violations": int(price_source_violation.sum()),
                "nodes": node_report,
                "graph": graph_report,
                "score": score_report,
                "strictly_prior_all": bool(
                    panel_train["date"].max() < fold_start
                    and graph_dates.max() < fold_start
                    and graph_report["strictly_prior_graph"]
                    and graph_report["strictly_prior_conditional_history"]
                ),
            }
        )
        del (
            panel_train,
            panel_score,
            nodes,
            graph,
            c00_model,
            c01_model,
            graph_score,
            graph_eligible,
            mutated_graph_score,
            mutated_graph_eligible,
        )
        gc.collect()

    scores = {
        candidate_id: pd.concat(parts, ignore_index=True)
        for candidate_id, parts in score_parts.items()
    }
    all_picks: list[pd.DataFrame] = []
    metrics: dict[str, Any] = {}
    for candidate_id in ALL_IDS:
        metrics[candidate_id] = {}
        for top_k in (1, 2):
            picks = select_slots(
                scores[candidate_id], score_sessions, top_k
            )
            picks["top_k"] = top_k
            all_picks.append(picks)
            metrics[candidate_id][f"top{top_k}"] = metrics_for_picks(
                picks, top_k
            )
    picks_frame = pd.concat(all_picks, ignore_index=True)
    picks_path = Path(f"{prefix}_picks.csv")
    picks_frame.to_csv(
        picks_path,
        index=False,
        date_format="%Y-%m-%d",
        float_format="%.17g",
        lineterminator="\n",
    )
    result = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "authority": {
            "classification": "retrospective_exploratory_no_production_promotion",
            "maximum_decision": "forward_shadow_only",
            "production_model_changed": False,
            "orders_allowed": False,
        },
        "protocol_sha256": integrity["protocol_sha256"],
        "runner_sha256": sha256_file(Path(__file__)),
        "integrity": {
            **integrity,
            "strictly_prior_all_folds": all(
                fold["strictly_prior_all"] for fold in fold_reports
            ),
            "price_source_violations": int(
                sum(fold["price_source_violations"] for fold in fold_reports)
            ),
            "target_session_outcome_mutation": {
                "passes": all(row["passes"] for row in mutation_reports),
                "folds": mutation_reports,
            },
            "z17_same_date_peer_residual_target_reimplemented": False,
            "external_unavailable_sources_proxied": False,
        },
        "coverage": {
            "panel_sessions": int(len(sessions)),
            "score_sessions": int(len(score_sessions)),
            "score_months": [str(value) for value in score_months],
            "score_start": str(score_sessions.min().date()),
            "score_end": str(score_sessions.max().date()),
        },
        "implementation": {
            "graph_candidate_ids": list(GRAPH_IDS),
            "comparator_ids": list(COMPARATOR_IDS),
            "graph_window_sessions": GRAPH_WINDOW,
            "minimum_observed_returns": MIN_OBSERVED,
            "node_capacity": NODE_CAPACITY,
            "neighbor_count": NEIGHBORS,
            "community_count": COMMUNITIES,
            "hyperparameter_or_threshold_grid_searched": False,
            "graph_candidate_fitted_regression_coefficients": False,
            "fail_closed_external_hypotheses": [
                item["id"]
                for item in protocol[
                    "external_data_limit_and_fail_closed_hypotheses"
                ]
            ],
        },
        "folds": fold_reports,
        "metrics": metrics,
        "known_z17_reference": {
            "top1_net40_mean_pct": -0.4964166195142521,
            "top2_net40_mean_pct": -0.49913787107355917,
            "paired_in_familywise_inference": False,
        },
        "decision": {
            "status": "pending_independent_audit",
            "production_model_changed": False,
        },
        "artifacts": {
            "picks_csv": str(picks_path),
            "picks_sha256": sha256_file(picks_path),
        },
    }
    result_path = Path(f"{prefix}_result.json")
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "result": str(result_path),
                "picks": str(picks_path),
                "score_sessions": len(score_sessions),
                "score_months": len(score_months),
                "graph_candidates": len(GRAPH_IDS),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
