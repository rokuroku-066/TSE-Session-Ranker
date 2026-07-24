#!/usr/bin/env python3
"""Create a deterministic SHA-256 hash-chain manifest for canonical JSON."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


CHAIN_DOMAIN = b"model-v09-canonical-json-chain-v1"
DEFAULT_ARTIFACTS = (
    "model_v09_protocol_ledger.json",
    "model_v09_result.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def chain_step(previous: str, logical_path: str, content_sha256: str) -> str:
    payload = (
        CHAIN_DOMAIN
        + b"\n"
        + previous.encode("ascii")
        + b"\n"
        + logical_path.encode("utf-8")
        + b"\n"
        + content_sha256.encode("ascii")
        + b"\n"
    )
    return hashlib.sha256(payload).hexdigest()


def build_manifest(
    root: Path,
    output: Path,
    artifact_names: tuple[str, ...] = DEFAULT_ARTIFACTS,
) -> dict[str, Any]:
    previous = hashlib.sha256(CHAIN_DOMAIN).hexdigest()
    entries: list[dict[str, Any]] = []
    for sequence, name in enumerate(artifact_names, start=1):
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(path)
        content_sha256 = sha256_file(path)
        current = chain_step(previous, name, content_sha256)
        entries.append(
            {
                "sequence": sequence,
                "path": name,
                "sha256": content_sha256,
                "previous_chain_sha256": previous,
                "chain_sha256": current,
            }
        )
        previous = current
    manifest = {
        "schema_version": 1,
        "artifact_id": "model_v09_hash_chain_manifest_20260723",
        "algorithm": "sha256",
        "chain_domain": CHAIN_DOMAIN.decode("ascii"),
        "chain_seed_sha256": hashlib.sha256(CHAIN_DOMAIN).hexdigest(),
        "artifacts": entries,
        "final_chain_sha256": previous,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=False)
        handle.write("\n")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Defaults to ROOT/model_v09_manifest.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or args.root / "model_v09_manifest.json"
    manifest = build_manifest(args.root, output)
    print(output)
    print(manifest["final_chain_sha256"])


if __name__ == "__main__":
    main()
