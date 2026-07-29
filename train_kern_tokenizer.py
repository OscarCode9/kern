"""Train a reproducible 16K byte-level BPE for Kern compact source.

The training corpus is derived from the repository-disjoint CodeSearchNet
Python train partition.  CodeSearchNet validation is used only to select the
pre-tokenizer configuration.  Modern benchmark programs and the published
Toke/Python pairs are excluded by both source and normalized-AST hashes where
the public artifacts make those hashes available.

Raw third-party source is never written to the repository.  The output is a
tokenizer JSON plus a manifest containing provenance, configuration, hashes,
and aggregate filtering statistics.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.metadata
import json
import platform
import warnings
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import hf_hub_download
from pyarrow import parquet
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from benchmark_modern import (
    load_bigcodebench_tasks,
    load_evalplus_tasks,
    normalize_ast,
)
from kern_compact import compact_tree
from kern_compiler import compile_kern
from kern_transpiler import transpile

CODESEARCHNET_REPO = "code-search-net/code_search_net"
CODESEARCHNET_REVISION = "bd0cf261e357a3eb5c8fba490d23ec1a1cd59555"
CODESEARCHNET_FILES = {
    "train": {
        "path": "python/train-00000-of-00001.parquet",
        "sha256": "ad9e3a4ab10c2c1d8926d2b26ca2bfcc3aadda1477ba29a933391f93806b9fed",
        "rows": 412_178,
    },
    "validation": {
        "path": "python/validation-00000-of-00001.parquet",
        "sha256": "22eaacb46ed7e74d582409b85692ef63f5a43e99f9395c2eb736b5c8451422bb",
        "rows": 23_107,
    },
}
DEFAULT_TRAIN_PROGRAMS = 25_953
DEFAULT_VALIDATION_PROGRAMS = 2_048
DEFAULT_VOCAB_SIZE = 16_384
DEFAULT_MAX_SOURCE_CHARS = 20_000
SPECIAL_TOKENS = ("<unk>",)
PRETOKENIZER_CANDIDATES = (
    "bytelevel_regex",
    "bytelevel_no_regex",
)


@dataclass(frozen=True)
class Exclusions:
    source_sha256: frozenset[str]
    ast_sha256: frozenset[str]
    modern_programs: int
    toke_pair_hashes: int


@dataclass
class CorpusStats:
    split: str
    requested: int
    scanned: int
    accepted: int
    accepted_characters: int
    accepted_repositories: int
    rejections: dict[str, int]
    source_aggregate_sha256: str
    kern_aggregate_sha256: str


@dataclass(frozen=True)
class Corpus:
    kern_sources: tuple[str, ...]
    source_sha256: frozenset[str]
    ast_sha256: frozenset[str]
    kern_sha256: frozenset[str]
    repositories: frozenset[str]
    stats: CorpusStats


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def aggregate_hash(values: Iterable[str]) -> str:
    """Hash an ordered string sequence without concatenation ambiguity."""
    digest = hashlib.sha256()
    for value in values:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def normalize_source(source: str) -> str:
    return source.replace("\r\n", "\n").replace("\r", "\n").strip() + "\n"


def expected_compact_ast(tree: ast.AST) -> str:
    compacted = compact_tree(tree)
    return normalize_ast(ast.unparse(compacted))


def load_exclusions(
    toke_details: Path,
    toke_eval: Path | None = None,
) -> Exclusions:
    source_hashes: set[str] = set()
    ast_hashes: set[str] = set()
    tasks = load_evalplus_tasks() + load_bigcodebench_tasks()
    for task in tasks:
        source = normalize_source(task.source)
        source_hashes.add(sha256_text(source))
        ast_hashes.add(sha256_text(normalize_ast(source)))

    toke_hashes = 0
    if toke_details.exists():
        with toke_details.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                value = row.get("python_sha256", "")
                if value:
                    source_hashes.add(value)
                    toke_hashes += 1

    if toke_eval is not None:
        from benchmark_toke import build_python_programs

        solutions = (
            toke_eval
            / "benchmark"
            / "baselines"
            / "python"
            / "solutions.py"
        )
        programs = build_python_programs(solutions)
        if len(programs) != 60:
            raise RuntimeError(
                "Pinned Toke exclusion corpus must contain 60 programs."
            )
        for source in programs.values():
            normalized_source = normalize_source(source)
            source_hashes.add(sha256_text(normalized_source))
            ast_hashes.add(sha256_text(normalize_ast(normalized_source)))

    return Exclusions(
        source_sha256=frozenset(source_hashes),
        ast_sha256=frozenset(ast_hashes),
        modern_programs=len(tasks),
        toke_pair_hashes=toke_hashes,
    )


def download_split(split: str) -> Path:
    metadata = CODESEARCHNET_FILES[split]
    path = Path(
        hf_hub_download(
            repo_id=CODESEARCHNET_REPO,
            repo_type="dataset",
            revision=CODESEARCHNET_REVISION,
            filename=str(metadata["path"]),
        )
    )
    actual_hash = sha256_file(path)
    if actual_hash != metadata["sha256"]:
        raise RuntimeError(
            f"{split} Parquet SHA-256 mismatch: {actual_hash} "
            f"!= {metadata['sha256']}"
        )
    row_count = parquet.ParquetFile(path).metadata.num_rows
    if row_count != metadata["rows"]:
        raise RuntimeError(
            f"{split} row count mismatch: {row_count} != {metadata['rows']}"
        )
    return path


def parquet_rows(path: Path) -> Iterator[tuple[str, str]]:
    columns = ["repository_name", "whole_func_string"]
    source = parquet.ParquetFile(path)
    for batch in source.iter_batches(batch_size=1_024, columns=columns):
        repositories = batch.column(0).to_pylist()
        programs = batch.column(1).to_pylist()
        yield from zip(repositories, programs, strict=True)


def build_corpus(
    *,
    split: str,
    path: Path,
    requested: int,
    max_source_chars: int,
    exclusions: Exclusions,
    prior_source_hashes: frozenset[str] = frozenset(),
    prior_ast_hashes: frozenset[str] = frozenset(),
    prior_kern_hashes: frozenset[str] = frozenset(),
) -> Corpus:
    rejection_counts: Counter[str] = Counter()
    kern_sources: list[str] = []
    accepted_sources: list[str] = []
    source_hashes: set[str] = set()
    ast_hashes: set[str] = set()
    kern_hashes: set[str] = set()
    repositories: set[str] = set()
    scanned = 0

    for repository, raw_source in parquet_rows(path):
        if len(kern_sources) >= requested:
            break
        scanned += 1
        if not isinstance(raw_source, str) or not raw_source.strip():
            rejection_counts["empty_source"] += 1
            continue
        source = normalize_source(raw_source)
        if len(source) > max_source_chars:
            rejection_counts["source_too_long"] += 1
            continue

        source_hash = sha256_text(source)
        if source_hash in exclusions.source_sha256:
            rejection_counts["evaluation_source_excluded"] += 1
            continue
        if source_hash in source_hashes or source_hash in prior_source_hashes:
            rejection_counts["duplicate_source"] += 1
            continue

        try:
            tree = ast.parse(source)
            normalized = normalize_ast(source)
        except (SyntaxError, ValueError, TypeError):
            rejection_counts["python_parse"] += 1
            continue

        ast_hash = sha256_text(normalized)
        if ast_hash in exclusions.ast_sha256:
            rejection_counts["evaluation_ast_excluded"] += 1
            continue
        if ast_hash in ast_hashes or ast_hash in prior_ast_hashes:
            rejection_counts["duplicate_ast"] += 1
            continue

        try:
            kern = transpile(source, compact=True).strip()
            if "# UNSUPPORTED:" in kern:
                rejection_counts["unsupported_node"] += 1
                continue
            decoded = compile_kern(kern)
            if normalize_ast(decoded) != expected_compact_ast(tree):
                rejection_counts["roundtrip_ast"] += 1
                continue
        # Corpus construction is a compatibility filter: one unsupported
        # third-party program must not abort the pinned deterministic scan.
        except Exception:  # noqa: BLE001
            rejection_counts["transform_or_compile"] += 1
            continue

        kern_hash = sha256_text(kern)
        if kern_hash in kern_hashes or kern_hash in prior_kern_hashes:
            rejection_counts["duplicate_kern"] += 1
            continue

        kern_sources.append(kern)
        accepted_sources.append(source)
        source_hashes.add(source_hash)
        ast_hashes.add(ast_hash)
        kern_hashes.add(kern_hash)
        repositories.add(str(repository))

    if len(kern_sources) != requested:
        raise RuntimeError(
            f"{split}: accepted {len(kern_sources)} of {requested} requested "
            f"after scanning {scanned} rows"
        )

    stats = CorpusStats(
        split=split,
        requested=requested,
        scanned=scanned,
        accepted=len(kern_sources),
        accepted_characters=sum(map(len, kern_sources)),
        accepted_repositories=len(repositories),
        rejections=dict(sorted(rejection_counts.items())),
        source_aggregate_sha256=aggregate_hash(accepted_sources),
        kern_aggregate_sha256=aggregate_hash(kern_sources),
    )
    return Corpus(
        kern_sources=tuple(kern_sources),
        source_sha256=frozenset(source_hashes),
        ast_sha256=frozenset(ast_hashes),
        kern_sha256=frozenset(kern_hashes),
        repositories=frozenset(repositories),
        stats=stats,
    )


def create_tokenizer(candidate: str) -> Tokenizer:
    if candidate not in PRETOKENIZER_CANDIDATES:
        raise ValueError(f"Unknown pre-tokenizer candidate: {candidate}")
    tokenizer = Tokenizer(
        models.BPE(
            unk_token=SPECIAL_TOKENS[0],
            byte_fallback=True,
        )
    )
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False,
        use_regex=candidate == "bytelevel_regex",
    )
    tokenizer.decoder = decoders.ByteLevel()
    return tokenizer


def train_candidate(
    *,
    candidate: str,
    sources: tuple[str, ...],
    vocab_size: int,
) -> Tokenizer:
    tokenizer = create_tokenizer(candidate)
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=list(SPECIAL_TOKENS),
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        max_token_length=64,
        show_progress=True,
    )
    tokenizer.train_from_iterator(
        sources,
        trainer=trainer,
        length=len(sources),
    )
    actual_vocab = tokenizer.get_vocab_size(with_added_tokens=True)
    if actual_vocab != vocab_size:
        raise RuntimeError(
            f"{candidate}: trained vocabulary {actual_vocab} != {vocab_size}"
        )
    return tokenizer


def exact_roundtrip_failures(
    tokenizer: Tokenizer,
    sources: Iterable[str],
) -> int:
    return sum(
        tokenizer.decode(tokenizer.encode(source).ids) != source
        for source in sources
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_results/native-tokenizer"),
    )
    parser.add_argument("--train-programs", type=int, default=DEFAULT_TRAIN_PROGRAMS)
    parser.add_argument(
        "--validation-programs",
        type=int,
        default=DEFAULT_VALIDATION_PROGRAMS,
    )
    parser.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE)
    parser.add_argument(
        "--max-source-chars",
        type=int,
        default=DEFAULT_MAX_SOURCE_CHARS,
    )
    parser.add_argument(
        "--toke-details",
        type=Path,
        default=Path(
            "benchmark_results/toke/toke-public-pair-details.csv"
        ),
    )
    parser.add_argument(
        "--toke-eval",
        type=Path,
        help=(
            "Optional pinned toke-eval checkout; when supplied, exclude "
            "normalized ASTs for all 60 public Python pairs as well as the "
            "published exact source hashes."
        ),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings("ignore", category=SyntaxWarning)

    exclusions = load_exclusions(args.toke_details, args.toke_eval)
    train_path = download_split("train")
    validation_path = download_split("validation")

    print("Building repository-disjoint CodeSearchNet training corpus...")
    training = build_corpus(
        split="train",
        path=train_path,
        requested=args.train_programs,
        max_source_chars=args.max_source_chars,
        exclusions=exclusions,
    )
    print(asdict(training.stats))

    print("Building independent tokenizer-selection corpus...")
    validation = build_corpus(
        split="validation",
        path=validation_path,
        requested=args.validation_programs,
        max_source_chars=args.max_source_chars,
        exclusions=exclusions,
        prior_source_hashes=training.source_sha256,
        prior_ast_hashes=training.ast_sha256,
        prior_kern_hashes=training.kern_sha256,
    )
    print(asdict(validation.stats))
    repository_overlap = training.repositories.intersection(
        validation.repositories
    )
    if repository_overlap:
        raise RuntimeError(
            "CodeSearchNet train/validation repository overlap: "
            f"{sorted(repository_overlap)[:5]}"
        )

    candidate_results: list[dict[str, object]] = []
    trained: dict[str, Tokenizer] = {}
    for candidate in PRETOKENIZER_CANDIDATES:
        print(f"Training candidate: {candidate}")
        tokenizer = train_candidate(
            candidate=candidate,
            sources=training.kern_sources,
            vocab_size=args.vocab_size,
        )
        validation_tokens = sum(
            len(tokenizer.encode(source).ids)
            for source in validation.kern_sources
        )
        failures = exact_roundtrip_failures(
            tokenizer,
            (
                *training.kern_sources[:256],
                *validation.kern_sources,
            ),
        )
        candidate_results.append(
            {
                "name": candidate,
                "validation_tokens": validation_tokens,
                "validation_characters": validation.stats.accepted_characters,
                "characters_per_token": (
                    validation.stats.accepted_characters / validation_tokens
                ),
                "exact_roundtrip_failures": failures,
            }
        )
        trained[candidate] = tokenizer

    winner = min(
        candidate_results,
        key=lambda row: (
            int(row["exact_roundtrip_failures"]),
            int(row["validation_tokens"]),
            str(row["name"]),
        ),
    )
    if winner["exact_roundtrip_failures"] != 0:
        raise RuntimeError(f"Winning tokenizer is not lossless: {winner}")

    tokenizer_path = args.output_dir / "kern-16k-tokenizer.json"
    tokenizer = trained[str(winner["name"])]
    tokenizer.save(str(tokenizer_path), pretty=True)
    tokenizer_sha = sha256_file(tokenizer_path)

    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "contract": (
            "16K byte-level BPE trained only on valid Kern compact "
            "round-trips; validation selects configuration; final benchmark "
            "corpora are held out"
        ),
        "dataset": {
            "repository": CODESEARCHNET_REPO,
            "revision": CODESEARCHNET_REVISION,
            "partition_rule": (
                "official repository-level train/validation partitions"
            ),
            "files": CODESEARCHNET_FILES,
        },
        "exclusions": {
            "modern_programs": exclusions.modern_programs,
            "toke_pair_source_hashes": exclusions.toke_pair_hashes,
            "toke_pair_ast_hashes": (
                60 if args.toke_eval is not None else 0
            ),
            "unique_source_hashes": len(exclusions.source_sha256),
            "unique_ast_hashes": len(exclusions.ast_sha256),
            "rule": (
                "exclude exact normalized-source SHA-256 and normalized-AST "
                "SHA-256 before training/selection; a pinned toke-eval "
                "checkout additionally supplies all 60 paired ASTs"
            ),
        },
        "training": asdict(training.stats),
        "selection": {
            "corpus": asdict(validation.stats),
            "repository_overlap_with_training": 0,
            "candidates": candidate_results,
            "winner": winner["name"],
            "selection_metric": (
                "fewest exact-roundtrip failures, then fewest aggregate "
                "validation tokens, then lexical candidate name"
            ),
        },
        "tokenizer": {
            "file": tokenizer_path.name,
            "sha256": tokenizer_sha,
            "vocab_size": tokenizer.get_vocab_size(with_added_tokens=True),
            "model": "BPE",
            "byte_fallback": True,
            "special_tokens": list(SPECIAL_TOKENS),
            "min_frequency": 2,
            "max_token_length": 64,
            "pre_tokenizer": winner["name"],
            "decoder": "ByteLevel",
            "normalizer": None,
        },
        "runtime": {
            "python": platform.python_version(),
            "tokenizers": package_version("tokenizers"),
            "pyarrow": package_version("pyarrow"),
            "huggingface_hub": package_version("huggingface-hub"),
            "datasets": package_version("datasets"),
        },
    }
    manifest_path = args.output_dir / "kern-16k-training-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"Winner: {winner['name']} "
        f"({winner['validation_tokens']} validation tokens)"
    )
    print(f"Tokenizer: {tokenizer_path.resolve()} ({tokenizer_sha})")
    print(f"Manifest: {manifest_path.resolve()}")


if __name__ == "__main__":
    main()
