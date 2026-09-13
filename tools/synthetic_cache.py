"""Cache infrastructure for synthetic dataset generators.

Generated synthetic data (per-prompt hidden functions, feature pools, demo/query
selections) is cached under the HuggingFace ``datasets`` cache root, keyed by SHA256
of the canonical hyperparameter JSON. Cache hit on identical hyperparameters returns
prior data; cache miss generates and persists.

All cache files are JSON / JSONL (not pickle) so they are human-inspectable,
version-stable across Python versions, and safe to load (no arbitrary-code-execution
risk from pickle).

Cache layout::

    <HF_DATASETS_CACHE>/icl_synthetic/<task_name>/<hash>/
        config.json        # hyperparameters (input to the hash)
        feature_pool.json  # feature value pool actually used
        prompts.jsonl      # one prompt per line (hidden W, demos, query)
        meta.json          # generator_module, generator_version, timestamp

Honors ``HF_DATASETS_CACHE`` / ``HF_HOME`` env vars via the ``datasets`` library.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import time
from typing import Any, Callable, Iterable

try:
    import datasets  # type: ignore
    _HF_DATASETS_AVAILABLE = True
except ImportError:
    _HF_DATASETS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Cache root resolution
# ---------------------------------------------------------------------------

def get_cache_root() -> pathlib.Path:
    """Return the ``icl_synthetic`` cache root, piggy-backing the HF datasets cache.

    Honors ``HF_DATASETS_CACHE`` and ``HF_HOME`` env vars. Falls back to
    ``~/.cache/huggingface/datasets/icl_synthetic`` if ``datasets`` library isn't
    available.
    """
    if _HF_DATASETS_AVAILABLE:
        base = pathlib.Path(datasets.config.HF_DATASETS_CACHE)
    else:
        hf_cache = os.environ.get("HF_DATASETS_CACHE")
        if hf_cache is None:
            hf_home = os.environ.get(
                "HF_HOME",
                os.path.expanduser("~/.cache/huggingface"),
            )
            hf_cache = os.path.join(hf_home, "datasets")
        base = pathlib.Path(hf_cache)
    return base / "icl_synthetic"


# ---------------------------------------------------------------------------
# Cache key / path
# ---------------------------------------------------------------------------

def canonical_config_json(config: dict[str, Any]) -> str:
    """Canonical JSON string for hashing: sorted keys, no whitespace variation."""
    return json.dumps(config, sort_keys=True, separators=(",", ":"))


def compute_cache_hash(config: dict[str, Any], hex_len: int = 16) -> str:
    """SHA256 of the canonical config JSON, truncated to ``hex_len`` hex chars."""
    digest = hashlib.sha256(canonical_config_json(config).encode("utf-8")).hexdigest()
    return digest[:hex_len]


def get_cache_dir(task_name: str, config: dict[str, Any]) -> pathlib.Path:
    """Resolve the cache subdirectory for ``(task_name, config)``. Does not create it."""
    return get_cache_root() / task_name / compute_cache_hash(config)


# ---------------------------------------------------------------------------
# Atomic write / load
# ---------------------------------------------------------------------------

_EXPECTED_FILES = ("config.json", "feature_pool.json", "prompts.jsonl", "meta.json")


def cache_exists(cache_dir: pathlib.Path) -> bool:
    """True iff ``cache_dir`` is a directory containing all expected files."""
    return cache_dir.is_dir() and all(
        (cache_dir / name).is_file() for name in _EXPECTED_FILES
    )


def _atomic_write_text(path: pathlib.Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically via tempfile + rename."""
    tmp = path.parent / (path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    tmp.replace(path)


def load_cache(cache_dir: pathlib.Path) -> dict[str, Any]:
    """Load all cache files. Returns dict with keys: ``config``, ``feature_pool``,
    ``prompts``, ``meta``."""
    with open(cache_dir / "config.json", encoding="utf-8") as f:
        config = json.load(f)
    with open(cache_dir / "feature_pool.json", encoding="utf-8") as f:
        feature_pool = json.load(f)
    prompts: list[dict[str, Any]] = []
    with open(cache_dir / "prompts.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            prompts.append(json.loads(line))
    with open(cache_dir / "meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    return {
        "config": config,
        "feature_pool": feature_pool,
        "prompts": prompts,
        "meta": meta,
    }


def save_cache(
    cache_dir: pathlib.Path,
    *,
    config: dict[str, Any],
    feature_pool: dict[str, Any],
    prompts: Iterable[dict[str, Any]],
    generator_module: str,
    generator_version: str = "1",
    extra_meta: dict[str, Any] | None = None,
) -> None:
    """Save all cache files atomically. Each file is written to ``<name>.tmp`` then
    renamed, so a partial write does not corrupt an existing cache.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    _atomic_write_text(
        cache_dir / "config.json",
        json.dumps(config, sort_keys=True, indent=2) + "\n",
    )
    _atomic_write_text(
        cache_dir / "feature_pool.json",
        json.dumps(feature_pool, sort_keys=True, indent=2) + "\n",
    )

    prompt_lines = [json.dumps(p, separators=(",", ":")) for p in prompts]
    _atomic_write_text(
        cache_dir / "prompts.jsonl",
        "\n".join(prompt_lines) + "\n",
    )

    meta = {
        "generator_module": generator_module,
        "generator_version": generator_version,
        "generated_at_unix": int(time.time()),
        "cache_hash": compute_cache_hash(config),
    }
    if extra_meta:
        meta.update(extra_meta)
    _atomic_write_text(
        cache_dir / "meta.json",
        json.dumps(meta, sort_keys=True, indent=2) + "\n",
    )


# ---------------------------------------------------------------------------
# High-level: get-or-generate
# ---------------------------------------------------------------------------

GeneratorFn = Callable[..., tuple[dict[str, Any], list[dict[str, Any]]]]
"""Generator signature: ``generator_fn(config=...) -> (feature_pool, prompts)``."""


def get_or_generate(
    task_name: str,
    config: dict[str, Any],
    generator_fn: GeneratorFn,
    generator_module: str,
    generator_version: str = "1",
    force_regenerate: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    """Load cache if it exists, otherwise call ``generator_fn(config=...)`` and persist.

    Returns ``{"config", "feature_pool", "prompts", "meta"}``.
    """
    cache_dir = get_cache_dir(task_name, config)

    if not force_regenerate and cache_exists(cache_dir):
        if verbose:
            print(f"[synthetic_cache] cache HIT: {cache_dir}")
        return load_cache(cache_dir)

    if verbose:
        print(f"[synthetic_cache] cache MISS: generating to {cache_dir}")

    feature_pool, prompts = generator_fn(config=config)
    prompts = list(prompts)
    save_cache(
        cache_dir,
        config=config,
        feature_pool=feature_pool,
        prompts=prompts,
        generator_module=generator_module,
        generator_version=generator_version,
    )

    return {
        "config": config,
        "feature_pool": feature_pool,
        "prompts": prompts,
        "meta": {
            "generator_module": generator_module,
            "generator_version": generator_version,
            "cache_hash": compute_cache_hash(config),
        },
    }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"cache root            = {get_cache_root()}")
    test_config = {"task": "synthetic_test", "seed": 42, "n_classes": 6}
    print(f"canonical JSON        = {canonical_config_json(test_config)}")
    print(f"hash (16 hex)         = {compute_cache_hash(test_config)}")
    print(f"cache dir for example = {get_cache_dir('synthetic_test', test_config)}")
