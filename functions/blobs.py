"""Lazy blob store: the local data/ tree, backed by the GCS buckets.

The training blobs (tensor_cache/*.npy), the metadata CSVs and the model
artifacts all live in GCS. This module makes any machine trainable with ZERO
manual downloads: ask for what a training session needs, and only the files
NOT already on disk are fetched. A laptop that has everything downloads
nothing; a fresh clone pulls exactly the blobs its model requires.

Auth: the gcloud CLI (`gcloud auth login`, project savetheknees).
Buckets are overridable for forks/tests via STK_DATA_BUCKET / STK_MODEL_BUCKET.
"""
import os
import shutil
import subprocess

from functions import paths

DATA_BUCKET = os.environ.get("STK_DATA_BUCKET", "gs://knees-images").rstrip("/")
MODEL_BUCKET = os.environ.get("STK_MODEL_BUCKET", "gs://knees-models").rstrip("/")


def _gcloud():
    exe = shutil.which("gcloud")
    if exe is None:
        raise RuntimeError("gcloud CLI not found -- install it, then run "
                           "`gcloud auth login` (https://cloud.google.com/sdk/docs/install)")
    return exe


def _fetch(urls, dest_dir):
    """Batch-download URLs into dest_dir (gcloud reads the list on stdin)."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    joined = chr(10).join(urls)
    return subprocess.run([_gcloud(), "storage", "cp", "-I", str(dest_dir)],
                          input=joined, text=True, capture_output=True)


def ensure(pairs, label="blobs"):
    """[(gs_url, local_path)] -> download the missing ones, grouped by folder.

    Returns the paths still missing afterwards. A blob absent from the bucket
    too (gcloud: "matched no objects") is tolerated -- e.g. the one unreadable
    axial study. Any OTHER gcloud failure with zero blobs recovered raises,
    so a trainer never starts on a silently incomplete cache.
    """
    missing = [(u, p) for u, p in pairs if not p.exists()]
    if not missing:
        return []
    by_dir = {}
    for url, path in missing:
        by_dir.setdefault(path.parent, []).append(url)
    print(f"blobs: {len(missing)}/{len(pairs)} {label} missing locally -> "
          f"fetching from {DATA_BUCKET.split('/')[-1]}...")
    last = None
    for dest, urls in by_dir.items():
        last = _fetch(urls, dest)
    still = [p for _, p in missing if not p.exists()]
    hard_fail = (last is not None and last.returncode != 0
                 and "matched no objects" not in (last.stderr or ""))
    if len(still) == len(missing) and hard_fail:
        raise RuntimeError("no blob could be fetched -- gcloud says: "
                           + (last.stderr or "").strip()[-500:])
    if still:
        print(f"blobs: {len(still)} not in the bucket either (skipped): "
              + ", ".join(p.name for p in still[:3]) + ("..." if len(still) > 3 else ""))
    return still


def ensure_cache(uids, axes=("X",), cache=None):
    """Every {uid}_{axis}.npy a training session will read, fetched if absent."""
    cache = paths.CACHE if cache is None else cache
    pairs = [(f"{DATA_BUCKET}/tensor_cache/{u}_{a}.npy", cache / f"{u}_{a}.npy")
             for u in uids for a in axes]
    return ensure(pairs, label="tensors")


def ensure_meta(rel_paths):
    """CSV/TXT metadata under data/ (e.g. 'meta/reports_en.csv', 'train.csv')."""
    pairs = [(f"{DATA_BUCKET}/{r}", paths.DATA / r) for r in rel_paths]
    return ensure(pairs, label="metadata")


def ensure_artifacts(names):
    """Trained artifacts under models/ (warm-start checkpoints, vectorizer)."""
    pairs = [(f"{MODEL_BUCKET}/{n}", paths.REPO_ROOT / "models" / n) for n in names]
    return ensure(pairs, label="artifacts")
