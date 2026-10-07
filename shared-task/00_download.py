#!/usr/bin/env python3
"""
Download the SIMBig 2026 Task 1 corpora from Mozilla Data Collective.

Supersedes data/download_data.py, whose dataset IDs point at the retired
v25 / v3 releases and now return 404. These are the IDs linked from the
shared-task announcement of 2026-09-17.

Setup:
    cp shared-task/.env.example .env      # then fill in MDC_API_KEY
    python shared-task/00_download.py

Get the key from https://mozilladatacollective.com -> Profile -> API.
Already-extracted datasets are skipped, so re-running is cheap.
"""

import argparse
import os
import sys
import tarfile
import zipfile
from pathlib import Path

import requests
from dotenv import load_dotenv
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# Both hostnames serve the same API; the first that answers wins.
API_HOSTS = [
    "https://mozilladatacollective.com/api/datasets",
    "https://datacollective.mozillafoundation.org/api/datasets",
]

DATASETS = {
    "scripted": {
        "id": "cmu62ctgp00o9nq07canzmu07",
        "name": "Common Voice Scripted Speech 27.0 - Puno Quechua",
        "approx_gb": 0.72,
    },
    "spontaneous": {
        "id": "cmu60wldk00jcnq07eaxdrdl5",
        "name": "Common Voice Spontaneous Speech 5.0 - Puno Quechua",
        "approx_gb": 0.73,
    },
}


def get_download_url(dataset_id, api_key):
    last_error = None
    for host in API_HOSTS:
        try:
            resp = requests.post(
                f"{host}/{dataset_id}/download",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                timeout=60,
            )
            resp.raise_for_status()
            return resp.json()["downloadUrl"]
        except Exception as e:  # noqa: BLE001
            last_error = f"{host}: {e}"
    raise RuntimeError(f"Could not get a download URL. Last error: {last_error}")


def download_file(url, dest):
    """Resumable download — a dropped connection does not cost the whole file."""
    done = dest.stat().st_size if dest.exists() else 0
    headers = {"Range": f"bytes={done}-"} if done else {}
    with requests.get(url, stream=True, headers=headers, timeout=120) as r:
        if r.status_code == 416:  # already complete
            return
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0)) + done
        mode = "ab" if done and r.status_code == 206 else "wb"
        if mode == "wb":
            done = 0
        with open(dest, mode) as f, tqdm(
            total=total, initial=done, unit="B", unit_scale=True, desc=dest.name
        ) as bar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                bar.update(len(chunk))


def extract(archive: Path, dest_dir: Path):
    dest_dir.mkdir(parents=True, exist_ok=True)
    if tarfile.is_tarfile(archive):
        with tarfile.open(archive) as tar:
            tar.extractall(dest_dir, filter="data")
    elif zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest_dir)
    else:
        raise RuntimeError(f"{archive} is neither a tar nor a zip archive.")


def already_extracted(dest_dir: Path) -> bool:
    """A dataset counts as present once it holds a TSV and some audio."""
    if not dest_dir.exists():
        return False
    has_tsv = any(dest_dir.rglob("*.tsv"))
    has_audio = any(dest_dir.rglob("*.mp3")) or any(dest_dir.rglob("*.wav"))
    return has_tsv and has_audio


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default=str(ROOT / "data"))
    p.add_argument("--only", choices=sorted(DATASETS), help="Download just one corpus")
    p.add_argument("--keep-archive", action="store_true")
    args = p.parse_args()

    api_key = os.getenv("MDC_API_KEY")
    if not api_key:
        sys.exit(
            "MDC_API_KEY is not set.\n"
            "  cp shared-task/.env.example .env   # then paste your key\n"
            "  Key: https://mozilladatacollective.com -> Profile -> API"
        )

    out_root = Path(args.output)
    targets = {args.only: DATASETS[args.only]} if args.only else DATASETS

    for key, ds in targets.items():
        dest_dir = out_root / key
        if already_extracted(dest_dir):
            print(f"[skip] {key}: already present in {dest_dir}")
            continue

        print(f"\n[{key}] {ds['name']}  (~{ds['approx_gb']:.2f} GB)")
        archive = out_root / f"{key}.archive"
        out_root.mkdir(parents=True, exist_ok=True)

        url = get_download_url(ds["id"], api_key)
        download_file(url, archive)

        print("  extracting...")
        extract(archive, dest_dir)
        if not args.keep_archive:
            archive.unlink()
        print(f"  done -> {dest_dir}")

    print("\nAll set. Next: python shared-task/01_build_manifests.py")


if __name__ == "__main__":
    main()
