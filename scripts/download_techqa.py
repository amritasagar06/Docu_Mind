"""Download the pinned, public TechQA-RAG-Eval source files."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from urllib.error import URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from documind.datasets.techqa import (  # noqa: E402
    DATASET_ID,
    DATASET_LICENSE,
    DATASET_REVISION,
    DATASET_URL,
    sha256_file,
)


RAW_DIR = ROOT / "data" / "raw" / "techqa"
MANIFEST_PATH = ROOT / "data" / "manifests" / "techqa.json"
SOURCE_FILES = ("train.json", "corpus.zip")


def _atomic_json_write(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def download_file(
    filename: str,
    destination: Path,
    *,
    expected_sha256: str | None = None,
    force: bool = False,
) -> dict[str, object]:
    """Download or verify one immutable file, returning its observed checksum."""
    url = f"{DATASET_URL}/resolve/{DATASET_REVISION}/{filename}"
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and not force:
        actual_sha256 = sha256_file(destination)
        if expected_sha256 and actual_sha256 != expected_sha256:
            raise RuntimeError(
                f"Existing {destination} checksum does not match the manifest. "
                "Keep the original raw file or rerun with --force to fetch this pinned revision."
            )
        return {
            "path": destination.relative_to(ROOT).as_posix(),
            "size_bytes": destination.stat().st_size,
            "sha256": actual_sha256,
            "download_url": url,
            "verified_existing_file": True,
        }

    temporary = destination.with_suffix(destination.suffix + ".partial")
    request = Request(url, headers={"User-Agent": "DocuMind-AI/0.1 dataset-acquisition"})
    digest_bytes = 0
    try:
        with urlopen(request, timeout=90) as response, temporary.open("wb") as output:
            if response.status != 200:
                raise RuntimeError(f"Download of {filename} returned HTTP {response.status}")
            while block := response.read(1024 * 1024):
                output.write(block)
                digest_bytes += len(block)
        os.replace(temporary, destination)
    except (OSError, URLError, TimeoutError) as exc:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise RuntimeError(f"Could not download {filename} from the pinned Hugging Face revision: {exc}") from exc

    return {
        "path": destination.relative_to(ROOT).as_posix(),
        "size_bytes": digest_bytes,
        "sha256": sha256_file(destination),
        "download_url": url,
        "verified_existing_file": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-download source files even if local copies already exist",
    )
    args = parser.parse_args()

    existing_manifest: dict[str, object] = {}
    if MANIFEST_PATH.exists():
        existing_manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        old_revision = existing_manifest.get("revision")
        if old_revision not in (None, DATASET_REVISION) and not args.force:
            raise SystemExit(
                f"Manifest is pinned to {old_revision}; use --force to intentionally switch to {DATASET_REVISION}."
            )

    old_files = existing_manifest.get("files", {})
    if not isinstance(old_files, dict):
        old_files = {}
    file_records: dict[str, object] = {}
    for filename in SOURCE_FILES:
        old_record = old_files.get(filename, {})
        expected_hash = old_record.get("sha256") if isinstance(old_record, dict) else None
        result = download_file(
            filename,
            RAW_DIR / filename,
            expected_sha256=expected_hash if isinstance(expected_hash, str) else None,
            force=args.force,
        )
        file_records[filename] = result
        print(f"{filename}: {result['size_bytes']} bytes, sha256={result['sha256']}")

    manifest: dict[str, object] = {
        "dataset_id": DATASET_ID,
        "dataset_url": DATASET_URL,
        "revision": DATASET_REVISION,
        "license": DATASET_LICENSE,
        "retrieved_at_utc": existing_manifest.get("retrieved_at_utc")
        if existing_manifest.get("revision") == DATASET_REVISION
        else datetime.now(timezone.utc).isoformat(),
        "files": file_records,
    }
    if "normalization" in existing_manifest and existing_manifest.get("revision") == DATASET_REVISION:
        manifest["normalization"] = existing_manifest["normalization"]
        if "normalized_files" in existing_manifest:
            manifest["normalized_files"] = existing_manifest["normalized_files"]

    _atomic_json_write(MANIFEST_PATH, manifest)
    print(f"Manifest written to {MANIFEST_PATH.relative_to(ROOT).as_posix()}")
    print(f"Pinned revision: {DATASET_REVISION}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
