"""Resume large Hugging Face weights through small concurrent HTTP ranges.

Use this when a single multi-GB curl or Xet transfer stalls on an unstable link.
Completed ranges are kept in models/<model>/.parts until SHA-256 verification
succeeds, so restarting this command does not repeat finished work.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

CHUNK = 2 * 1024 * 1024
ROOT = Path(__file__).resolve().parent.parent


def manifest(repo: str) -> list[dict]:
    url = f"https://huggingface.co/api/models/{repo}?blobs=true"
    with urllib.request.urlopen(url, timeout=30) as response:
        data = json.load(response)
    return [s for s in data["siblings"] if s["rfilename"].endswith(".safetensors")]


def fetch_range(repo: str, name: str, size: int, part: Path,
                start: int, end: int) -> None:
    expected = end - start + 1
    if part.exists() and part.stat().st_size == expected:
        return
    part.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/{repo}/resolve/main/{name}"
    for attempt in range(30):
        have = part.stat().st_size if part.exists() else 0
        if have == expected:
            return
        if have > expected:
            part.unlink()
            have = 0
        segment = part.with_suffix(".transfer")
        result = subprocess.run(
            ["curl", "-sSL", "--fail", "--max-time", "90",
             "--connect-timeout", "20", "--range", f"{start + have}-{end}",
             "-o", str(segment), "-w", "%{http_code}", url],
            capture_output=True, text=True,
        )
        code = result.stdout.strip()
        if code == "206" and segment.exists():
            received = segment.stat().st_size
            if 0 < received <= expected - have:
                with part.open("ab") as out, segment.open("rb") as source:
                    while block := source.read(1024 * 1024):
                        out.write(block)
        segment.unlink(missing_ok=True)
        if result.returncode and not part.exists():
            time.sleep(min(attempt + 1, 10))
    raise RuntimeError(f"range {name} {start}-{end} stopped at {part.stat().st_size if part.exists() else 0}/{expected}")


def fetch_file(repo: str, item: dict, dest: Path, workers: int) -> None:
    name = item["rfilename"]
    size = int(item["size"])
    sha = item["lfs"]["sha256"]
    target = dest / name
    if target.exists() and target.stat().st_size == size:
        if hash_file(target) == sha:
            print(f"✓ {name} already verified", flush=True)
            return
    parts = dest / ".parts" / name
    ranges = [(i, min(i + CHUNK, size) - 1) for i in range(0, size, CHUNK)]
    completed = 0
    print(f"→ {name}: {size / 1e9:.2f} GB in {len(ranges)} resumable ranges", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(fetch_range, repo, name, size, parts / f"{i // CHUNK:05d}.part", i, end): i
            for i, end in ranges
        }
        for future in as_completed(futures):
            future.result()
            completed += 1
            if completed % 100 == 0 or completed == len(ranges):
                print(f"  {name}: {completed}/{len(ranges)} ranges", flush=True)
    digest = hashlib.sha256()
    temporary = target.with_suffix(target.suffix + ".verified")
    with temporary.open("wb") as out:
        for i, _ in ranges:
            with (parts / f"{i // CHUNK:05d}.part").open("rb") as source:
                while block := source.read(1024 * 1024):
                    digest.update(block)
                    out.write(block)
    if digest.hexdigest() != sha or temporary.stat().st_size != size:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"SHA-256 or size mismatch for {name}")
    temporary.replace(target)
    for part in parts.glob("*.part"):
        part.unlink()
    parts.rmdir()
    print(f"✓ {name} SHA-256 verified", flush=True)


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="mlx-community/Hy-MT2-7B-8bit")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 64:
        parser.error("--workers must be 1–64")
    dest = ROOT / "models" / args.repo.split("/")[-1]
    dest.mkdir(parents=True, exist_ok=True)
    for item in manifest(args.repo):
        fetch_file(args.repo, item, dest, args.workers)
    print("✓ all weight shards verified", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
