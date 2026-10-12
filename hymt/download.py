"""Resumable model downloader.

Weights are fetched with `curl -C -` straight into `models/<name>/`, one file at
a time, because a multi-gigabyte download over a slow link needs resume above
all else — `huggingface_hub` abandons the whole snapshot when a connection dies,
and it stalls outright on some networks.

The same MLX conversions are published on HuggingFace and mirrored on
hf-mirror.com and ModelScope. Throughput between them varies by an order of
magnitude depending on time and location, so `fetch` measures all three and uses
whichever is fastest right now. Pin one with HF_ENDPOINT or `--endpoint`.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

WEIGHT_PATTERNS = (".safetensors", ".json", ".jinja", ".model", ".txt")
SKIP_PREFIXES = ("imgs/", ".git")

HF = "https://huggingface.co"
HF_MIRROR = "https://hf-mirror.com"
MODELSCOPE = "https://www.modelscope.cn"
ENDPOINTS = (HF, HF_MIRROR, MODELSCOPE)


@dataclass
class RemoteFile:
    name: str
    size: int


def endpoint() -> str:
    return os.environ.get("HF_ENDPOINT", HF).rstrip("/")


def file_url(base: str, repo: str, name: str) -> str:
    """ModelScope serves blobs through its own API; the others are HF-compatible."""
    if base.rstrip("/") == MODELSCOPE:
        return f"{MODELSCOPE}/api/v1/models/{repo}/repo?Revision=master&FilePath={name}"
    return f"{base.rstrip('/')}/{repo}/resolve/main/{name}"


def _get_json(url: str, timeout: int = 30) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "hy-ebook-translator"})
    with urllib.request.urlopen(req, timeout=timeout) as fh:
        return json.load(fh)


def _keep(name: str) -> bool:
    return not name.startswith(SKIP_PREFIXES) and name.endswith(WEIGHT_PATTERNS)


def _list_from(base: str, repo: str) -> list[RemoteFile]:
    base = base.rstrip("/")
    if base == MODELSCOPE:
        data = _get_json(f"{MODELSCOPE}/api/v1/models/{repo}/repo/files?Revision=master")
        files = (data.get("Data") or {}).get("Files") or []
        return [RemoteFile(f["Path"], int(f.get("Size") or 0))
                for f in files if _keep(f["Path"])]
    data = _get_json(f"{base}/api/models/{repo}?blobs=true")
    return [RemoteFile(s["rfilename"], int(s.get("size") or 0))
            for s in data.get("siblings", []) if _keep(s["rfilename"])]


def list_files(repo: str) -> list[RemoteFile]:
    """The weight manifest, from whichever host answers first."""
    errors: list[str] = []
    seen: set[str] = set()
    for base in (endpoint(), HF, HF_MIRROR, MODELSCOPE):
        if base in seen:
            continue
        seen.add(base)
        try:
            out = _list_from(base, repo)
            if out:
                return out
        except (urllib.error.URLError, OSError, ValueError, KeyError) as exc:
            errors.append(f"{base}: {exc}")
    raise RuntimeError(f"could not list files for {repo} ({'; '.join(errors)})")


def probe_endpoints(repo: str, sample: str, seconds: int = 8) -> list[tuple[str, float]]:
    """Measure each host's throughput on a real weight file."""
    results: list[tuple[str, float]] = []
    for base in ENDPOINTS:
        try:
            out = subprocess.run(
                # No Range header: some mirrors refuse ranged GETs, which would
                # make a ranged probe reject an otherwise fast endpoint.
                ["curl", "-sL", "-o", os.devnull, "--max-time", str(seconds),
                 "-w", "%{speed_download}", file_url(base, repo, sample)],
                capture_output=True, text=True, timeout=seconds + 15,
            )
            results.append((base, float(out.stdout.strip() or 0)))
        except (subprocess.SubprocessError, ValueError):
            results.append((base, 0.0))
    results.sort(key=lambda r: -r[1])
    return results


def pick_endpoint(repo: str, files: list[RemoteFile], quiet: bool = False) -> None:
    """Choose the fastest host unless HF_ENDPOINT is already pinned."""
    if os.environ.get("HF_ENDPOINT"):
        if not quiet:
            print(f"→ using pinned endpoint {endpoint()}")
        return
    sample = max(files, key=lambda f: f.size).name
    if not quiet:
        print("→ measuring download endpoints …")
    ranked = probe_endpoints(repo, sample)
    for base, speed in ranked:
        if not quiet:
            print(f"    {base:30s} {speed / 1e3:7.0f} KB/s")
    best, speed = ranked[0]
    if speed <= 0:
        return
    os.environ["HF_ENDPOINT"] = best
    if not quiet:
        total = sum(f.size for f in files)
        print(f"  using {best} — roughly {total / speed / 60:.0f} min at this rate")


def local_dir(root: Path, repo: str) -> Path:
    return root / "models" / repo.split("/")[-1]


def is_complete(root: Path, repo: str) -> bool:
    d = local_dir(root, repo)
    if not (d / "config.json").exists():
        return False
    index = d / "model.safetensors.index.json"
    if index.exists():
        try:
            shards = set(json.loads(index.read_text())["weight_map"].values())
        except (OSError, ValueError, KeyError):
            return False
        if not shards or any(not (d / name).exists() for name in shards):
            return False
    elif not any(d.glob("*.safetensors")):
        return False
    try:
        want = {f.name: f.size for f in list_files(repo)}
    except Exception:
        return False  # A partial shard cannot be trusted while offline.
    for name, size in want.items():
        path = d / name
        if not path.exists() or (size and path.stat().st_size != size):
            return False
    return True


def missing_bytes(root: Path, repo: str) -> int:
    d = local_dir(root, repo)
    total = 0
    for f in list_files(repo):
        have = (d / f.name).stat().st_size if (d / f.name).exists() else 0
        total += max(f.size - have, 0)
    return total


def fetch(root: Path, repo: str, quiet: bool = False) -> Path:
    """Download every weight file for `repo`, resuming partial files."""
    if not shutil.which("curl"):
        raise RuntimeError("curl is required to download weights")
    dest = local_dir(root, repo)
    dest.mkdir(parents=True, exist_ok=True)
    files = list_files(repo)
    pick_endpoint(repo, files, quiet=quiet)
    total = sum(f.size for f in files)
    if not quiet:
        print(f"→ {repo}  ({total / 1e9:.2f} GB in {len(files)} files) → {dest}")

    for idx, f in enumerate(files, 1):
        target = dest / f.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and f.size and target.stat().st_size == f.size:
            if not quiet:
                print(f"  [{idx}/{len(files)}] {f.name} — already complete")
            continue
        if not quiet:
            have = target.stat().st_size if target.exists() else 0
            suffix = f", resuming at {have / 1e6:.0f} MB" if have else ""
            print(f"  [{idx}/{len(files)}] {f.name}  {f.size / 1e6:.0f} MB{suffix}")
        _curl(file_url(endpoint(), repo, f.name), target, f, quiet)
    if not quiet:
        print(f"✓ weights ready in {dest}")
    return dest


def _curl(url: str, target: Path, f: RemoteFile, quiet: bool) -> None:
    base = [
        "curl", "-L", "--fail",
        "--retry", "20", "--retry-delay", "3", "--retry-all-errors",
        "--connect-timeout", "30",
        # Abandon a connection that goes quiet so a retry can pick a new one.
        "--speed-time", "60", "--speed-limit", "2048",
        "-o", str(target), url,
    ]
    base.insert(1, "-s" if quiet else "--progress-bar")
    err = subprocess.DEVNULL if quiet else None
    code = subprocess.call(base[:2] + ["-C", "-"] + base[2:], stderr=err)
    if code == 33:
        # The host refused a range request; start this file over without resume.
        target.unlink(missing_ok=True)
        code = subprocess.call(base, stderr=err)
    have = target.stat().st_size if target.exists() else 0
    if code != 0:
        raise RuntimeError(
            f"download of {f.name} stopped (curl exit {code}) at {have / 1e6:.0f} MB. "
            "Re-run `./hy setup` to resume."
        )
    if f.size and have != f.size:
        raise RuntimeError(
            f"{f.name} is {have} bytes, expected {f.size}. Re-run `./hy setup` to resume."
        )


def resolve(root: Path, repo: str) -> str:
    """What to hand `mlx_lm.load`: a local directory if present, else the repo id."""
    d = local_dir(root, repo)
    if (d / "config.json").exists() and any(d.glob("*.safetensors")):
        return str(d)
    return repo


def main(argv: list[str]) -> int:
    root = Path(__file__).resolve().parent.parent
    repo = argv[0] if argv else "mlx-community/Hy-MT2-7B-8bit"
    fetch(root, repo)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
