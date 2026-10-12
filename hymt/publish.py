"""Git publishing to GitHub Pages.

The repository doubles as the website: Pages is served from the repository root,
`outputs/*.md` are the documents, and `index.html` / `reader.html` render them.
That keeps exactly one copy of every translation.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from .config import Config

# Everything the website and the pipeline itself need; `inputs/` is added only
# when [publish].include_inputs is on.
TRACKED = [
    "index.html", "reader.html", "library.json", ".nojekyll", ".gitignore",
    "README.md", "config.toml", "pyproject.toml", "requirements.txt", "hy",
    "assets", "hymt", "outputs", "tests",
]


@dataclass
class PublishResult:
    committed: bool = False
    pushed: bool = False
    commit: str = ""
    branch: str = ""
    remote: str = ""
    pages_url: str = ""
    changed: list[str] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)


def git(cfg: Config, *args: str, check: bool = True, quiet: bool = True) -> str:
    out = subprocess.run(
        ["git", "-C", str(cfg.root), *args],
        capture_output=True, text=True,
    )
    if check and out.returncode != 0:
        detail = (out.stderr or out.stdout).strip()
        raise RuntimeError(f"git {' '.join(args)} failed: {detail}")
    return out.stdout.strip()


def remote_url(cfg: Config) -> str:
    repo = cfg.publish.repo
    if cfg.publish.protocol == "https":
        return f"https://github.com/{repo}.git"
    return f"git@github.com:{repo}.git"


def is_repo(cfg: Config) -> bool:
    return (cfg.root / ".git").exists()


def ensure_repo(cfg: Config, result: PublishResult) -> None:
    branch = cfg.publish.branch
    if not is_repo(cfg):
        git(cfg, "init", "-q")
        git(cfg, "checkout", "-q", "-B", branch)
        result.messages.append(f"initialised a git repository on {branch}")
    current = git(cfg, "rev-parse", "--abbrev-ref", "HEAD", check=False) or branch
    if current in {"HEAD", ""}:
        git(cfg, "checkout", "-q", "-B", branch)
        current = branch
    result.branch = current

    want = remote_url(cfg)
    have = git(cfg, "remote", "get-url", "origin", check=False)
    if not have:
        git(cfg, "remote", "add", "origin", want)
        result.messages.append(f"added remote origin → {want}")
    elif have != want:
        git(cfg, "remote", "set-url", "origin", want)
        result.messages.append(f"re-pointed origin → {want}")
    result.remote = want


def pending_changes(cfg: Config) -> list[str]:
    status = git(cfg, "status", "--porcelain", check=False)
    return [line[3:] for line in status.split("\n") if line.strip()]


def stage(cfg: Config) -> list[str]:
    paths = list(TRACKED)
    if cfg.publish.include_inputs:
        paths.append("inputs")
    for rel in paths:
        if (cfg.root / rel).exists():
            git(cfg, "add", "--", rel, check=False)
    staged = git(cfg, "diff", "--cached", "--name-only", check=False)
    return [p for p in staged.split("\n") if p.strip()]


def commit_message(cfg: Config, docs: list[str]) -> str:
    if not docs:
        subject = "Update bilingual library"
    elif len(docs) == 1:
        subject = f"Add bilingual translation of {docs[0]}"
    else:
        subject = f"Add bilingual translations ({len(docs)} documents)"
    body = [
        "",
        f"Translated locally with {cfg.model.repo} via Apple MLX.",
        f"Direction: {cfg.translate.source_lang} → {cfg.translate.target_lang}.",
    ]
    if docs:
        body.append("")
        body.extend(f"- {name}" for name in sorted(docs)[:40])
        if len(docs) > 40:
            body.append(f"- … and {len(docs) - 40} more")
    return subject + "\n" + "\n".join(body) + "\n"


def publish(cfg: Config, docs: list[str], push: bool = True,
            dry_run: bool = False) -> PublishResult:
    result = PublishResult()
    if not shutil.which("git"):
        result.messages.append("git is not installed; nothing published")
        return result

    ensure_repo(cfg, result)
    result.changed = stage(cfg)
    if not result.changed:
        result.messages.append("working tree already matches the published state")
        if push and git(cfg, "rev-parse", "--verify", "HEAD", check=False):
            try:
                git(cfg, "push", "-u", "origin", f"HEAD:{result.branch}")
                result.pushed = True
            except RuntimeError as exc:
                result.messages.append(str(exc))
        return result
    if dry_run:
        result.messages.append(f"dry run: {len(result.changed)} file(s) would be committed")
        return result

    git(cfg, "commit", "-q", "-m", commit_message(cfg, docs))
    result.committed = True
    result.commit = git(cfg, "rev-parse", "--short", "HEAD", check=False)

    if not push:
        result.messages.append("committed locally; push skipped")
        return result

    try:
        git(cfg, "push", "-u", "origin", f"HEAD:{result.branch}")
        result.pushed = True
    except RuntimeError as exc:
        result.messages.append(str(exc))
        result.messages.append(
            "commit is saved locally — fix access and run `./hy publish` again"
        )
        return result

    owner, _, name = cfg.publish.repo.partition("/")
    result.pages_url = f"https://{owner}.github.io/{name}/"
    return result


def enable_pages(cfg: Config) -> tuple[bool, str]:
    """Turn on GitHub Pages for the repository root, via the gh CLI."""
    if not shutil.which("gh"):
        return False, "gh CLI not found; enable Pages in the repository settings"
    repo = cfg.publish.repo
    probe = subprocess.run(
        ["gh", "api", f"repos/{repo}/pages"], capture_output=True, text=True
    )
    if probe.returncode == 0:
        try:
            info = json.loads(probe.stdout)
            return True, f"Pages already enabled at {info.get('html_url', '')}"
        except json.JSONDecodeError:
            return True, "Pages already enabled"
    create = subprocess.run(
        ["gh", "api", "-X", "POST", f"repos/{repo}/pages",
         "-f", f"source[branch]={cfg.publish.branch}", "-f", "source[path]=/"],
        capture_output=True, text=True,
    )
    if create.returncode == 0:
        return True, "Pages enabled (first build takes a minute)"
    detail = (create.stderr or create.stdout).strip().split("\n")[-1][:200]
    return False, f"could not enable Pages automatically: {detail}"
