"""Artifact identity helpers; no inference dependencies."""

from pathlib import Path
import hashlib
import subprocess


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_identity(directory):
    """Unknown values stay null rather than borrowing the caller's repository."""
    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(directory), *args], stderr=subprocess.DEVNULL,
            text=True, timeout=10,
        ).strip()

    try:
        if Path(git("rev-parse", "--show-toplevel")).resolve() != Path(directory).resolve():
            return {"commit": None, "dirty": None}
        commit = git("rev-parse", "HEAD")
        dirty = bool(git("status", "--porcelain", "--untracked-files=normal"))
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": dirty}
