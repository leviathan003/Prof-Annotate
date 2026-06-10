"""
profannotate/core/git/reader.py
Read-only git log integration. Prof Annotate never commits.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class AnnotationCommit:
    author: str
    email: str
    timestamp: str
    commit_hash: str
    message: str


def is_git_repo(path: str | Path) -> bool:
    try:
        import git

        git.Repo(str(path), search_parent_directories=True)
        return True
    except Exception:
        return False


def get_last_annotation_commit(
    label_path: str | Path,
    repo_root: Optional[str | Path] = None,
) -> Optional[AnnotationCommit]:
    try:
        import git

        search = str(repo_root) if repo_root else str(Path(label_path).parent)
        repo = git.Repo(search, search_parent_directories=True)
        commits = list(repo.iter_commits(paths=str(label_path), max_count=1))
        if not commits:
            return None
        c = commits[0]
        return AnnotationCommit(
            author=c.author.name or "Unknown",
            email=c.author.email or "",
            timestamp=c.authored_datetime.isoformat(),
            commit_hash=c.hexsha[:8],
            message=(c.message or "").strip()[:80],
        )
    except Exception as exc:
        logger.debug("Git read failed for %s: %s", label_path, exc)
        return None


def find_repo_root(path: str | Path) -> Optional[Path]:
    try:
        import git

        repo = git.Repo(str(path), search_parent_directories=True)
        return Path(repo.working_tree_dir)
    except Exception:
        return None
