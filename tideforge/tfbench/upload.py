"""B3 upload of a release folder to the private HF dataset repo named by TF_BENCH_REPO.

HF_TOKEN is read by huggingface_hub from the environment; it is never printed, logged or written.
Refuses: a public repo; a release tag that already exists (a rebuild never overwrites a released version); a folder above
max_gb. --dry-run lists the files and the total size and makes no network call. Resumable: huggingface_hub's
upload_large_folder keeps its own progress under <folder>/.cache/huggingface and skips files already uploaded.
After the upload the commit is tagged with the version.
"""
from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger("tfbench.upload")


class Refused(Exception):
    pass


def plan(folder: Path) -> tuple[list[tuple[str, int]], int]:
    files = sorted((p.relative_to(folder).as_posix(), p.stat().st_size) for p in Path(folder).rglob("*")
                   if p.is_file() and ".cache" not in p.relative_to(folder).parts)
    return files, sum(b for _, b in files)


def check_repo(api, repo: str, version: str) -> bool:
    """-> True when the repo must be created. Refused when it is public or the version tag exists."""
    from huggingface_hub.utils import RepositoryNotFoundError
    try:
        info = api.repo_info(repo, repo_type="dataset")
    except RepositoryNotFoundError:
        return True
    if not getattr(info, "private", False):
        raise Refused(f"{repo} is public: tfbench is uploaded only to a private repo")
    tags = {t.name for t in api.list_repo_refs(repo, repo_type="dataset").tags}
    if version in tags:
        raise Refused(f"{repo} already has release {version}: a released version is never overwritten")
    return False


def upload(folder: Path, repo: str, version: str, max_gb: float, dry_run: bool, api=None) -> dict:
    files, total = plan(folder)
    log.info("%d files, %.2f GB in %s", len(files), total / 1e9, folder)
    if total > max_gb * 1e9:
        raise Refused(f"total {total / 1e9:.1f} GB > --max-gb {max_gb:g}")
    if dry_run:
        for p, b in files:
            log.info("  %12d  %s", b, p)
        log.info("dry run: nothing uploaded (%d files, %.2f GB)", len(files), total / 1e9)
        return {"files": len(files), "bytes": total, "uploaded": False}
    if not repo:
        raise Refused("TF_BENCH_REPO is not set")
    if api is None:
        from huggingface_hub import HfApi
        api = HfApi()
    if check_repo(api, repo, version):
        api.create_repo(repo, repo_type="dataset", private=True)
        log.info("created private repo %s", repo)
        check_repo(api, repo, version)
    api.upload_large_folder(repo_id=repo, folder_path=str(folder), repo_type="dataset", private=True)
    api.create_tag(repo, tag=version, repo_type="dataset", tag_message=f"tfbench {version}")
    log.info("uploaded %d files (%.2f GB) to %s, tagged %s", len(files), total / 1e9, repo, version)
    return {"files": len(files), "bytes": total, "uploaded": True}
