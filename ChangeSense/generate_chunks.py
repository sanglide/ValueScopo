#!/usr/bin/env python3
"""
generate_chunks.py — 支持本地仓库与远程 public 仓库（owner/repo）浅克隆并生成 chunks.json
python generate_chunks.py --remote owner/repo --out pr_chunks.json --compare-base HEAD~1 --depth 2

python ChangeSense/generate_chunks.py --remote tensorflow/tensorflow --out results/pr_chunks.json --compare-base HEAD~1 --depth 2

用法（两种）：
1) 在 CI 中对当前仓库运行（默认行为）：
   python generate_chunks.py

2) 在 main() 中指定远程仓库（owner/repo），脚本会浅克隆该仓库并生成 chunks.json：
   修改 main() 中 REMOTE_REPO 变量，或通过命令行参数 --remote owner/repo

输出：
  - 默认写入 ./chunks.json （可通过 --out 指定路径）
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from typing import List, Dict, Optional


def run(cmd: List[str], cwd: Optional[str] = None) -> str:
    """Run command and return stdout (raise on failure)."""
    return subprocess.check_output(cmd, universal_newlines=True, cwd=cwd).strip()


def safe_run(cmd: List[str], cwd: Optional[str] = None) -> str:
    """Run command, return stdout or empty on error (no raise)."""
    try:
        return run(cmd, cwd=cwd)
    except subprocess.CalledProcessError:
        return ""


def clone_remote_repo(owner_repo: str, dest: str, depth: int = 2) -> bool:
    """
    Shallow-clone a public GitHub repo to dest.
    Returns True on success, False otherwise.
    """
    url = f"https://github.com/{owner_repo}.git"
    try:
        run(["git", "clone", "--depth", str(depth), url, dest])
        return True
    except Exception as e:
        # try fallback with depth=1
        try:
            if depth != 1:
                run(["git", "clone", "--depth", "1", url, dest])
                return True
        except Exception:
            return False
    return False


def get_changed_files(base: str = "HEAD~1", repo_dir: Optional[str] = None) -> List[str]:
    """
    Get changed files between base and HEAD.
    If base resolves (e.g., HEAD~1 exists in shallow clone), returns names; otherwise returns all files.
    """
    # Try standard compare
    diff_out = safe_run(["git", "diff", "--name-only", f"{base}...HEAD"], cwd=repo_dir)
    if diff_out:
        return [l for l in diff_out.splitlines() if l.strip()]
    # Fallback: if diff failed (e.g., no HEAD~1), list all tracked files
    ls_out = safe_run(["git", "ls-tree", "-r", "--name-only", "HEAD"], cwd=repo_dir)
    if ls_out:
        return [l for l in ls_out.splitlines() if l.strip()]
    return []


def extract_hunks(file_path: str, repo_dir: Optional[str], base: str = "HEAD~1") -> List[Dict]:
    """
    Extract hunks (unified diff) for a file and produce chunk-like objects.
    Uses 'git diff -U3 base...HEAD -- file_path' to get hunks.
    """
    try:
        diff = safe_run(["git", "diff", "-U3", f"{base}...HEAD", "--", file_path], cwd=repo_dir)
        if not diff:
            # no diff for this file (maybe unchanged in diff), return empty
            return []
    except Exception:
        return []

    hunks = []
    current = None
    lines = diff.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("@@"):
            if current:
                hunks.append(current)
            current = {"hunk_header": line, "body": []}
        elif current is not None:
            current["body"].append(line)
    if current:
        hunks.append(current)

    chunks = []
    for idx, h in enumerate(hunks):
        diff_hunk = h["hunk_header"] + "\n" + "\n".join(h["body"])
        # try extract +start line
        m = re.search(r"\+([0-9]+)(?:,([0-9]+))?", h["hunk_header"])
        if m:
            start_line = int(m.group(1))
        else:
            start_line = None

        surrounding = []
        # Try to read the file content from repo_dir to provide surrounding lines
        try:
            abs_path = os.path.join(repo_dir, file_path) if repo_dir else file_path
            with open(abs_path, "r", encoding="utf-8") as fh:
                file_lines = fh.readlines()
            if start_line is None:
                start_line = 1
            s = max(1, start_line - 8)
            e = min(len(file_lines), start_line + 8)
            for ln in range(s, e + 1):
                surrounding.append({"line": ln, "text": file_lines[ln - 1].rstrip("\n")})
        except Exception:
            # If we cannot open file (binary or missing), leave surrounding empty
            surrounding = []

        chunks.append(
            {
                "chunk_id": f"{file_path}|hunk{idx}",
                "file_path": file_path,
                "language": os.path.splitext(file_path)[1].lstrip("."),
                "diff_hunk": diff_hunk,
                "surrounding_lines": surrounding,
                "static_hints": [],
            }
        )

    return chunks


def generate_chunks_from_repo(repo_dir: str, base: str = "HEAD~1", repo_name: Optional[str] = None) -> List[Dict]:
    """
    Produce chunks for all changed files in the given repo_dir.
    repo_name is included in chunk metadata when present.
    """
    files = get_changed_files(base=base, repo_dir=repo_dir)
    all_chunks = []
    for f in files:
        hunks = extract_hunks(f, repo_dir=repo_dir, base=base)
        for h in hunks:
            # enrich chunk with repo/pr info if possible
            h["repo"] = repo_name or os.getenv("GITHUB_REPOSITORY", "local")
            h["pr_id"] = os.getenv("PR_NUMBER", "unknown")
            # ensure chunk_id fully qualified with repo
            if "chunk_id" in h:
                h["chunk_id"] = f"{h.get('repo')}|{h['chunk_id']}"
            all_chunks.append(h)
    return all_chunks


def write_chunks(chunks: List[Dict], out_path: str = "chunks.json") -> None:
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(chunks, fh, indent=2, ensure_ascii=False)
    print(f"Wrote {len(chunks)} chunks to {out_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--remote", help="Remote public repository in owner/repo format to clone (optional)", default=None)
    p.add_argument("--compare-base", help="Compare base (git ref) for diff, default HEAD~1", default="HEAD~1")
    p.add_argument("--out", help="Output chunks.json path", default="chunks.json")
    p.add_argument("--depth", type=int, help="Shallow clone depth (default 2)", default=2)
    return p.parse_args()


def main():
    """
    Main entry.
    To use remote public repo, set REMOTE_REPO variable below or pass --remote owner/repo via CLI.
    """
    # Option A: hardcode REMOTE_REPO here (as user requested), e.g.:
    # REMOTE_REPO = "octocat/Hello-World"
    # REMOTE_REPO = None  # use None to operate on local repo
    REMOTE_REPO = None  # <-- 在这里把远程仓库名填入 "owner/repo"，或使用 CLI --remote

    args = parse_args()
    # CLI overrides hardcoded variable
    remote = args.remote if args.remote else REMOTE_REPO
    compare_base = args.compare_base
    out_path = args.out
    depth = args.depth

    temp_dir = None
    try:
        if remote:
            # clone remote public repo into temp dir
            temp_dir = tempfile.mkdtemp(prefix="valueguard_repo_")
            print(f"Cloning https://github.com/{remote}.git into {temp_dir} (depth={depth}) ...")
            ok = clone_remote_repo(remote, temp_dir, depth=depth)
            if not ok:
                print("Shallow clone failed, trying fallback depth=1 ...")
                ok = clone_remote_repo(remote, temp_dir, depth=1)
                if not ok:
                    raise RuntimeError(f"Failed to clone remote repo: {remote}")
            # If depth=2 clone was successful, compare_base HEAD~1 should work. If only depth=1, HEAD~1 may not exist.
            # We'll attempt to use provided compare_base; if it fails, get_changed_files will fall back to listing files.
            repo_dir = temp_dir
            repo_name = remote
            print("Generating chunks from remote repo...")
            chunks = generate_chunks_from_repo(repo_dir=repo_dir, base=compare_base, repo_name=repo_name)
        else:
            # operate on local repo (current working directory)
            print("No remote specified — operating on local repository (current working dir).")
            cwd = os.getcwd()
            chunks = generate_chunks_from_repo(repo_dir=cwd, base=compare_base, repo_name=None)

        # Write out chunks.json
        write_chunks(chunks, out_path=out_path)

    finally:
        # cleanup temp_dir if we cloned
        if temp_dir and os.path.isdir(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except Exception:
                pass


if __name__ == "__main__":
    main()
