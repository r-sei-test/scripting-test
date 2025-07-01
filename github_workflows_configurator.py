#!/usr/bin/env python3
"""
Batch-sync or delete GitHub Actions workflows and clean up stale configurations.

Operations:
    Sync: copy workflow YAMLs from a source repo to a target repo.
    Delete: remove workflow YAMLs, disable workflows via the Actions API,
    and delete stale analyses

Usage:
    python3 github_workflows_configurator.py [--branch BRANCH]
        [--mode MODE] [--files FILES] [--group NAME] [--delete]
        owner/repo
"""

import argparse  # For parsing command-line arguments
import logging  # For logging progress and errors
import os # For calls to operating system services
import sys  # For system exit
import shutil  # For file operations like removing directories
import subprocess  # For running Git commands
import time  # For retry backoff delays
from pathlib import Path  # For path manipulations
import tempfile  # For creating isolated temp directories
from typing import List, Dict  # For type annotations
import requests  # For interacting with GitHub REST API
import certifi   # Reliable Mozilla CA bundle

# Force Python's ssl module & requests to use certifi's bundle:
os.environ["SSL_CERT_FILE"] = certifi.where()

## Constants ---------------------------------------------------------------
# Defaults and configuration values
DEFAULT_BRANCH = "main"  # Default Git branch to use
SOURCE_REPO = (
    "https://github.com/r-sei-test/test"  # Source repo URL to pull workflows from
)
WORKFLOW_SUBDIR = ".github/workflows"  # Subdirectory where workflows live
GITHUB_API = "https://api.github.com"  # Base URL for GitHub API
if sys.platform.startswith("win"):
    GIT_PATH = "C:\\Program Files\\Git"
else:
    GIT_PATH = "/usr/bin/git"

# Lists of workflow file names
ALL_WORKFLOWS = [
    "bandit.yml",
    "codacy.yml",
    "dependency-review.yml",
    "devskim.yml",
    "osv-scanner.yml",
    "scorecard.yml",
    "semgrep.yml",
    "snyk-security.yml",
    "sonarcloud.yml",
    "trivy.yml",
    "gitleaks.yml",
]
# Workflows requiring external accounts (excluded in 'internal' mode)
EXTERNAL_WORKFLOWS = {"snyk-security.yml", "sonarcloud.yml", "semgrep.yml"}
# Predefined groups of workflows for 'group' mode
PREDEFINED_GROUPS = {
    "python": [
        "bandit.yml",
        "codacy.yml",
        "devskim.yml",
        "semgrep.yml",
        "snyk-security.yml",
        "sonarcloud.yml",
        "dependency-review.yml",
        "osv-scanner.yml",
    ],
    "c": [
        "codacy.yml",
        "devskim.yml",
        "semgrep.yml",
        "snyk-security.yml",
        "sonarcloud.yml",
        "dependency-review.yml",
        "osv-scanner.yml",
    ],
    "c++": [
        "codacy.yml",
        "devskim.yml",
        "semgrep.yml",
        "snyk-security.yml",
        "sonarcloud.yml",
        "dependency-review.yml",
        "osv-scanner.yml",
    ],
    "java": [
        "codacy.yml",
        "devskim.yml",
        "semgrep.yml",
        "snyk-security.yml",
        "sonarcloud.yml",
        "dependency-review.yml",
        "osv-scanner.yml",
    ],
    "misc": [
        "dependency-review.yml",
        "osv-scanner.yml",
        "scorecard.yml",
        "trivy.yml",
        "gitleaks.yml",
    ],
}

## Logging Setup ---------------------------------------------------------------
logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


## Exception Classes ---------------------------------------------------------------
# Custom exceptions for clearer error handling
class WorkflowError(Exception):
    """Base exception for workflow-related errors."""


class GitError(WorkflowError):
    """Raised when a Git command fails."""


class APIError(WorkflowError):
    """Raised when a GitHub API call fails."""


## Git Repository Operations ---------------------------------------------------------------
class GitRepo:
    """
    Encapsulates operations on a local clone of a Git repository.
    """

    def __init__(self, url: str, path: Path, branch: str) -> None:
        # URL of the remote repository
        self.url = url
        # Local filesystem path where the repo will be cloned
        self.path = path
        # Branch to checkout and operate on
        self.branch = branch

    def clone(self) -> None:
        """
        Clone the repository into `self.path`, removing it first if it exists.
        Uses a shallow clone (depth=1) to speed up.
        """
        if self.path.exists():
            if shutil.rmtree.avoids_symlink_attacks is True:
                # Remove existing directory tree to ensure a clean clone
                shutil.rmtree(self.path)
            else:
                raise RuntimeError("No safe shutil.rmtree on this platform.")

        logger.info("Cloning %s@%s -> %s", self.url, self.branch, self.path)
        # Run the Git clone command
        self.run(
            [
                GIT_PATH,
                "clone",
                "--depth",
                "1",
                "--branch",
                self.branch,
                self.url,
                str(self.path),
            ],
            use_cwd=False,  # Don't set cwd during clone
        )

    def checkout(self) -> None:
        """
        Switch the working copy to the configured branch.
        """
        self.run([GIT_PATH, "checkout", self.branch])

    def add(self) -> None:
        """
        Stage changes under the workflows subdirectory.
        """
        self.run([GIT_PATH, "add", WORKFLOW_SUBDIR])

    def commit_and_push(self, message: str) -> None:
        """
        Commit any staged changes with `message` and push to remote.
        Skips commit if there are no staged changes.
        """
        if self.diff_cached_empty():
            print("No staged changes to commit.")
            return
        # Commit the changes
        subprocess.run(
            [GIT_PATH, "commit", "-m", message], cwd=self.path, check=True, shell=False
        )
        # Push to origin/<branch>
        subprocess.run([GIT_PATH, "push"], cwd=self.path, check=True, shell=False)
        logger.info("Pushed %s: %s", self.branch, message)

    def diff_cached_empty(self) -> bool:
        """
        Return True if `git diff --cached --quiet` indicates no staged changes.
        """
        result = subprocess.run(
            [GIT_PATH, "diff", "--cached", "--quiet"],
            cwd=self.path,
            check=False,
            shell=False,
        )
        return result.returncode == 0  # True if no changes, False if there are

    def run(self, cmd: List[str], use_cwd: bool = True) -> None:
        """
        Run a Git command `cmd` in the repo's working directory (unless use_cwd=False).
        """
        try:
            subprocess.run(
                cmd, cwd=self.path if use_cwd else None, check=True, shell=False
            )
        except subprocess.CalledProcessError as e:
            raise GitError(f"Git failed: {' '.join(cmd)}") from e


## Workflow File Manipulation ---------------------------------------------------------------
def select_workflows(mode: str, files_arg: str, group: str) -> List[str]:
    """
    Choose which workflow filenames to process based on `mode`:
      - all: every file in ALL_WORKFLOWS
      - internal: those not requiring external accounts
      - custom: comma-separated list from `files_arg`
      - group: predefined set in PREDEFINED_GROUPS

    Raises WorkflowError on invalid input.
    """
    if mode == "all":
        return ALL_WORKFLOWS
    if mode == "internal":
        # Filter out external workflows
        return [f for f in ALL_WORKFLOWS if f not in EXTERNAL_WORKFLOWS]
    if mode == "custom":
        # Split by comma, strip whitespace
        chosen = [f.strip() for f in files_arg.split(",") if f.strip()]
        # Check for invalid names
        invalid = set(chosen) - set(ALL_WORKFLOWS)
        if invalid:
            raise WorkflowError(f"Invalid filenames: {invalid}")
        return chosen
    if mode == "group":
        # Verify group exists
        if group not in PREDEFINED_GROUPS:
            raise WorkflowError(f"Unknown group: {group}")
        return PREDEFINED_GROUPS[group]
    raise WorkflowError(f"Unsupported mode: {mode}")


def sync_files(src: Path, dst: Path, files: List[str]) -> None:
    """
    Copy specified workflow files from `src` directory to `dst` directory.
    Creates `dst` if it does not exist.

    Skips any file missing in `src` with a warning.
    """
    dst.mkdir(parents=True, exist_ok=True)
    for name in files:
        src_file = src / name
        if not src_file.is_file():
            logger.warning("Source missing: %s", name)
            continue
        # Read the file and write it to the destination
        (dst / name).write_bytes(src_file.read_bytes())
        logger.info("Copied: %s", name)


def delete_and_stage_files(repo: GitRepo, files: List[str]) -> None:
    """
    Remove each workflow file via `git rm -f` so the deletion is
    both on disk and staged for commit.
    """
    for name in files:
        rel = f"{WORKFLOW_SUBDIR}/{name}"
        try:
            repo.run([GIT_PATH, "rm", "-f", rel])
            logger.info("Deleted & staged: %s", name)
        except GitError:
            logger.warning("File not found or git error, skipped: %s", name)


## GitHub API Client with Retry Logic -------------------------------------------------------------
class GitHubAPI:
    """
    GitHub REST API client for workflows and analyses
    with simple retry on transient errors.
    """

    def __init__(self, owner: str, repo: str, token: str) -> None:
        self.base = GITHUB_API
        self.owner = owner
        self.repo = repo
        # Use a session for connection pooling
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github.v3+json",
            }
        )
        self.session.verify = certifi.where()
        self.max_retries = 2  # Number of retry attempts
        self.backoff = 2  # Base for exponential backoff delay

    def _request(
        self, method: str, url: str, retry: bool = False, **kwargs
    ) -> requests.Response:
        """
        Internal helper to perform HTTP requests with retry on 429/5xx.
        Returns a Response or raises on final failure.
        """
        for attempt in range(1, self.max_retries + 1):
            resp = self.session.request(method, url, **kwargs)
            # Retry on rate limit or server errors
            if resp.status_code in (429, 500, 502, 503, 504) or retry:
                wait = self.backoff**attempt
                logger.warning(
                    "HTTP %s on %s; retry #%s in %ss",
                    resp.status_code,
                    url,
                    attempt,
                    wait,
                )
                time.sleep(wait)
                continue
            return resp
        # After retries, raise for status
        resp.raise_for_status()

    def list_workflows(self) -> List[Dict]:
        """
        Get a list of workflows in the repo via Actions API.
        Returns the 'workflows' list from JSON.
        Raises APIError on failure.
        """
        url = f"{self.base}/repos/{self.owner}/{self.repo}/actions/workflows"
        resp = self._request("GET", url)
        if resp.status_code != 200:
            raise APIError(f"Failed to list workflows: {resp.text}")
        return resp.json().get("workflows", [])

    def disable_workflow(self, wf_id: int) -> None:
        """
        Disable a workflow by its numeric ID.
        If it's already disabled, ignore the 403 “not active” error.
        """
        url = f"{self.base}/repos/{self.owner}/{self.repo}/actions/workflows/{wf_id}/disable"
        resp = self._request("PUT", url)
        if resp.status_code in (200, 204):
            return
        # If the workflow is already disabled, GitHub returns 403 with this message
        if resp.status_code == 403:
            payload = resp.json()
            msg = payload.get("message", "")
            if "not active" in msg:
                logger.info("Workflow %s already disabled; skipping.", wf_id)
                return
        raise APIError(f"Failed to disable workflow {wf_id}: {resp.text}")

    def list_analyses(self) -> List[Dict]:
        """
        List analyses for the repo.
        Returns the JSON array of analyses.
        Raises APIError on failure.
        """
        url = f"{self.base}/repos/{self.owner}/{self.repo}/code-scanning/analyses"
        resp = self._request("GET", url)
        if resp.status_code != 200:
            raise APIError(f"Failed to list analyses: {resp.text}")
        return resp.json()

    def delete_analysis(self, analysis_url: str) -> None:
        """
        Delete a single analysis, following GitHub’s confirm_delete_url if provided.
        Handles HTTP 200 or 202 as 'needs confirm'.
        """
        resp = self._request("DELETE", analysis_url)

        # Case 1: Already gone
        if resp.status_code in (204, 404):
            logger.info("Analysis already removed: %s", analysis_url)
            return

        # Case 2: GitHub asks for confirmation (200 or 202)
        if resp.status_code in (200, 202):
            payload = resp.json()
            confirm_url = payload.get("confirm_delete_url")
            if not confirm_url:
                raise APIError(f"No confirm_delete_url in response: {payload}")
            # Append confirm_delete parameter if missing
            if "?confirm_delete" not in confirm_url:
                confirm_url += "?confirm_delete=true"
            resp2 = self._request("DELETE", confirm_url)
            # treat 204 or 404 as success
            if resp2.status_code in (204, 404):
                logger.info(
                    "Confirmed (or already deleted) analysis at %s", confirm_url
                )
                return
            # fallback retry once on transient 200/202
            if resp2.status_code in (200, 202):
                time.sleep(self.backoff)
                resp3 = self._request("DELETE", confirm_url)
                if resp3.status_code in (204, 404):
                    logger.info("Confirmed on retry analysis at %s", confirm_url)
                    return
            raise APIError(f"Confirm-delete failed ({resp2.status_code}): {resp2.text}")

        # Case 3: Something else went wrong
        raise APIError(f"Failed to delete analysis ({resp.status_code}): {resp.text}")


def cleanup_remote_workflows(api: GitHubAPI, deleted: List[str]) -> None:
    """
    Disable any remote GitHub workflows whose filenames match those deleted locally.
    """
    # Build map of filename -> workflow ID
    id_map = {wf["path"].split("/")[-1]: wf["id"] for wf in api.list_workflows()}
    for name in deleted:
        wf_id = id_map.get(name)
        if wf_id:
            api.disable_workflow(wf_id)
            logger.info("Disabled remote workflow: %s", name)
        else:
            logger.debug("No remote workflow entry for %s", name)


def cleanup_analysis(api: GitHubAPI, deleted: List[str]) -> None:
    """
    Remove all deletable analyses related to the deleted workflows.
    Only runs through the current list once to avoid infinite loops.
    """
    # Fetch all analyses once
    all_analyses = api.list_analyses()
    # Filter those matching our deleted workflows
    to_delete = [
        a
        for a in all_analyses
        if a.get("deletable", False)
        and any(
            a["analysis_key"].startswith(f"{WORKFLOW_SUBDIR}/{name}:")
            for name in deleted
        )
    ]
    # Delete each exactly once
    for entry in to_delete:
        try:
            api.delete_analysis(entry["url"])
            logger.info("Deleted analysis: %s", entry["id"])
        except APIError as e:
            logger.warning("Failed to delete analysis %s: %s", entry["id"], e)


## Helper: Credential Helper Check ----------------------------------------------------------------
def check_credential_helper() -> None:
    """
    Warn the user if no Git credential.helper is configured.
    Without it, HTTPS operations may prompt or fail.
    """
    proc = subprocess.run(
        [GIT_PATH, "config", "--get", "credential.helper"],
        check=True,
        capture_output=True,
        text=True,
        shell=False,
    )
    if not proc.stdout.strip():
        logger.warning(
            "No Git credential.helper configured; Git operations may prompt for credentials or fail"
        )


def get_github_token() -> str:
    """Fetch GitHub token securely using Git credential helper."""

    input_data = b"protocol=https\nhost=github.com\n\n"
    proc = subprocess.Popen(
        [GIT_PATH, "credential", "fill"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        shell=False,
    )
    stdout, _ = proc.communicate(input_data)

    for line in stdout.decode().splitlines():
        if line.startswith("password="):
            return line.partition("=")[2].strip()

    raise RuntimeError("No GitHub token found via credential helper.")


## Main Orchestration ---------------------------------------------------------------
def main() -> None:
    """Orchestrate cloning, selection, operation, API cleanup, and Git push."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Sync or delete GitHub Actions workflows in batch."
    )
    parser.add_argument("target", help="owner/repo format, e.g. acme/myrepo")
    parser.add_argument(
        "--branch", default=DEFAULT_BRANCH, help="Git branch to use (default: main)"
    )
    parser.add_argument(
        "--mode",
        choices=["all", "internal", "custom", "group"],
        default="all",
        help="Workflow selection mode",
    )
    parser.add_argument(
        "--files", default="", help="Comma-separated list for custom mode"
    )
    parser.add_argument("--group", default="", help="Group name for group mode")
    parser.add_argument(
        "--delete", action="store_true", help="Delete workflows instead of syncing"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable debug-level logging"
    )
    args = parser.parse_args()

    # Set logging level if verbose
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Validate target format
    if "/" not in args.target:
        parser.error("Target must be in owner/repo format.")

    # Require GitHub token
    token = get_github_token()
    if not token:
        logger.error("Environment variable GITHUB_TOKEN is required.")
        sys.exit(1)

    # Warn about missing credential helper
    check_credential_helper()

    # Extract owner and repo name
    owner, repo = args.target.split("/", 1)

    # Create isolated temp dirs for cloning
    source_path = Path(tempfile.mkdtemp(prefix="src_repo_"))
    target_path = Path(tempfile.mkdtemp(prefix="dst_repo_"))

    # Initialize Git and API clients
    source_repo = GitRepo(SOURCE_REPO, source_path, args.branch)
    target_repo = GitRepo(
        f"https://github.com/{owner}/{repo}.git", target_path, args.branch
    )

    api_client = GitHubAPI(owner, repo, token)

    try:
        # Clone repositories
        source_repo.clone()
        target_repo.clone()

        # Choose which workflows to act on
        workflows = select_workflows(args.mode, args.files, args.group)

        # Paths to workflow directories
        src_dir = source_path / WORKFLOW_SUBDIR
        dst_dir = target_path / WORKFLOW_SUBDIR

        if args.delete:
            # Delete local files
            delete_and_stage_files(target_repo, workflows)
            # Disable remote workflows
            cleanup_remote_workflows(api_client, workflows)
            # Remove stale  analyses
            cleanup_analysis(api_client, workflows)
            # Commit and push deletions
            target_repo.checkout()
            target_repo.commit_and_push("Delete selected workflow YAMLs")
        else:
            # Sync (copy) files
            sync_files(src_dir, dst_dir, workflows)
            # Commit and push additions
            target_repo.checkout()
            target_repo.add()
            target_repo.commit_and_push("Sync selected workflow YAMLs")

    except WorkflowError as e:
        # Log any known errors and exit
        logger.error(e)
        sys.exit(1)

    finally:
        # Remove temp clone directories
        if shutil.rmtree.avoids_symlink_attacks is True:
            # Remove existing directory tree to ensure a clean clone
            shutil.rmtree(source_path, ignore_errors=True)
            shutil.rmtree(target_path, ignore_errors=True)
        else:
            raise RuntimeError("No safe shutil.rmtree on this platform.")
        logger.info("Cleaned up temp directories")


if __name__ == "__main__":
    main()
