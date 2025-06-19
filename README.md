# GitHub Workflows Configurator

## Description  
Batch-sync or delete GitHub Actions workflow files and clean up stale code‑scanning analyses. Ideal for CI/CD maintainers who need to propagate or prune workflow configurations across repositories.

## Features  
- Sync workflow YAMLs from a source repo to a target repo  
- Delete workflow YAMLs and disable remote workflows via GitHub API  
- Clean up stale code‑scanning analyses  
- Support for selecting all, internal‑only, custom lists or predefined groups  
- Exponential‑backoff retries on API rate limits and server errors  
- Verbose logging and error handling  

## Installation  

1. Clone this repository or download `github_workflows_configurator.py`.  
2. Ensure Python 3.7+ is installed.  
3. Install dependencies:

   ```bash
   pip install requests
   ```

## GitHub Personal Access Tokens (PATs)

This script relies on GitHub PATs to simplfy automation. To create your GitHub personal access token follow these instructions: https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens

To configure GitHub Personal Access Tokens (PATs) for seamless use with Git on different platforms, use the following command-line instructions. These configure Git to cache or store your PAT securely using the appropriate credential helper for your OS.

---

## macOS

Use the native Keychain to store your Git credentials securely:

```sh
git config --global credential.helper osxkeychain
```

**Note:** Git for macOS should already have the credential helper installed. If not:

```sh
git credential-osxkeychain
```

This helper caches your PAT in the macOS Keychain after your first use (e.g., `git push`).

---

## Windows

Use the Windows Credential Manager:

```sh
git config --global credential.helper manager
```

Or for the older version:

```sh
git config --global credential.helper manager-core
```

The token is saved securely in the Windows Credential Manager after first use.

---

## Linux

Use the Git credential cache or store method:

### Option 1: Cache in memory (temporary)

```sh
git config --global credential.helper 'cache --timeout=3600'
```

This caches your credentials for 1 hour (3600 seconds) in memory.

### Option 2: Store in plaintext (insecure)

```sh
git config --global credential.helper store
```

This writes the PAT in `~/.git-credentials`. Use only in secured environments.

---

## Test Configuration

To verify the credential helper:

```sh
git config --show-origin --get credential.helper
```

To manually inspect saved credentials:

* **macOS:** `Keychain Access`
* **Windows:** `Credential Manager`
* **Linux:** `~/.git-credentials` or memory (no file for cache)

---

## Usage  

```bash
python3 github_workflows_configurator.py [OPTIONS] owner/repo
```

**Options**  
- `--branch BRANCH`  
- `--mode {all,internal,custom,group}`  
- `--files FILE1,FILE2`  
- `--group GROUP_NAME`  
- `--delete` (delete mode)  
- `--verbose` (debug logging)  

**Examples**

- Sync all workflows to `acme/myrepo` on branch `main`:

  ```bash
  python3 github_workflows_configurator.py acme/myrepo
  ```

- Delete internal workflows in `acme/myrepo` and disable remote:

  ```bash
  python3 github_workflows_configurator.py --mode internal --delete acme/myrepo
  ```

- Sync custom list on branch `develop`:

  ```bash
  python3 github_workflows_configurator.py     --branch develop     --mode custom     --files bandit.yml,scorecard.yml     acme/myrepo
  ```

## Real‑World Example  

_Source repo:_ `https://github.com/r/test`  
_Target repo:_ `acme/myrepo`  

```bash
# Sync Python group workflows
python3 github_workflows_configurator.py   --mode group   --group python   acme/myrepo
```

_Result:_  
- Workflow YAMLs copied into `acme/myrepo/.github/workflows`  
- Changes committed and pushed to `main`  

## Testing  

_No automated tests provided._  
To validate behavior, run the script in `--verbose` mode against a test repository and inspect logs.

## Contributing  

Contributions welcome via pull requests.  
1. Fork the repo  
2. Create a feature branch  
3. Submit a pull request with tests and documentation updates  

## License  


