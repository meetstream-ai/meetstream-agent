# Project Setup (with uv)

This project uses [Astral’s uv](https://github.com/astral-sh/uv) for fast, reproducible Python environments. You don’t need to activate `.venv` manually; uv handles it.

## Prerequisites

- **Python:** Version specified in `pyproject.toml` or `.python-version` (if present)
- **uv installed**

### Install uv

- **macOS/Linux:**  
    ```sh
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
- **Windows (PowerShell):**  
    ```powershell
    iwr https://astral.sh/uv/install.ps1 -UseBasicParsing | iex
    ```
- **Verify:**  
    ```sh
    uv --version
    ```

Make sure `pyproject.toml` and `uv.lock` are committed. The lockfile guarantees identical dependencies.

## Quickstart

```sh
uv sync
```
- Creates or refreshes `.venv/` in the project directory
- Installs exactly what’s in `uv.lock`
- Honors Python version declared in the project

To have uv also install the required Python:
```sh
uv python install
uv sync
```
## Node.js Setup (if needed)

If your project also requires Node.js (for frontend or tooling), install it using [nvm](https://github.com/nvm-sh/nvm):

```sh
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
# restart your shell, then:
nvm install --lts
```
## Running the App

You can run without activating the virtualenv:
```sh
uv run python server.py
```

## Common Tasks

- **Install/Update dependencies (respecting the lockfile):**
    ```sh
    uv sync
    ```
- **Add a new dependency (updates lockfile):**
    ```sh
    uv add <package>
    # or for dev-only:
    uv add --dev <package>
    ```
- **Rebuild/refresh the lock (e.g., after pyproject edits):**
    ```sh
    uv lock --refresh
    # or to upgrade everything within constraints:
    uv lock --upgrade
    ```
- **Use a specific Python version in the venv:**
    ```sh
    uv venv --python 3.12
    uv sync
    ```
- **Run any module/script via uv:**
    ```sh
    uv run python -m pytest
    uv run ruff check .
    uv run python scripts/something.py
    ```

## Environment Variables

If the app needs configuration, create `.env` (or use your own config strategy).

**Example:**
```env
# .env.example
PORT=8000
DEBUG=false
```
Document required variables in this README and provide a `.env.example`.

## Repo Hygiene

Add these (if not already):

```gitignore
# .gitignore
.venv/
.pytest_cache/
__pycache__/
*.pyc
.env
```

## Troubleshooting

- **Different OS/architecture wheels:**  
    If you move across OS/CPU types and encounter resolution issues, regenerate the lock on the target system:
    ```sh
    uv lock --refresh
    uv sync
    ```

- **Python version mismatch:**  
    If `uv sync` says the Python version doesn’t match, install the right one:
    ```sh
    uv python install <version>
    uv sync
    ```

- **“Works locally but not here”:**  
    Clear cache, then resync:
    ```sh
    uv cache clean
    uv sync --frozen  # ensure you're using only what's in uv.lock
    ```

- **Inspect dependency tree:**
    ```sh
    uv tree
    ```

## TL;DR for your exact flow

On a new machine:

```sh
# 1) install uv (see above)
# 2) in the project folder:
uv python install              # optional, installs the required Python
uv sync                        # creates .venv and installs from uv.lock
uv run python server.py        # run the app
```