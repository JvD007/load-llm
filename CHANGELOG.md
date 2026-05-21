# Changelog

All notable changes to this project are documented here.

## [Unreleased]

### Fixed
- Chat/generate 404 "Endpoint not found": inference proxy requires the short name after `--` (e.g. `llama-test-app`), not the full display-prefixed name; `ep_bare` is now used for all inference proxy URLs
- Stop/delete 404 on bare endpoint names: user ID is now cached in session state after the endpoint list loads and passed directly to `_gw_qname`, avoiding a redundant `/v1/auth/me` call that could fail silently and leave the name unqualified

## [1.5.0] — 2026-05-21
### Added
- GridWeave SDK upgraded to v0.2.0 — adds `start`, `stop`, `delete`, `endpoint`, `endpoints` as top-level exports; custom image/spec endpoint support; owner-qualified endpoint names

### Changed
- `deploy.sh` and `hooks/pre-push` now copy and reinstall the SDK wheel when it changes, so SDK updates reach the server on push without a full reinstall

### Fixed
- Endpoints deployed by the current user always appeared under "Other Endpoints" when the `get_user_id()` API call failed; all endpoints now fall back to "Your Endpoints" in that case

## [1.4.1] — 2026-05-20
### Changed
- Updated README release notes to include v1.4.0
- Updated CHANGELOG for v1.4.0

## [1.4.0] — 2026-05-20
### Added
- `SECURITY.md` with vulnerability reporting policy
- `CODE_OF_CONDUCT.md` based on Contributor Covenant v2.1
- `CONTRIBUTING.md` with setup, workflow and PR guidelines

## [1.3.1] — 2026-05-20
### Added
- Release notes section to README

## [1.3.0] — 2026-05-20
### Added
- `CONTRIBUTORS.md` listing project authors
- Apache 2.0 `LICENSE`
- `README.md` with setup, deployment and usage instructions
- `hooks/pre-push` git hook for automatic server deployment on push
- `--install-hook` flag in `deploy.sh` to wire up the hook locally
- `.gitignore` covering venv, pycache, Streamlit config and OS files

### Changed
- `deploy.sh` updated to support hook installation

### Fixed
- `WHL_PATH` now resolves relative to `app.py` instead of a hardcoded dev path, fixing HTTP 403 on Deploy
- Removed hardcoded HuggingFace token from default input value

## [1.2.0] — 2026-05-19
### Added
- `deploy.sh` script to automate app update and service restart
- Futuristic Groningen University login screen
- Auto-detect real GPU name from worker via `nvidia-smi`
- Compact chat info bar showing server hostname and GPU info per endpoint
- S3/R2 credentials form with placeholder examples

### Changed
- Renamed "Admin Token" label throughout the UI
- Renamed R2 labels to S3/R2 in credentials form
- Applied Groningen University dark theme across the full app
- Renamed ISC to CIT on login page footer

### Fixed
- Admin token wiped by password widget on every rerun
- Auth state reset causing "Not authenticated" on Stop/Start/Delete actions
- `ValueError` unpacking 9 variables from 8-column endpoint list
- Unauthenticated endpoints list on page load
- Chat output for base models

## [1.1.0] — 2026-05-19
### Added
- Endpoint manager with list, stop, start, delete and connect actions
- Non-blocking deploy and chat interface
- Login screen with token authentication
- GPU column removed from endpoint table

## [1.0.0] — 2026-05-19
### Added
- Initial Streamlit UI for GridWeave LLM deployment
- SDK install on Debian/Ubuntu system Python
