# Groningen University — AI Compute Depot

A Streamlit-based web interface for deploying and managing LLM endpoints on the GridWeave platform.

## Features

- Deploy vLLM endpoints with a HuggingFace model and token
- Monitor endpoint status in real time
- Chat with deployed models directly in the browser
- Start / stop / delete endpoints

## Requirements

- Python 3.10+
- A [GridWeave](https://platform.gridweave.io) account and API token
- A [HuggingFace](https://huggingface.co) account and access token

## Installation

Run the installer on a clean Ubuntu server:

```bash
sudo bash install.sh
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--port` | `8501` | Streamlit listen port |
| `--dir` | `/opt/gridweave-depot` | Installation directory |
| `--user` | `gridweave` | Service OS user |

The installer:
1. Installs system dependencies
2. Creates a dedicated service user
3. Sets up a Python virtual environment
4. Registers and starts a `systemd` service

## Deployment

To manually push a code update to the server:

```bash
sudo bash deploy.sh
```

To install the git hook so every `git push origin main` auto-deploys:

```bash
bash deploy.sh --install-hook
```

## Usage

Open the app in your browser (default port 8501) and log in with your GridWeave token.

### Deploy an endpoint

1. Expand **Credentials** and enter your HuggingFace token
2. Set the **Model ID** (e.g. `meta-llama/Llama-3.2-1B-Instruct`)
3. Choose **VRAM** and an **Endpoint Name**
4. Click **Deploy**

### Chat

Once an endpoint is running (🟢), click **Chat** to open the chat interface.

## Service management

```bash
sudo systemctl status  gridweave-depot
sudo systemctl restart gridweave-depot
sudo journalctl -u gridweave-depot -f
```

## Release notes

### v1.3.0 — 2026-05-20
- Added `deploy.sh` with `--install-hook` flag for auto-deploy on push
- Added pre-push git hook (`hooks/pre-push`)
- Fixed HTTP 403 on Deploy caused by hardcoded wheel path
- Removed hardcoded HuggingFace token from UI default
- Added `README`, `LICENSE` (Apache 2.0), `CHANGELOG`, `.gitignore`, and `CONTRIBUTORS`

### v1.2.0 — 2026-05-19
- Groningen University dark theme and futuristic login screen
- Auto-detect GPU name via `nvidia-smi`
- S3/R2 credentials form
- Renamed ISC → CIT in footer
- Fixed auth state reset and ValueError on endpoint list

### v1.1.0 — 2026-05-19
- Endpoint manager (list, stop, start, delete, connect)
- Non-blocking deploy and chat interface
- Login screen with token authentication

### v1.0.0 — 2026-05-19
- Initial Streamlit UI for GridWeave LLM deployment

See [CHANGELOG.md](CHANGELOG.md) for the full history.
