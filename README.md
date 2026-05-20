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
