import os
import queue
import re
import subprocess
import sys
import threading
import time

import streamlit as st

st.set_page_config(page_title="Groningen University — AI Compute Depot", page_icon="🏛", layout="wide")

WHL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gridweave_sdk-0.2.0-py3-none-any.whl")
WHL_URL  = "https://pub-cbb8992ad1bd437b81d58d5b2da09787.r2.dev/tarball/gridweave_sdk-0.2.0-py3-none-any.whl"

# ── Session state ─────────────────────────────────────────────────────────────
for _k, _v in [
    ("authenticated",  False),
    ("admin_token",    ""),
    ("platform_url",   "https://platform.gridweave.io"),
    ("action_state",   "idle"),
    ("action_label",   ""),
    ("action_log",     []),
    ("action_error",   None),
    ("endpoint",       None),
    ("chat_history",   []),
    ("_result_queue",  None),
    ("gpu_cache",      {}),       # (vendor, vram_gb) -> "RTX 3090" etc.
    ("_gpu_detecting", set()),    # set of (vendor, vram_gb) currently probing
    ("_gpu_queue",     None),
]:
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ── GPU name: cache lookup → inferred fallback ────────────────────────────────
def _gpu_name(vendor: str, vram_gb: int) -> str:
    cached = st.session_state.gpu_cache.get((vendor, vram_gb))
    if cached:
        return cached
    if (vendor, vram_gb) in st.session_state._gpu_detecting:
        return "detecting…"
    # VRAM-based inference as fallback
    v = vendor.upper()
    if v == "NVIDIA":
        if vram_gb >= 78: return "A100 / H100 80 GB"
        if vram_gb >= 46: return "A6000 48 GB"
        if vram_gb >= 38: return "A100 40 GB"
        if vram_gb >= 20: return f"NVIDIA {vram_gb} GB"   # e.g. RTX 5000 / 3090 / 4090 — detection confirms
        if vram_gb >= 15: return "RTX 4080 / A4000 16 GB"
        if vram_gb >= 11: return "RTX 3080 Ti 12 GB"
        return f"NVIDIA {vram_gb} GB"
    if v == "AMD":
        if vram_gb >= 23: return "RX 7900 XTX 24 GB"
        if vram_gb >= 19: return "RX 7900 XT 20 GB"
        if vram_gb >= 15: return "RX 6800 XT 16 GB"
        return f"AMD {vram_gb} GB"
    return f"{vendor} {vram_gb} GB"


# ── Hardware helpers ──────────────────────────────────────────────────────────
def _hw_lookup() -> dict:
    try:
        import gridweave as _gw
        return {
            n["node_id"]: {
                "host":    n.get("host", "—"),
                "vendor":  n.get("vendor", "—"),
                "vram_gb": round(max(n.get("per_gpu_vram_mb", {}).values(), default=0) / 1024),
            }
            for n in _gw.resources()
        }
    except Exception:
        return {}


def _hw(ep_info: dict, lookup: dict) -> tuple[str, str, str, int]:
    """Return (host, gpu_name, vendor, vram_gb)."""
    ray     = ep_info.get("rayservice_name", "")
    node    = lookup.get(ray, {})
    host    = node.get("host") or "—"
    vendor  = node.get("vendor") or ep_info.get("vendor", "—")
    vram_mb = ep_info.get("vram_mb") or 0
    vram_gb = round(vram_mb / 1024) if vram_mb else (node.get("vram_gb") or 0)
    gpu     = _gpu_name(vendor, vram_gb)
    return host, gpu, vendor, vram_gb


# ── Background workers ────────────────────────────────────────────────────────
def _install_deps():
    import os, urllib.request
    if not os.path.exists(WHL_PATH):
        urllib.request.urlretrieve(WHL_URL, WHL_PATH)
    for pkg in ["httpx", "cloudpickle"]:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet", "--break-system-packages", pkg],
            capture_output=True, text=True,
        )
    r = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", "--force-reinstall",
         "--no-deps", "--break-system-packages", WHL_PATH],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        raise RuntimeError(f"SDK install failed: {r.stderr[:400]}")


class _Tee:
    def __init__(self, orig, q): self._orig, self._q = orig, q
    def write(self, s):
        self._orig.write(s); self._orig.flush()
        if s.strip(): self._q.put(("log", s.strip()))
    def flush(self): self._orig.flush()


def _friendly_error(raw: str) -> str:
    r = raw.lower()
    if "expired" in r and ("token" in r or "access" in r):
        return "Your HuggingFace token has expired. Please generate a new one at huggingface.co/settings/tokens."
    if "401" in r or "unauthorized" in r:
        return (
            "HuggingFace token rejected (401 Unauthorized). "
            "Your token may be expired or invalid — generate a new one at huggingface.co/settings/tokens. "
            "If the model is gated (e.g. Llama), make sure you have accepted its license on the model's HuggingFace page."
        )
    if "repositorynotfounderror" in r or ("repository not found" in r):
        return "Model not found on HuggingFace. Check the Model ID is correct and that your token has access to it."
    if "403" in r or "forbidden" in r:
        return "Access denied by HuggingFace. Your token may not have permission to access this model."
    if "failed to deploy" in r and "3 times" in r:
        return "The endpoint failed to start. This is usually caused by an invalid HuggingFace token or a model ID that requires special access."
    if "insufficient" in r and "credit" in r:
        return "Insufficient credits on your GridWeave account."
    if "sdk install failed" in r:
        return "Failed to install the GridWeave SDK. Please check your internet connection and try again."
    return raw


def _deploy_worker(cfg: dict, q: queue.Queue):
    try:
        q.put(("log", "Installing SDK…"))
        _install_deps()
        import importlib, gridweave
        importlib.reload(gridweave)
        gridweave.auth(cfg["admin_token"], platform_url=cfg["platform_url"])
        q.put(("log", f"Deploying {cfg['model_id']} ({cfg['vram']}) as '{cfg['endpoint_name']}'…"))
        old = sys.stdout; sys.stdout = _Tee(old, q)
        try:
            ep = gridweave.serve(
                model=cfg["model_id"], hf_token=cfg["hf_token"],
                s3_endpoint=cfg.get("s3_endpoint", ""),
                s3_access_key=cfg.get("s3_access_key", ""),
                s3_secret_key=cfg.get("s3_secret_key", ""),
                vram=cfg["vram"], name=cfg["endpoint_name"],
            )
        finally:
            sys.stdout = old
        q.put(("done", ep))
    except Exception as exc:
        q.put(("error", _friendly_error(str(exc))))


def _start_worker(name: str, admin_token: str, platform_url: str, q: queue.Queue):
    try:
        import gridweave
        gridweave.auth(admin_token, platform_url=platform_url)
        q.put(("log", f"Starting '{name}'…"))
        old = sys.stdout; sys.stdout = _Tee(old, q)
        try:
            gridweave.start(name)
        finally:
            sys.stdout = old
        ep = gridweave.endpoint(name)
        q.put(("done", ep))
    except Exception as exc:
        q.put(("error", str(exc)))


def _gpu_detect_worker(vendor: str, vram_gb: int,
                        admin_token: str, platform_url: str,
                        q: queue.Queue):
    """Runs nvidia-smi / rocm-smi on a matching remote worker to get the real GPU name."""
    try:
        import gridweave
        gridweave.auth(admin_token, platform_url=platform_url)

        def _probe():
            import os, subprocess

            def _read(path):
                try:
                    with open(path, "r", errors="ignore") as f: return f.read().strip()
                except: return ""

            # Server brand from DMI
            parts = [p for p in [
                _read("/sys/class/dmi/id/sys_vendor"),
                _read("/sys/class/dmi/id/product_name"),
            ] if p and p.lower() not in {"none", "not specified"}]
            server = " ".join(parts) if parts else os.uname().nodename

            # GPU name
            gpu_out = ""
            for cmd in [["nvidia-smi", "-L"], ["rocm-smi", "--showproductname"]]:
                try:
                    r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
                    if r.returncode == 0 and r.stdout.strip():
                        gpu_out = r.stdout.strip(); break
                except FileNotFoundError:
                    continue
                except Exception as e:
                    gpu_out = f"error: {e}"; break

            return {"server": server, "gpu_out": gpu_out}

        probe_fn = gridweave.remote(vram=f"{vram_gb}GB", vendor=vendor)(_probe)
        result   = gridweave.run(probe_fn)

        # Parse "GPU 0: NVIDIA A100-SXM4-80GB (UUID: ...)" → "NVIDIA A100-SXM4-80GB"
        gpu_out  = result.get("gpu_out", "")
        names    = re.findall(r"GPU \d+:\s*(.+?)\s*\(UUID:", gpu_out)
        gpu_name = names[0] if names else (gpu_out.splitlines()[0].strip() if gpu_out else "")

        q.put(("gpu_detected", {
            "vendor":   vendor,
            "vram_gb":  vram_gb,
            "gpu_name": gpu_name,
            "server":   result.get("server", ""),
        }))
    except Exception as exc:
        q.put(("gpu_detect_failed", (vendor, vram_gb, str(exc))))


def _launch(target, args):
    q = queue.Queue()
    st.session_state._result_queue = q
    threading.Thread(target=target, args=(*args, q), daemon=True).start()


def _start_gpu_detect(vendor: str, vram_gb: int):
    """Spin up a GPU probe for this (vendor, vram_gb) if not already running."""
    key = (vendor, vram_gb)
    if key in st.session_state.gpu_cache or key in st.session_state._gpu_detecting:
        return
    if st.session_state._gpu_queue is None:
        st.session_state._gpu_queue = queue.Queue()
    st.session_state._gpu_detecting.add(key)
    threading.Thread(
        target=_gpu_detect_worker,
        args=(vendor, vram_gb,
              st.session_state.admin_token,
              st.session_state.platform_url,
              st.session_state._gpu_queue),
        daemon=True,
    ).start()


# ── Poll queues on every rerun ────────────────────────────────────────────────
def _poll():
    q = st.session_state._result_queue
    if q is None: return
    while True:
        try: kind, value = q.get_nowait()
        except queue.Empty: break
        if   kind == "log":   st.session_state.action_log.append(value)
        elif kind == "done":
            st.session_state.action_state  = "idle"
            st.session_state.endpoint      = value
            st.session_state.chat_history  = []
            st.session_state._result_queue = None
        elif kind == "error":
            st.session_state.action_state  = "error"
            st.session_state.action_error  = value
            st.session_state._result_queue = None


def _gpu_poll():
    q = st.session_state._gpu_queue
    if q is None: return
    while True:
        try: kind, value = q.get_nowait()
        except queue.Empty: break
        if kind == "gpu_detected":
            key = (value["vendor"], value["vram_gb"])
            if value.get("gpu_name"):
                st.session_state.gpu_cache[key] = value["gpu_name"]
            st.session_state._gpu_detecting.discard(key)
        elif kind == "gpu_detect_failed":
            vendor, vram_gb, _ = value
            st.session_state._gpu_detecting.discard((vendor, vram_gb))


_poll()
_gpu_poll()

# ═════════════════════════════════════════════════════════════════════════════
# LOGIN SCREEN
# ═════════════════════════════════════════════════════════════════════════════
if not st.session_state.authenticated:
    st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: radial-gradient(ellipse at 50% -10%, #0d2045 0%, #020c1e 55%, #010810 100%);
}
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stToolbar"] { display: none; }
#MainMenu, footer { visibility: hidden; }

.depot-wrap { text-align: center; padding-top: 2.5rem; }

.depot-ascii {
    display: inline-block;
    font-family: 'Courier New', Courier, monospace;
    font-size: 0.58rem;
    line-height: 1.4;
    color: #29b6f6;
    text-shadow: 0 0 6px #29b6f6, 0 0 16px #0277bd;
    white-space: pre;
    letter-spacing: 0.06em;
}

.scanline {
    width: 62%;
    height: 1px;
    background: linear-gradient(90deg, transparent, #29b6f6 20%, #e3f2fd 50%, #29b6f6 80%, transparent);
    box-shadow: 0 0 10px #29b6f6, 0 0 22px #0277bd;
    margin: 1.4rem auto 0.8rem auto;
}

.uni-title {
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 2.7rem;
    font-weight: 900;
    letter-spacing: 0.3em;
    color: #ffffff;
    text-transform: uppercase;
    text-shadow: 0 0 8px #29b6f6, 0 0 24px #0277bd, 0 0 55px #01579b;
    margin: 0.3rem 0 0.15rem 0;
}

.uni-sub {
    font-family: 'Courier New', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.6em;
    color: #29b6f6;
    text-transform: uppercase;
    text-shadow: 0 0 8px #29b6f6;
    margin-bottom: 0.4rem;
    opacity: 0.9;
}

.access-label {
    color: #81d4fa;
    font-size: 0.68rem;
    letter-spacing: 0.4em;
    text-align: center;
    font-family: 'Courier New', monospace;
    text-transform: uppercase;
    margin: 1.2rem 0 0.6rem 0;
    text-shadow: 0 0 6px #29b6f6;
}

[data-testid="stTextInput"] > label {
    color: #81d4fa !important;
    font-size: 0.68rem !important;
    letter-spacing: 0.2em !important;
    text-transform: uppercase !important;
    font-family: 'Courier New', monospace !important;
}
[data-testid="stTextInput"] input {
    background: rgba(1, 18, 40, 0.88) !important;
    border: 1px solid rgba(41, 182, 246, 0.4) !important;
    color: #e3f2fd !important;
    border-radius: 4px !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: #29b6f6 !important;
    box-shadow: 0 0 0 1px rgba(41,182,246,0.55), 0 0 14px rgba(41,182,246,0.22) !important;
}
div[data-testid="stButton"] button[kind="primary"] {
    background: linear-gradient(135deg, #01579b 0%, #0277bd 100%) !important;
    border: 1px solid #29b6f6 !important;
    color: #e3f2fd !important;
    font-weight: 700 !important;
    letter-spacing: 0.28em !important;
    text-transform: uppercase !important;
    border-radius: 4px !important;
    box-shadow: 0 0 18px rgba(41,182,246,0.28) !important;
}
div[data-testid="stButton"] button[kind="primary"]:hover {
    background: linear-gradient(135deg, #0277bd 0%, #0288d1 100%) !important;
    box-shadow: 0 0 28px rgba(41,182,246,0.48) !important;
}
</style>

<div class="depot-wrap">
  <div class="depot-ascii">
╔══════════════════════════════════════════════════════════════════════════╗
║                                                                          ║
║  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ║
║  │░░░░░░░░░░│  │░░░░░░░░░░│  │▓▓▓▓▓▓▓▓▓▓│  │░░░░░░░░░░│  │░░░░░░░░░░│  ║
║  │  SERVER  │  │  SERVER  │  │   GPU    │  │  SERVER  │  │  SERVER  │  ║
║  │   RACK   │  │   RACK   │  │ CLUSTER  │  │   RACK   │  │   RACK   │  ║
║  │ ● ● ● ● │  │ ● ● ● ● │  │ ■ ■ ■ ■ │  │ ● ● ● ● │  │ ● ● ● ● │  ║
║  │ ○ ○ ○ ○ │  │ ○ ○ ○ ○ │  │ □ □ □ □ │  │ ○ ○ ○ ○ │  │ ○ ○ ○ ○ │  ║
║  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘  ║
╠═══════╧═════════════╧═════════════╧═════════════╧═════════════╧═════════╣
║   ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬ COMPUTE DEPOT PLATFORM ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬  ║
╚══════════════════════════════════════════════════════════════════════════╝</div>
  <div class="scanline"></div>
  <div class="uni-title">Groningen University</div>
  <div class="uni-sub">◈ &nbsp; A I &nbsp; C o m p u t e &nbsp; D e p o t &nbsp; ◈</div>
</div>
""", unsafe_allow_html=True)

    _, centre, _ = st.columns([1, 2, 1])
    with centre:
        st.markdown('<div class="access-label">⬡ &nbsp; Secure Access Terminal &nbsp; ⬡</div>',
                    unsafe_allow_html=True)
        platform_url_in = st.text_input("Platform URL", value=st.session_state.platform_url)
        token_in = st.text_input("User Token", type="password", placeholder="Enter your user token")
        if st.button("⚡  Authenticate", type="primary", use_container_width=True):
            if not token_in.strip():
                st.error("Please enter your user token.")
            else:
                with st.spinner("Verifying…"):
                    try:
                        import gridweave as _gw_check
                        _gw_check.auth(token_in.strip(), platform_url=platform_url_in)
                        _gw_check.endpoints()
                        st.session_state.admin_token   = token_in.strip()
                        st.session_state.platform_url  = platform_url_in
                        st.session_state.authenticated = True
                        st.rerun()
                    except ImportError:
                        st.session_state.admin_token   = token_in.strip()
                        st.session_state.platform_url  = platform_url_in
                        st.session_state.authenticated = True
                        st.rerun()
                    except Exception as e:
                        st.error(f"Authentication failed: {e}")

    st.markdown("""
<div style="text-align:center;margin-top:2.5rem;color:#4a7a9b;
            font-family:'Courier New',monospace;font-size:0.62rem;
            letter-spacing:0.08em;opacity:0.75;">
    With support of Dell Technologies, GridWeave and Groningen University CIT
</div>
""", unsafe_allow_html=True)
    st.stop()

# ═════════════════════════════════════════════════════════════════════════════
# MAIN UI
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: radial-gradient(ellipse at 50% 0%, #0a1a35 0%, #020c1e 60%, #010810 100%);
}
[data-testid="stHeader"] {
    background: rgba(1,8,20,0.97) !important;
    border-bottom: 1px solid rgba(41,182,246,0.18) !important;
}
[data-testid="stToolbar"] { display: none; }
#MainMenu, footer { visibility: hidden; }
.stApp, .stApp p { color: #cce7ff; }
h1, h2, h3 { color: #ffffff !important; }
hr { border-color: rgba(41,182,246,0.2) !important; }
details[data-testid="stExpander"] {
    background: rgba(2,15,35,0.75) !important;
    border: 1px solid rgba(41,182,246,0.22) !important;
    border-radius: 8px !important;
}
details[data-testid="stExpander"] summary { color: #81d4fa !important; font-weight: 600; }
[data-testid="stTextInput"] > label, [data-testid="stSelectbox"] > label,
[data-testid="stSlider"] > label {
    color: #81d4fa !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    font-family: 'Courier New', monospace !important;
}
[data-testid="stTextInput"] input {
    background: rgba(1,15,35,0.9) !important;
    border: 1px solid rgba(41,182,246,0.38) !important;
    color: #e3f2fd !important;
    border-radius: 4px !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: #29b6f6 !important;
    box-shadow: 0 0 0 1px rgba(41,182,246,0.5), 0 0 10px rgba(41,182,246,0.15) !important;
}
[data-testid="stSelectbox"] > div > div {
    background: rgba(1,15,35,0.9) !important;
    border: 1px solid rgba(41,182,246,0.38) !important;
    color: #e3f2fd !important;
    border-radius: 4px !important;
}
div[data-testid="stButton"] button[kind="primary"] {
    background: linear-gradient(135deg, #01579b 0%, #0277bd 100%) !important;
    border: 1px solid #29b6f6 !important;
    color: #e3f2fd !important;
    font-weight: 700 !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    border-radius: 4px !important;
    box-shadow: 0 0 14px rgba(41,182,246,0.22) !important;
}
div[data-testid="stButton"] button[kind="primary"]:hover {
    background: linear-gradient(135deg, #0277bd 0%, #039be5 100%) !important;
    box-shadow: 0 0 22px rgba(41,182,246,0.4) !important;
}
div[data-testid="stButton"] button[kind="secondary"] {
    background: rgba(1,25,55,0.8) !important;
    border: 1px solid rgba(41,182,246,0.32) !important;
    color: #81d4fa !important;
    border-radius: 4px !important;
}
div[data-testid="stButton"] button[kind="secondary"]:hover {
    background: rgba(2,45,90,0.9) !important;
    border-color: #29b6f6 !important;
    color: #e3f2fd !important;
}
[data-testid="stAlert"] {
    background: rgba(1,25,55,0.85) !important;
    border-radius: 0 6px 6px 0 !important;
}
[data-testid="stAlert"] p { color: #e3f2fd !important; }
[data-testid="stCode"] code, [data-testid="stCode"] pre {
    background: rgba(1,10,22,0.95) !important;
    color: #81d4fa !important;
    border: 1px solid rgba(41,182,246,0.18) !important;
    border-radius: 4px !important;
}
[data-testid="stCaptionContainer"] p { color: #81d4fa !important; }
[data-testid="stChatMessage"] {
    background: rgba(1,15,35,0.65) !important;
    border: 1px solid rgba(41,182,246,0.12) !important;
    border-radius: 10px !important;
}
[data-testid="stChatInputContainer"] textarea {
    background: rgba(1,15,35,0.9) !important;
    border: 1px solid rgba(41,182,246,0.38) !important;
    color: #e3f2fd !important;
    border-radius: 4px !important;
}
[data-testid="stChatInputContainer"] textarea:focus {
    border-color: #29b6f6 !important;
    box-shadow: 0 0 0 1px rgba(41,182,246,0.5) !important;
}
</style>
""", unsafe_allow_html=True)

try:
    import gridweave as _gw
    _gw.auth(st.session_state.admin_token, platform_url=st.session_state.platform_url)
    _gw_available = True
except ImportError:
    _gw_available = False

h_left, h_right = st.columns([5, 1])
with h_left:
    st.markdown("""
<div style="padding:0.4rem 0 0.2rem 0;">
  <span style="font-size:1.7rem;font-weight:900;letter-spacing:0.15em;color:#fff;
               text-shadow:0 0 8px #29b6f6,0 0 22px #0277bd;text-transform:uppercase;
               font-family:'Segoe UI',Arial,sans-serif;">🏛 Groningen University</span>
  <span style="font-size:0.65rem;letter-spacing:0.4em;color:#29b6f6;
               font-family:'Courier New',monospace;text-transform:uppercase;
               margin-left:1.2rem;text-shadow:0 0 6px #29b6f6;vertical-align:middle;">
    AI Compute Depot</span>
</div>""", unsafe_allow_html=True)
with h_right:
    st.write("")
    if st.button("Logout", use_container_width=True):
        for k in ["authenticated", "admin_token", "endpoint", "chat_history",
                  "action_state", "action_log", "action_error", "_result_queue",
                  "gpu_cache", "_gpu_detecting", "_gpu_queue"]:
            st.session_state[k] = (
                False  if k == "authenticated" else
                ""     if k == "admin_token"   else
                "idle" if k == "action_state"  else
                None   if k in ("endpoint", "action_error", "_result_queue", "_gpu_queue") else
                set()  if k == "_gpu_detecting" else
                {}     if k == "gpu_cache"      else [])
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# 1. Endpoint Manager
# ══════════════════════════════════════════════════════════════════════════════
with st.expander("📡 Endpoint Manager", expanded=(st.session_state.endpoint is None)):

    if _gw_available:
        hdr_col, _, ref_col = st.columns([4, 3, 1])
        with hdr_col: st.subheader("Your Endpoints")
        with ref_col: st.button("🔄 Refresh", use_container_width=True)

        try:
            _gw.auth(st.session_state.admin_token, platform_url=st.session_state.platform_url)
            eps = _gw.endpoints()
            hw  = _hw_lookup()
        except Exception as e:
            eps = []; hw = {}; st.warning(f"Could not load endpoints: {e}")

        # Split into own vs others by user-id prefix in the qualified name
        try:
            from gridweave.auth import get_user_id as _get_uid
            _uid = _get_uid()
        except Exception:
            _uid = None
        my_eps    = [e for e in eps if _uid and e.get("name", "").startswith(f"{_uid}/")]
        other_eps = [e for e in eps if not (_uid and e.get("name", "").startswith(f"{_uid}/"))]

        _CW = [3, 2, 3, 3, 1, 1, 1, 1]

        # ── Your Endpoints ──────────────────────────────────────────────────
        if my_eps:
            h1,h2,h3,h4,h5,_,_,_ = st.columns(_CW)
            h1.caption("Endpoint"); h2.caption("Status"); h3.caption("Model")
            h4.caption("Server");   h5.caption("×GPU")

            for ep_info in my_eps:
                name   = ep_info.get("name", "")
                status = ep_info.get("status", "")
                model  = ep_info.get("model", "—")
                gpus   = ep_info.get("gpus", "?")
                icon   = "🟢" if status == "running" else ("🟡" if status in ("deploying","allocating") else "🔴")
                host, gpu, vendor, vram_gb = _hw(ep_info, hw)

                c1,c2,c3,c4,c5,c6,c7,c8 = st.columns(_CW)
                c1.write(f"**{name}**")
                c2.write(f"{icon} {status}")
                c3.write(model.split("/")[-1])
                c4.write(host)
                c5.write(str(gpus))

                with c6:
                    if status == "running" and st.button("Chat", key=f"chat_{name}", use_container_width=True):
                        from gridweave.serve import Endpoint
                        st.session_state.endpoint = Endpoint(
                            name=ep_info["name"], status="running",
                            endpoint_type=ep_info.get("endpoint_type", "vllm"),
                            model=ep_info.get("model", ""), image=ep_info.get("image", ""),
                            gpus=ep_info.get("gpus", 1), vendor=ep_info.get("vendor"),
                            spec=ep_info.get("spec"),
                        )
                        st.session_state.chat_history = []
                        st.rerun()
                with c7:
                    if status == "running":
                        if st.button("Stop", key=f"stop_{name}", use_container_width=True):
                            _gw.auth(st.session_state.admin_token, platform_url=st.session_state.platform_url)
                            _gw.stop(name)
                            if st.session_state.endpoint and st.session_state.endpoint.name == name:
                                st.session_state.endpoint = None
                            st.rerun()
                    elif status == "stopped":
                        if st.button("Start", key=f"start_{name}", use_container_width=True):
                            _gw.auth(st.session_state.admin_token, platform_url=st.session_state.platform_url)
                            st.session_state.action_state = "busy"
                            st.session_state.action_label = f"Starting '{name}'…"
                            st.session_state.action_log   = []
                            st.session_state.action_error = None
                            _launch(_start_worker, (name, st.session_state.admin_token,
                                                    st.session_state.platform_url))
                            st.rerun()
                with c8:
                    if st.button("Delete", key=f"del_{name}", use_container_width=True):
                        _gw.auth(st.session_state.admin_token, platform_url=st.session_state.platform_url)
                        _gw.delete(name)
                        if st.session_state.endpoint and st.session_state.endpoint.name == name:
                            st.session_state.endpoint = None
                        st.rerun()
        else:
            st.info("No endpoints found.")

        # ── Other Endpoints ─────────────────────────────────────────────────
        if other_eps:
            st.divider()
            st.subheader("Other Endpoints")
            h1,h2,h3,h4,h5,_,_,_ = st.columns(_CW)
            h1.caption("Endpoint"); h2.caption("Status"); h3.caption("Model")
            h4.caption("Server");   h5.caption("×GPU")

            for ep_info in other_eps:
                name   = ep_info.get("name", "")
                status = ep_info.get("status", "")
                model  = ep_info.get("model", "—")
                gpus   = ep_info.get("gpus", "?")
                icon   = "🟢" if status == "running" else ("🟡" if status in ("deploying","allocating") else "🔴")
                host, gpu, vendor, vram_gb = _hw(ep_info, hw)

                c1,c2,c3,c4,c5,c6,_,_ = st.columns(_CW)
                c1.write(f"**{name}**")
                c2.write(f"{icon} {status}")
                c3.write(model.split("/")[-1])
                c4.write(host)
                c5.write(str(gpus))

                with c6:
                    if status == "running" and st.button("Chat", key=f"chat_{name}", use_container_width=True):
                        from gridweave.serve import Endpoint
                        st.session_state.endpoint = Endpoint(
                            name=ep_info["name"], status="running",
                            endpoint_type=ep_info.get("endpoint_type", "vllm"),
                            model=ep_info.get("model", ""), image=ep_info.get("image", ""),
                            gpus=ep_info.get("gpus", 1), vendor=ep_info.get("vendor"),
                            spec=ep_info.get("spec"),
                        )
                        st.session_state.chat_history = []
                        st.rerun()

        # Auto-refresh while any GPU probe is in flight
        if st.session_state._gpu_detecting:
            time.sleep(3); st.rerun()
    else:
        st.info("Deploy a model below to install the SDK and create your first endpoint.")

    st.divider()
    st.subheader("Deploy New Endpoint")

    with st.expander("🔑 Credentials", expanded=False):
        source = st.radio(
            "Model source",
            ["🤗 HuggingFace", "📦 S3/R2 (own LLMs)"],
            index=None,
            horizontal=True,
            help="Choose where to load the model from. Select HuggingFace for public or gated models, or S3/R2 if you host your own pre-downloaded models.",
        )
        use_hf = source == "🤗 HuggingFace"
        use_s3 = source == "📦 S3/R2 (own LLMs)"

        cc1, cc2 = st.columns(2)
        with cc1:
            hf_token = st.text_input(
                "HuggingFace Token", value="", type="password",
                disabled=not use_hf,
                help="Your HuggingFace access token. Required for gated models such as Llama. Generate one at huggingface.co/settings/tokens.",
            )
        with cc2:
            s3_endpoint   = st.text_input(
                "S3/R2 Endpoint", placeholder="https://<account>.r2.cloudflarestorage.com",
                disabled=not use_s3,
                help="URL of your S3-compatible storage endpoint (e.g. Cloudflare R2 or AWS S3).",
            )
            s3_access_key = st.text_input(
                "S3/R2 Access Key", placeholder="your-access-key-id", type="password",
                disabled=not use_s3,
                help="Access key ID for your S3/R2 bucket.",
            )
            s3_secret_key = st.text_input(
                "S3/R2 Secret Key", placeholder="your-secret-access-key", type="password",
                disabled=not use_s3,
                help="Secret access key for your S3/R2 bucket.",
            )
            s3_bucket = st.text_input(
                "S3/R2 Bucket", placeholder="my-bucket",
                disabled=not use_s3,
                help="Name of the bucket where your model files are stored.",
            )

    ma, mb, mc = st.columns(3)
    with ma: model_id      = st.text_input("Model ID",      value="meta-llama/Llama-3.2-1B-Instruct")
    with mb: vram          = st.selectbox("VRAM", ["4GB","8GB","16GB","24GB","40GB","80GB"])
    with mc: endpoint_name = st.text_input("Endpoint Name", value="llama-eric")

    action = st.session_state.action_state
    if action == "idle":
        if st.button("🚀 Deploy", type="primary", use_container_width=True):
            if not source:
                st.error("Please select a model source (HuggingFace or S3/R2) in the Credentials section.")
            elif use_hf and not hf_token:
                st.error("Please enter your HuggingFace token in the Credentials section.")
            elif use_s3 and not s3_endpoint:
                st.error("Please enter your S3/R2 endpoint in the Credentials section.")
            else:
                st.session_state.action_state = "busy"
                st.session_state.action_label = f"Deploying {model_id}…"
                st.session_state.action_log   = []
                st.session_state.action_error = None
                _launch(_deploy_worker, (dict(
                    platform_url=st.session_state.platform_url,
                    admin_token=st.session_state.admin_token,
                    hf_token=hf_token if use_hf else "",
                    s3_endpoint=s3_endpoint if use_s3 else "",
                    s3_access_key=s3_access_key if use_s3 else "",
                    s3_secret_key=s3_secret_key if use_s3 else "",
                    model_id=model_id, vram=vram, endpoint_name=endpoint_name,
                ),))
                st.rerun()
    elif action == "busy":
        st.info(st.session_state.action_label)
        st.code("\n".join(st.session_state.action_log) or "Starting…", language=None)
        time.sleep(2); st.rerun()
    elif action == "error":
        st.error(st.session_state.action_error)
        if st.button("↩ Retry"):
            st.session_state.action_state = "idle"; st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# 2. Chat
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.endpoint:
    ep = st.session_state.endpoint
    st.divider()

    try:
        _active_hw = _hw_lookup() if _gw_available else {}
        _ep_info   = {}
        for _e in _gw.endpoints():
            if _e.get("name") == ep.name:
                _ep_info = _e; break
        _host, _gpu, _vendor, _vgb = _hw(_ep_info, _active_hw)
        _gpus = _ep_info.get("gpus", ep.gpus)
        if _vendor != "—" and _vgb:
            _start_gpu_detect(_vendor, _vgb)
    except Exception:
        _host, _gpu, _vendor, _vgb, _gpus = "—", ep.vendor or "—", ep.vendor or "—", 0, ep.gpus

    short_model = ep.model.split("/")[-1]
    st.success(f"✅ **{ep.name}**")
    st.caption(
        f"🖥 **Server:** {_host}  &nbsp;·&nbsp;  "
        f"🤖 **Model:** {short_model}  &nbsp;·&nbsp;  "
        f"⚡ **GPU:** {_gpu}  &nbsp;·&nbsp;  "
        f"🔢 **×{_gpus}**"
    )

    # Refresh while GPU name is still being detected
    if _gpu == "detecting…":
        time.sleep(3); st.rerun()

    disc_col, _ = st.columns([1, 5])
    with disc_col:
        if st.button("✖ Disconnect", use_container_width=True):
            st.session_state.endpoint = None
            st.session_state.chat_history = []
            st.rerun()

    left, right = st.columns([3, 1])
    _is_instruct = "instruct" in ep.model.lower()

    with right:
        st.caption("Settings")
        max_tokens  = st.slider("Max tokens",  64, 2048, 512, step=64)
        temperature = st.slider("Temperature", 0.0, 2.0,  0.7, step=0.05)
        if not _is_instruct:
            st.warning("Base model — completion mode", icon="⚠️")
        if st.button("🗑 Clear chat", use_container_width=True):
            st.session_state.chat_history = []; st.rerun()

    with left:
        st.subheader(f"Chat — {short_model}")

        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        if prompt := st.chat_input(f"Message {short_model}…"):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)
            with st.chat_message("assistant"):
                with st.spinner("Thinking…"):
                    try:
                        if _is_instruct:
                            response = ep.chat(
                                [{"role": m["role"], "content": m["content"]}
                                 for m in st.session_state.chat_history],
                                max_tokens=max_tokens, temperature=temperature,
                            )
                        else:
                            prompt_text = "\n".join(
                                f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
                                for m in st.session_state.chat_history
                            ) + "\nAssistant:"
                            raw = ep.generate(prompt_text, max_tokens=max_tokens,
                                              temperature=temperature)
                            response = re.split(r"\n(User|Assistant):", raw)[0].strip()
                    except Exception as exc:
                        response = f"⚠️ Error: {exc}"
                st.write(response)
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            st.rerun()

else:
    if st.session_state.action_state == "idle":
        st.info("Deploy a new endpoint or click **Chat** on an existing one above to start chatting.")
