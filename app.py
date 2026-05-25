import base64
import json
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

_DEFAULT_LLM_LIST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "llm-list.txt")

def _llm_list_path(uid: str | None) -> str:
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, f"llm-list-{uid}.txt") if uid else _DEFAULT_LLM_LIST

def _load_llm_list(uid: str | None) -> list[str]:
    paths = [_llm_list_path(uid), _DEFAULT_LLM_LIST] if uid else [_DEFAULT_LLM_LIST]
    for path in paths:
        try:
            models = [l.strip() for l in open(path) if l.strip()]
            if models:
                return models
        except Exception:
            continue
    return ["Qwen/Qwen2.5-0.5B-Instruct"]

def _save_llm_list(models: list[str], uid: str | None):
    try:
        with open(_llm_list_path(uid), "w") as f:
            f.write("\n".join(models) + "\n")
    except Exception:
        pass

# ── Session state ─────────────────────────────────────────────────────────────
for _k, _v in [
    ("authenticated",  False),
    ("user_token",     ""),
    ("model_list",     None),
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


def _token_expires_at(token: str) -> float:
    """Decode JWT exp claim without verification. Returns 0 if unreadable."""
    try:
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        data = json.loads(base64.b64decode(payload))
        return float(data.get("exp", 0))
    except Exception:
        return 0


def _token_ok(token: str) -> bool:
    exp = _token_expires_at(token)
    return exp == 0 or exp > time.time()


def _token_warning(token: str) -> str | None:
    """Return a warning string if the token expires soon, else None."""
    exp = _token_expires_at(token)
    if exp == 0:
        return None
    remaining = exp - time.time()
    if remaining <= 0:
        return "Your GridWeave session has expired. Please log out and log in again."
    if remaining < 300:
        mins = int(remaining // 60)
        return f"Your GridWeave session expires in {mins} minute(s). Save your work and log in again soon."
    return None


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
        if "huggingface" in r or "hf_token" in r:
            return "Your HuggingFace token has expired. Please generate a new one at huggingface.co/settings/tokens."
        return "Your GridWeave session has expired. Please log out and log in again."
    if "401" in r or "unauthorized" in r:
        return (
            "This model requires a HuggingFace token, or your token does not have access. "
            "Open Credentials and enter a valid HuggingFace token. "
            "For gated models (e.g. Llama) also accept the license on the model's HuggingFace page."
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


def _gw_direct(method: str, path: str, **kwargs):
    """Call the GridWeave API with proper URL encoding."""
    import urllib.parse, httpx as _hx
    from gridweave.auth import get_platform_url, _headers
    safe = "/".join(urllib.parse.quote(seg, safe="") for seg in path.split("/"))
    r = getattr(_hx, method)(f"{get_platform_url()}{safe}", headers=_headers(), timeout=30, **kwargs)
    if not r.is_success:
        try:
            d = r.json(); detail = d.get("detail", d) if isinstance(d, dict) else d
        except Exception:
            detail = r.text or "(no body)"
        raise RuntimeError(f"{r.status_code}: {detail}")
    return r.json()


def _gw_qname(name: str, uid: str = None) -> str:
    """Qualify an endpoint name (prepend user_id/) with a bare-name fallback."""
    if "/" in name:
        return name
    if uid:
        return f"{uid}/{name}"
    try:
        from gridweave.auth import qualify_name
        return qualify_name(name)
    except Exception:
        return name


def _split_ep_name(name: str) -> tuple[str, str]:
    """Return (short_name, user) by splitting on '/' then '--'."""
    if "/" in name:
        user, _, rest = name.partition("/")
        return _ep_short_name(rest), user
    if "--" in name:
        user, _, short = name.partition("--")
        return short, user
    return name, "—"


def _ep_short_name(name: str) -> str:
    """Return the inference-proxy short name, stripping uid/ and DisplayName-- prefixes."""
    if "/" in name:
        name = name.partition("/")[2]
    if "--" in name:
        name = name.partition("--")[2]
    return name


def _deploy_worker(cfg: dict, q: queue.Queue):
    try:
        q.put(("log", "Installing SDK…"))
        _install_deps()
        import importlib, gridweave
        importlib.reload(gridweave)
        gridweave.auth(cfg["user_token"], platform_url=cfg["platform_url"])
        q.put(("log", f"Deploying {cfg['model_id']} ({cfg['vram']}) as '{cfg['endpoint_name']}'…"))
        old = sys.stdout; sys.stdout = _Tee(old, q)
        try:
            serve_kwargs = dict(
                model=cfg["model_id"],
                vram=cfg["vram"],
                name=cfg["endpoint_name"],
            )
            if cfg.get("hf_token"):
                serve_kwargs["hf_token"] = cfg["hf_token"]
            if cfg.get("s3_endpoint"):
                serve_kwargs["s3_endpoint"]   = cfg["s3_endpoint"]
                serve_kwargs["s3_access_key"] = cfg.get("s3_access_key", "")
                serve_kwargs["s3_secret_key"] = cfg.get("s3_secret_key", "")
            ep = gridweave.serve(**serve_kwargs)
        finally:
            sys.stdout = old
        ep.name = _ep_short_name(ep.name)
        q.put(("done", ep))
    except Exception as exc:
        q.put(("error", _friendly_error(str(exc))))


def _start_worker(name: str, user_token: str, platform_url: str, uid: str, q: queue.Queue):
    try:
        import gridweave
        gridweave.auth(user_token, platform_url=platform_url)
        q.put(("log", f"Starting '{name}'…"))
        _gw_direct("post", f"/v1/endpoints/{name}/start")
        t0 = time.time()
        while True:
            try:
                h = _gw_direct("get", f"/v1/endpoints/{name}/health")
                if h.get("failure"):
                    raise RuntimeError(f"Start failed: {h.get('failure_reason', 'unknown')}")
                if h.get("db_status") == "running":
                    break
                state = h.get("pod_phase") or h.get("db_status") or "?"
                q.put(("log", f"  [endpoint] {int(time.time() - t0)}s — {state}"))
            except RuntimeError:
                raise
            except Exception:
                pass
            time.sleep(3)
        data = _gw_direct("get", f"/v1/endpoints/{name}")
        from gridweave.serve import Endpoint
        ep = Endpoint(
            name=data.get("display_name") or _ep_short_name(data["name"]), status="running",
            endpoint_type=data.get("endpoint_type", "vllm"),
            model=data.get("model", ""), image=data.get("image", ""),
            gpus=data.get("gpus", 1), vendor=data.get("vendor"), spec=data.get("spec"),
        )
        q.put(("done", ep))
    except Exception as exc:
        q.put(("error", str(exc)))


def _gpu_detect_worker(vendor: str, vram_gb: int,
                        user_token: str, platform_url: str,
                        q: queue.Queue):
    """Runs nvidia-smi / rocm-smi on a matching remote worker to get the real GPU name."""
    try:
        import gridweave
        gridweave.auth(user_token, platform_url=platform_url)

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
              st.session_state.user_token,
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
/* ── Reset ── */
[data-testid="stAppViewContainer"] { background: #010810 !important; }
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stToolbar"] { display: none; }
#MainMenu, footer { visibility: hidden; }

/* ── Full-page 3D scene (fixed behind everything) ── */
#lscene {
    position: fixed; inset: 0; z-index: 0; overflow: hidden;
    background: radial-gradient(ellipse at 50% 5%, #0b1e45 0%, #020c1e 50%, #010810 100%);
}

/* Perspective grid floor */
#lscene .grid {
    position: absolute; bottom: 0; left: -50%; right: -50%; height: 15%;
    background-image:
        linear-gradient(rgba(41,182,246,0.13) 1px, transparent 1px),
        linear-gradient(90deg, rgba(41,182,246,0.13) 1px, transparent 1px);
    background-size: 72px 72px;
    transform: perspective(440px) rotateX(76deg);
    transform-origin: center bottom;
    mask-image: radial-gradient(ellipse at 50% 100%, rgba(0,0,0,1) 25%, rgba(0,0,0,0) 72%);
    -webkit-mask-image: radial-gradient(ellipse at 50% 100%, rgba(0,0,0,1) 25%, rgba(0,0,0,0) 72%);
}

/* Horizon glow */
#lscene .horizon {
    position: absolute; bottom: 15%; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent 0%, #0277bd 12%, #29b6f6 50%, #0277bd 88%, transparent 100%);
    box-shadow: 0 0 50px 14px rgba(41,182,246,0.2), 0 -40px 80px rgba(41,182,246,0.05);
}

/* Vertical light beams */
.vbeam {
    position: absolute; bottom: 15%; width: 2px; top: 0;
    background: linear-gradient(180deg, transparent 5%, rgba(41,182,246,0.03) 55%, rgba(41,182,246,0.16) 100%);
    animation: bp 4s ease-in-out infinite;
}
.vb1 { left: 18%; animation-delay: 0s; }
.vb2 { left: 35%; animation-delay: -1.4s; }
.vb3 { left: 65%; animation-delay: -2.8s; }
.vb4 { left: 82%; animation-delay: -0.7s; }
@keyframes bp { 0%,100%{opacity:.55} 50%{opacity:1} }

/* ── Server rack row ── */
#lscene .racks {
    position: absolute; bottom: 13%; left: 50%;
    transform: translateX(-50%) scale(1.1);
    transform-origin: center bottom;
    display: flex; gap: 20px; align-items: flex-end;
}

/* Individual rack wrapper */
.rack3d {
    position: relative;
    display: flex; flex-direction: column;
    filter: drop-shadow(0 0 14px rgba(41,182,246,0.2));
    animation: rf 7s ease-in-out infinite;
}
.rack3d:nth-child(1){animation-delay:0s}
.rack3d:nth-child(2){animation-delay:-2.2s}
.rack3d:nth-child(3){animation-delay:-1.1s;animation-duration:5.8s}
.rack3d:nth-child(4){animation-delay:-3.6s}
.rack3d:nth-child(5){animation-delay:-0.6s;animation-duration:8.2s}
@keyframes rf { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-8px)} }

/* Top face (isometric illusion via skew) */
.rack3d .rt {
    height: 12px;
    background: linear-gradient(135deg, #1e4e84 0%, #0d2d58 55%, #061f3e 100%);
    border: 1px solid rgba(41,182,246,0.55); border-bottom: none;
    transform: skewX(-28deg);
    margin-left: 7px; /* aligns skewed bottom-left with front face x=0 */
    flex-shrink: 0;
}

/* Front face */
.rack3d .rf {
    background: linear-gradient(160deg, #0d2548 0%, #061830 55%, #030e22 100%);
    border: 1px solid rgba(41,182,246,0.48);
    border-top: none;
    padding: 5px 4px; display: flex; flex-direction: column; gap: 4px;
}

/* Right-side shadow strip (depth cue) */
.rack3d .rs {
    position: absolute; right: -10px; top: 12px;
    width: 10px;
    background: linear-gradient(180deg, #041228 0%, #020c1e 100%);
    border: 1px solid rgba(41,182,246,0.16); border-left: none;
    bottom: 0;
    transform: skewY(-2deg); transform-origin: top left;
}

/* Front face — tighter gap for SVG server rows */
.rack3d .rf { gap: 2px !important; }

/* ── Floating header above the scene ── */
.lhdr {
    position: relative; z-index: 10;
    text-align: center; padding-top: 1.8rem;
}
.uni-name {
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 2.125rem; font-weight: 900; letter-spacing: 0.3em;
    color: #fff; text-transform: uppercase;
    text-shadow: 0 0 10px #29b6f6, 0 0 28px #0277bd, 0 0 65px #01579b;
    margin: 0 0 0.12rem 0;
}
.uni-sub {
    font-family: 'Courier New', monospace;
    font-size: 0.67rem; letter-spacing: 0.56em;
    color: #29b6f6; text-transform: uppercase;
    text-shadow: 0 0 8px #29b6f6; opacity: 0.9;
}
.scanline {
    width: 52%; height: 1px; margin: 1.1rem auto 0.6rem;
    background: linear-gradient(90deg, transparent, #29b6f6 20%, #e3f2fd 50%, #29b6f6 80%, transparent);
    box-shadow: 0 0 10px #29b6f6, 0 0 24px #0277bd;
}

/* ── Glassmorphism login card (center Streamlit column) ── */
[data-testid="column"]:nth-child(2) > div:first-child {
    background: rgba(2,14,36,0.82) !important;
    border: 1px solid rgba(41,182,246,0.32) !important;
    border-radius: 16px !important;
    backdrop-filter: blur(24px) !important;
    -webkit-backdrop-filter: blur(24px) !important;
    box-shadow:
        0 0 0 1px rgba(41,182,246,0.05),
        0 12px 40px rgba(0,0,0,0.55),
        0 0 70px rgba(41,182,246,0.07),
        inset 0 1px 0 rgba(255,255,255,0.06) !important;
    padding: 1.7rem 1.5rem 2rem !important;
    position: relative !important; z-index: 10 !important;
}

.access-label {
    color: #81d4fa; font-size: 0.62rem; letter-spacing: 0.44em;
    text-align: center; font-family: 'Courier New', monospace;
    text-transform: uppercase; margin: 0 0 1rem 0;
    text-shadow: 0 0 6px #29b6f6;
}

/* Form inputs */
[data-testid="stTextInput"] > label {
    color: #81d4fa !important; font-size: 0.58rem !important;
    letter-spacing: 0.14em !important; text-transform: uppercase !important;
    font-family: 'Courier New', monospace !important;
}
[data-testid="stTextInput"] input {
    background: rgba(1,18,44,0.92) !important;
    border: 1px solid rgba(41,182,246,0.3) !important;
    color: #e3f2fd !important; border-radius: 6px !important;
    font-family: 'Courier New', monospace !important;
    font-size: 0.72rem !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: #29b6f6 !important;
    box-shadow: 0 0 0 1px rgba(41,182,246,0.45), 0 0 18px rgba(41,182,246,0.18) !important;
}

/* Authenticate button */
div[data-testid="stButton"] button[kind="primary"] {
    background: linear-gradient(135deg, #01579b 0%, #0277bd 100%) !important;
    border: 1px solid rgba(41,182,246,0.55) !important;
    color: #e3f2fd !important; font-weight: 600 !important;
    letter-spacing: 0.12em !important; text-transform: none !important;
    border-radius: 6px !important; font-size: 0.8rem !important;
    padding: 0.35rem 1rem !important;
    box-shadow: 0 0 18px rgba(41,182,246,0.2), inset 0 1px 0 rgba(255,255,255,0.1) !important;
    transition: all 0.2s ease !important;
}
div[data-testid="stButton"] button[kind="primary"]:hover {
    background: linear-gradient(135deg, #0277bd 0%, #039be5 100%) !important;
    box-shadow: 0 0 32px rgba(41,182,246,0.44) !important;
    transform: translateY(-1px) !important;
}
</style>

<div id="lscene">
  <div class="grid"></div>
  <div class="horizon"></div>
  <div class="vbeam vb1"></div>
  <div class="vbeam vb2"></div>
  <div class="vbeam vb3"></div>
  <div class="vbeam vb4"></div>
  <div class="racks">
  <div class="rack3d">
  <div class="rt" style="width:60px"></div>
  <div class="rf" style="width:60px">
  <svg width="52" height="11" viewBox="0 0 52 11"><rect width="52" height="11" fill="#1e1e1e"/><rect width="52" height="0.5" fill="#353535"/><rect y="10.5" width="52" height="0.5" fill="#0d0d0d"/><rect x="1" y="2" width="5.5" height="7" fill="#141414" stroke="#272727" stroke-width="0.5" rx="0.3"/><rect x="8" y="2" width="5.5" height="7" fill="#141414" stroke="#272727" stroke-width="0.5" rx="0.3"/><rect x="15" y="2" width="5.5" height="7" fill="#141414" stroke="#272727" stroke-width="0.5" rx="0.3"/><rect x="22" y="2" width="5.5" height="7" fill="#141414" stroke="#272727" stroke-width="0.5" rx="0.3"/><rect x="1.5" y="2.5" width="1.2" height="1.2" fill="#00e676"><animate attributeName="opacity" values="1;1;0.1;1;1" dur="3s" repeatCount="indefinite"/></rect><rect x="8.5" y="2.5" width="1.2" height="1.2" fill="#00e676"><animate attributeName="opacity" values="1;0.1;1;1;1" dur="2.3s" repeatCount="indefinite"/></rect><rect x="15.5" y="2.5" width="1.2" height="1.2" fill="#2a2a2a"/><rect x="22.5" y="2.5" width="1.2" height="1.2" fill="#00e676"><animate attributeName="opacity" values="0.1;1;1;1;0.1" dur="1.9s" repeatCount="indefinite"/></rect><circle cx="31" cy="5.5" r="1.4" fill="#00e676"><animate attributeName="opacity" values="1;0.3;1" dur="4s" repeatCount="indefinite"/></circle><circle cx="35.5" cy="5.5" r="1.1" fill="#29b6f6"/><circle cx="44" cy="5.5" r="3" fill="#181818" stroke="#2d2d2d" stroke-width="0.5"/><circle cx="44" cy="5.5" r="1.4" fill="#0277bd"><animate attributeName="opacity" values="1;0.5;1" dur="3.5s" repeatCount="indefinite"/></circle></svg>
  <svg width="52" height="11" viewBox="0 0 52 11"><rect width="52" height="11" fill="#1e1e1e"/><rect width="52" height="0.5" fill="#353535"/><rect y="10.5" width="52" height="0.5" fill="#0d0d0d"/><rect x="1" y="2" width="5.5" height="7" fill="#141414" stroke="#272727" stroke-width="0.5" rx="0.3"/><rect x="8" y="2" width="5.5" height="7" fill="#141414" stroke="#272727" stroke-width="0.5" rx="0.3"/><rect x="15" y="2" width="5.5" height="7" fill="#141414" stroke="#272727" stroke-width="0.5" rx="0.3"/><rect x="22" y="2" width="5.5" height="7" fill="#141414" stroke="#272727" stroke-width="0.5" rx="0.3"/><rect x="1.5" y="2.5" width="1.2" height="1.2" fill="#2a2a2a"/><rect x="8.5" y="2.5" width="1.2" height="1.2" fill="#00e676"><animate attributeName="opacity" values="1;1;1;0.1;1" dur="2.7s" repeatCount="indefinite"/></rect><rect x="15.5" y="2.5" width="1.2" height="1.2" fill="#00e676"><animate attributeName="opacity" values="0.1;1;1;0.1;1" dur="2.1s" repeatCount="indefinite"/></rect><rect x="22.5" y="2.5" width="1.2" height="1.2" fill="#2a2a2a"/><circle cx="31" cy="5.5" r="1.4" fill="#00e676"><animate attributeName="opacity" values="0.4;1;0.4" dur="3.5s" repeatCount="indefinite"/></circle><circle cx="35.5" cy="5.5" r="1.1" fill="#29b6f6"/><circle cx="44" cy="5.5" r="3" fill="#181818" stroke="#2d2d2d" stroke-width="0.5"/><circle cx="44" cy="5.5" r="1.4" fill="#0277bd"><animate attributeName="opacity" values="1;0.4;1" dur="4s" repeatCount="indefinite"/></circle></svg>
  <svg width="52" height="11" viewBox="0 0 52 11"><rect width="52" height="11" fill="#1e1e1e"/><rect width="52" height="0.5" fill="#353535"/><rect y="10.5" width="52" height="0.5" fill="#0d0d0d"/><rect x="1" y="2" width="5.5" height="7" fill="#141414" stroke="#272727" stroke-width="0.5" rx="0.3"/><rect x="8" y="2" width="5.5" height="7" fill="#141414" stroke="#272727" stroke-width="0.5" rx="0.3"/><rect x="15" y="2" width="5.5" height="7" fill="#141414" stroke="#272727" stroke-width="0.5" rx="0.3"/><rect x="22" y="2" width="5.5" height="7" fill="#141414" stroke="#272727" stroke-width="0.5" rx="0.3"/><rect x="1.5" y="2.5" width="1.2" height="1.2" fill="#00e676"><animate attributeName="opacity" values="1;0.1;1;1" dur="1.5s" repeatCount="indefinite"/></rect><rect x="8.5" y="2.5" width="1.2" height="1.2" fill="#00e676"><animate attributeName="opacity" values="1;1;0.1;1" dur="2.8s" repeatCount="indefinite"/></rect><rect x="15.5" y="2.5" width="1.2" height="1.2" fill="#00e676"><animate attributeName="opacity" values="1;1;1;0.1" dur="2s" repeatCount="indefinite"/></rect><rect x="22.5" y="2.5" width="1.2" height="1.2" fill="#00e676"><animate attributeName="opacity" values="0.1;1;1;1" dur="2.5s" repeatCount="indefinite"/></rect><circle cx="31" cy="5.5" r="1.4" fill="#00e676"><animate attributeName="opacity" values="1;0.4;1" dur="3s" repeatCount="indefinite"/></circle><circle cx="35.5" cy="5.5" r="1.1" fill="#29b6f6"/><circle cx="44" cy="5.5" r="3" fill="#181818" stroke="#2d2d2d" stroke-width="0.5"/><circle cx="44" cy="5.5" r="1.4" fill="#0277bd"><animate attributeName="opacity" values="0.7;1;0.7" dur="2.5s" repeatCount="indefinite"/></circle></svg>
  <svg width="52" height="11" viewBox="0 0 52 11"><rect width="52" height="11" fill="#1e1e1e"/><rect width="52" height="0.5" fill="#353535"/><rect y="10.5" width="52" height="0.5" fill="#0d0d0d"/><rect x="1" y="2" width="5.5" height="7" fill="#141414" stroke="#272727" stroke-width="0.5" rx="0.3"/><rect x="8" y="2" width="5.5" height="7" fill="#141414" stroke="#272727" stroke-width="0.5" rx="0.3"/><rect x="15" y="2" width="5.5" height="7" fill="#141414" stroke="#272727" stroke-width="0.5" rx="0.3"/><rect x="22" y="2" width="5.5" height="7" fill="#141414" stroke="#272727" stroke-width="0.5" rx="0.3"/><rect x="1.5" y="2.5" width="1.2" height="1.2" fill="#00e676"><animate attributeName="opacity" values="1;1;0.1;1;1" dur="3.2s" repeatCount="indefinite"/></rect><rect x="8.5" y="2.5" width="1.2" height="1.2" fill="#2a2a2a"/><rect x="15.5" y="2.5" width="1.2" height="1.2" fill="#00e676"><animate attributeName="opacity" values="1;0.1;1;1" dur="2.6s" repeatCount="indefinite"/></rect><rect x="22.5" y="2.5" width="1.2" height="1.2" fill="#00e676"><animate attributeName="opacity" values="0.1;0.1;1;1" dur="1.7s" repeatCount="indefinite"/></rect><circle cx="31" cy="5.5" r="1.4" fill="#00e676"><animate attributeName="opacity" values="1;0.3;1" dur="3.8s" repeatCount="indefinite"/></circle><circle cx="35.5" cy="5.5" r="1.1" fill="#29b6f6"/><circle cx="44" cy="5.5" r="3" fill="#181818" stroke="#2d2d2d" stroke-width="0.5"/><circle cx="44" cy="5.5" r="1.4" fill="#0277bd"><animate attributeName="opacity" values="1;0.6;1" dur="3s" repeatCount="indefinite"/></circle></svg>
  <svg width="52" height="11" viewBox="0 0 52 11"><rect width="52" height="11" fill="#1e1e1e"/><rect width="52" height="0.5" fill="#353535"/><rect y="10.5" width="52" height="0.5" fill="#0d0d0d"/><rect x="1" y="2" width="5.5" height="7" fill="#141414" stroke="#272727" stroke-width="0.5" rx="0.3"/><rect x="8" y="2" width="5.5" height="7" fill="#141414" stroke="#272727" stroke-width="0.5" rx="0.3"/><rect x="15" y="2" width="5.5" height="7" fill="#141414" stroke="#272727" stroke-width="0.5" rx="0.3"/><rect x="22" y="2" width="5.5" height="7" fill="#141414" stroke="#272727" stroke-width="0.5" rx="0.3"/><rect x="1.5" y="2.5" width="1.2" height="1.2" fill="#00e676"><animate attributeName="opacity" values="1;0.1;1;1;1" dur="2.4s" repeatCount="indefinite"/></rect><rect x="8.5" y="2.5" width="1.2" height="1.2" fill="#00e676"><animate attributeName="opacity" values="1;1;0.1;1;1" dur="3.1s" repeatCount="indefinite"/></rect><rect x="15.5" y="2.5" width="1.2" height="1.2" fill="#2a2a2a"/><rect x="22.5" y="2.5" width="1.2" height="1.2" fill="#00e676"><animate attributeName="opacity" values="1;1;1;0.1;1" dur="2.2s" repeatCount="indefinite"/></rect><circle cx="31" cy="5.5" r="1.4" fill="#00e676"><animate attributeName="opacity" values="0.5;1;0.5" dur="4.2s" repeatCount="indefinite"/></circle><circle cx="35.5" cy="5.5" r="1.1" fill="#29b6f6"/><circle cx="44" cy="5.5" r="3" fill="#181818" stroke="#2d2d2d" stroke-width="0.5"/><circle cx="44" cy="5.5" r="1.4" fill="#0277bd"><animate attributeName="opacity" values="1;0.5;1" dur="2.8s" repeatCount="indefinite"/></circle></svg>
  <svg width="52" height="11" viewBox="0 0 52 11"><rect width="52" height="11" fill="#1e1e1e"/><rect width="52" height="0.5" fill="#353535"/><rect y="10.5" width="52" height="0.5" fill="#0d0d0d"/><rect x="1" y="2" width="5.5" height="7" fill="#141414" stroke="#272727" stroke-width="0.5" rx="0.3"/><rect x="8" y="2" width="5.5" height="7" fill="#141414" stroke="#272727" stroke-width="0.5" rx="0.3"/><rect x="15" y="2" width="5.5" height="7" fill="#141414" stroke="#272727" stroke-width="0.5" rx="0.3"/><rect x="22" y="2" width="5.5" height="7" fill="#141414" stroke="#272727" stroke-width="0.5" rx="0.3"/><rect x="1.5" y="2.5" width="1.2" height="1.2" fill="#2a2a2a"/><rect x="8.5" y="2.5" width="1.2" height="1.2" fill="#00e676"><animate attributeName="opacity" values="0.1;1;1;1;0.1" dur="2s" repeatCount="indefinite"/></rect><rect x="15.5" y="2.5" width="1.2" height="1.2" fill="#00e676"><animate attributeName="opacity" values="1;1;0.1;1;1" dur="3.4s" repeatCount="indefinite"/></rect><rect x="22.5" y="2.5" width="1.2" height="1.2" fill="#00e676"><animate attributeName="opacity" values="1;0.1;1;1;1" dur="1.6s" repeatCount="indefinite"/></rect><circle cx="31" cy="5.5" r="1.4" fill="#00e676"><animate attributeName="opacity" values="1;0.4;1" dur="3.3s" repeatCount="indefinite"/></circle><circle cx="35.5" cy="5.5" r="1.1" fill="#29b6f6"/><circle cx="44" cy="5.5" r="3" fill="#181818" stroke="#2d2d2d" stroke-width="0.5"/><circle cx="44" cy="5.5" r="1.4" fill="#0277bd"><animate attributeName="opacity" values="0.8;1;0.8" dur="3.2s" repeatCount="indefinite"/></circle></svg>
  </div>
  <div class="rs"></div>
  </div>
  <div class="rack3d">
  <div class="rt" style="width:60px"></div>
  <div class="rf" style="width:60px">
  <svg width="52" height="11" viewBox="0 0 52 11"><rect width="52" height="11" fill="#1e1e1e"/><rect width="52" height="0.5" fill="#353535"/><rect y="10.5" width="52" height="0.5" fill="#0d0d0d"/><rect x="1" y="2" width="5.5" height="7" fill="#141414" stroke="#272727" stroke-width="0.5" rx="0.3"/><rect x="8" y="2" width="5.5" height="7" fill="#141414" stroke="#272727" stroke-width="0.5" rx="0.3"/><rect x="15" y="2" width="5.5" height="7" fill="#141414" stroke="#272727" stroke-width="0.5" rx="0.3"/><rect x="22" y="2" width="5.5" height="7" fill="#141414" stroke="#272727" stroke-width="0.5" rx="0.3"/><rect x="1.5" y="2.5" width="1.2" height="1.2" fill="#2a2a2a"/><rect x="8.5" y="2.5" width="1.2" height="1.2" fill="#00e676"><animate attributeName="opacity" values="1;1;0.1;1" dur="2.9s" repeatCount="indefinite"/></rect><rect x="15.5" y="2.5" width="1.2" height="1.2" fill="#00e676"><animate attributeName="opacity" values="0.1;1;1;0.1" dur="2.2s" repeatCount="indefinite"/></rect><rect x="22.5" y="2.5" width="1.2" height="1.2" fill="#2a2a2a"/><circle cx="31" cy="5.5" r="1.4" fill="#00e676"><animate attributeName="opacity" values="1;0.3;1" dur="3.7s" repeatCount="indefinite"/></circle><circle cx="35.5" cy="5.5" r="1.1" fill="#29b6f6"/><circle cx="44" cy="5.5" r="3" fill="#181818" stroke="#2d2d2d" stroke-width="0.5"/><circle cx="44" cy="5.5" r="1.4" fill="#0277bd"><animate attributeName="opacity" values="1;0.5;1" dur="3.1s" repeatCount="indefinite"/></circle></svg>
  <svg width="52" height="11" viewBox="0 0 52 11"><rect width="52" height="11" fill="#1e1e1e"/><rect width="52" height="0.5" fill="#353535"/><rect y="10.5" width="52" height="0.5" fill="#0d0d0d"/><rect x="1" y="2" width="5.5" height="7" fill="#141414" stroke="#272727" stroke-width="0.5" rx="0.3"/><rect x="8" y="2" width="5.5" height="7" fill="#141414" stroke="#272727" stroke-width="0.5" rx="0.3"/><rect x="15" y="2" width="5.5" height="7" fill="#141414" stroke="#272727" stroke-width="0.5" rx="0.3"/><rect x="22" y="2" width="5.5" height="7" fill="#141414" stroke="#272727" stroke-width="0.5" rx="0.3"/><rect x="1.5" y="2.5" width="1.2" height="1.2" fill="#00e676"><animate attributeName="opacity" values="1;1;0.1;1;1" dur="3s" repeatCount="indefinite"/></rect><rect x="8.5" y="2.5" width="1.2" height="1.2" fill="#2a2a2a"/><rect x="15.5" y="2.5" width="1.2" height="1.2" fill="#00e676"><animate attributeName="opacity" values="0.1;1;1;1;0.1" dur="2.4s" repeatCount="indefinite"/></rect><rect x="22.5" y="2.5" width="1.2" height="1.2" fill="#00e676"><animate attributeName="opacity" values="1;0.1;1;1;1" dur="1.8s" repeatCount="indefinite"/></rect><circle cx="31" cy="5.5" r="1.4" fill="#00e676"><animate attributeName="opacity" values="0.5;1;0.5" dur="4s" repeatCount="indefinite"/></circle><circle cx="35.5" cy="5.5" r="1.1" fill="#29b6f6"/><circle cx="44" cy="5.5" r="3" fill="#181818" stroke="#2d2d2d" stroke-width="0.5"/><circle cx="44" cy="5.5" r="1.4" fill="#0277bd"><animate attributeName="opacity" values="1;0.4;1" dur="3.8s" repeatCount="indefinite"/></circle></svg>
  <svg width="52" height="11" viewBox="0 0 52 11"><rect width="52" height="11" fill="#1e1e1e"/><rect width="52" height="0.5" fill="#353535"/><rect y="10.5" width="52" height="0.5" fill="#0d0d0d"/><rect x="1" y="2" width="5.5" height="7" fill="#141414" stroke="#272727" stroke-width="0.5" rx="0.3"/><rect x="8" y="2" width="5.5" height="7" fill="#141414" stroke="#272727" stroke-width="0.5" rx="0.3"/><rect x="15" y="2" width="5.5" height="7" fill="#141414" stroke="#272727" stroke-width="0.5" rx="0.3"/><rect x="22" y="2" width="5.5" height="7" fill="#141414" stroke="#272727" stroke-width="0.5" rx="0.3"/><rect x="1.5" y="2.5" width="1.2" height="1.2" fill="#00e676"><animate attributeName="opacity" values="1;0.1;1;1" dur="2s" repeatCount="indefinite"/></rect><rect x="8.5" y="2.5" width="1.2" height="1.2" fill="#00e676"><animate attributeName="opacity" values="1;1;0.1;1" dur="3.2s" repeatCount="indefinite"/></rect><rect x="15.5" y="2.5" width="1.2" height="1.2" fill="#00e676"><animate attributeName="opacity" values="1;1;1;0.1" dur="2.5s" repeatCount="indefinite"/></rect><rect x="22.5" y="2.5" width="1.2" height="1.2" fill="#00e676"><animate attributeName="opacity" values="0.1;1;1;1" dur="1.9s" repeatCount="indefinite"/></rect><circle cx="31" cy="5.5" r="1.4" fill="#00e676"><animate attributeName="opacity" values="1;0.3;1" dur="3.5s" repeatCount="indefinite"/></circle><circle cx="35.5" cy="5.5" r="1.1" fill="#29b6f6"/><circle cx="44" cy="5.5" r="3" fill="#181818" stroke="#2d2d2d" stroke-width="0.5"/><circle cx="44" cy="5.5" r="1.4" fill="#0277bd"><animate attributeName="opacity" values="0.8;1;0.8" dur="2.7s" repeatCount="indefinite"/></circle></svg>
  <svg width="52" height="11" viewBox="0 0 52 11"><rect width="52" height="11" fill="#1e1e1e"/><rect width="52" height="0.5" fill="#353535"/><rect y="10.5" width="52" height="0.5" fill="#0d0d0d"/><rect x="1" y="2" width="5.5" height="7" fill="#141414" stroke="#272727" stroke-width="0.5" rx="0.3"/><rect x="8" y="2" width="5.5" height="7" fill="#141414" stroke="#272727" stroke-width="0.5" rx="0.3"/><rect x="15" y="2" width="5.5" height="7" fill="#141414" stroke="#272727" stroke-width="0.5" rx="0.3"/><rect x="22" y="2" width="5.5" height="7" fill="#141414" stroke="#272727" stroke-width="0.5" rx="0.3"/><rect x="1.5" y="2.5" width="1.2" height="1.2" fill="#2a2a2a"/><rect x="8.5" y="2.5" width="1.2" height="1.2" fill="#00e676"><animate attributeName="opacity" values="1;0.1;1;1" dur="2.3s" repeatCount="indefinite"/></rect><rect x="15.5" y="2.5" width="1.2" height="1.2" fill="#2a2a2a"/><rect x="22.5" y="2.5" width="1.2" height="1.2" fill="#00e676"><animate attributeName="opacity" values="0.1;1;0.1;1" dur="1.4s" repeatCount="indefinite"/></rect><circle cx="31" cy="5.5" r="1.4" fill="#00e676"><animate attributeName="opacity" values="1;0.4;1" dur="4.5s" repeatCount="indefinite"/></circle><circle cx="35.5" cy="5.5" r="1.1" fill="#29b6f6"/><circle cx="44" cy="5.5" r="3" fill="#181818" stroke="#2d2d2d" stroke-width="0.5"/><circle cx="44" cy="5.5" r="1.4" fill="#0277bd"><animate attributeName="opacity" values="1;0.6;1" dur="3.3s" repeatCount="indefinite"/></circle></svg>
  <svg width="52" height="11" viewBox="0 0 52 11"><rect width="52" height="11" fill="#1e1e1e"/><rect width="52" height="0.5" fill="#353535"/><rect y="10.5" width="52" height="0.5" fill="#0d0d0d"/><rect x="1" y="2" width="5.5" height="7" fill="#141414" stroke="#272727" stroke-width="0.5" rx="0.3"/><rect x="8" y="2" width="5.5" height="7" fill="#141414" stroke="#272727" stroke-width="0.5" rx="0.3"/><rect x="15" y="2" width="5.5" height="7" fill="#141414" stroke="#272727" stroke-width="0.5" rx="0.3"/><rect x="22" y="2" width="5.5" height="7" fill="#141414" stroke="#272727" stroke-width="0.5" rx="0.3"/><rect x="1.5" y="2.5" width="1.2" height="1.2" fill="#00e676"><animate attributeName="opacity" values="1;1;0.1;1" dur="2.6s" repeatCount="indefinite"/></rect><rect x="8.5" y="2.5" width="1.2" height="1.2" fill="#00e676"><animate attributeName="opacity" values="1;0.1;1;1" dur="3s" repeatCount="indefinite"/></rect><rect x="15.5" y="2.5" width="1.2" height="1.2" fill="#00e676"><animate attributeName="opacity" values="0.1;1;1;1" dur="2.2s" repeatCount="indefinite"/></rect><rect x="22.5" y="2.5" width="1.2" height="1.2" fill="#2a2a2a"/><circle cx="31" cy="5.5" r="1.4" fill="#00e676"><animate attributeName="opacity" values="0.3;1;0.3" dur="3.9s" repeatCount="indefinite"/></circle><circle cx="35.5" cy="5.5" r="1.1" fill="#29b6f6"/><circle cx="44" cy="5.5" r="3" fill="#181818" stroke="#2d2d2d" stroke-width="0.5"/><circle cx="44" cy="5.5" r="1.4" fill="#0277bd"><animate attributeName="opacity" values="1;0.5;1" dur="3.6s" repeatCount="indefinite"/></circle></svg>
  <svg width="52" height="11" viewBox="0 0 52 11"><rect width="52" height="11" fill="#1e1e1e"/><rect width="52" height="0.5" fill="#353535"/><rect y="10.5" width="52" height="0.5" fill="#0d0d0d"/><rect x="1" y="2" width="5.5" height="7" fill="#141414" stroke="#272727" stroke-width="0.5" rx="0.3"/><rect x="8" y="2" width="5.5" height="7" fill="#141414" stroke="#272727" stroke-width="0.5" rx="0.3"/><rect x="15" y="2" width="5.5" height="7" fill="#141414" stroke="#272727" stroke-width="0.5" rx="0.3"/><rect x="22" y="2" width="5.5" height="7" fill="#141414" stroke="#272727" stroke-width="0.5" rx="0.3"/><rect x="1.5" y="2.5" width="1.2" height="1.2" fill="#00e676"><animate attributeName="opacity" values="0.1;1;1;1;0.1" dur="1.7s" repeatCount="indefinite"/></rect><rect x="8.5" y="2.5" width="1.2" height="1.2" fill="#2a2a2a"/><rect x="15.5" y="2.5" width="1.2" height="1.2" fill="#00e676"><animate attributeName="opacity" values="1;0.1;1;1;1" dur="2.9s" repeatCount="indefinite"/></rect><rect x="22.5" y="2.5" width="1.2" height="1.2" fill="#00e676"><animate attributeName="opacity" values="1;1;0.1;1;1" dur="2.1s" repeatCount="indefinite"/></rect><circle cx="31" cy="5.5" r="1.4" fill="#00e676"><animate attributeName="opacity" values="1;0.3;1" dur="4.1s" repeatCount="indefinite"/></circle><circle cx="35.5" cy="5.5" r="1.1" fill="#29b6f6"/><circle cx="44" cy="5.5" r="3" fill="#181818" stroke="#2d2d2d" stroke-width="0.5"/><circle cx="44" cy="5.5" r="1.4" fill="#0277bd"><animate attributeName="opacity" values="0.9;1;0.9" dur="2.9s" repeatCount="indefinite"/></circle></svg>
  </div>
  <div class="rs"></div>
  </div>
  <div class="rack3d">
  <div class="rt" style="width:78px"></div>
  <div class="rf" style="width:78px">
  <svg width="70" height="11" viewBox="0 0 70 11"><rect width="70" height="11" fill="#1e1e1e"/><rect width="70" height="0.5" fill="#353535"/><rect y="10.5" width="70" height="0.5" fill="#0d0d0d"/><rect x="1" y="2" width="5.5" height="7" fill="#141414" stroke="#272727" stroke-width="0.5" rx="0.3"/><rect x="8" y="2" width="5.5" height="7" fill="#141414" stroke="#272727" stroke-width="0.5" rx="0.3"/><rect x="1.5" y="2.5" width="1.2" height="1.2" fill="#00e676"><animate attributeName="opacity" values="1;1;0.1;1" dur="2.5s" repeatCount="indefinite"/></rect><rect x="8.5" y="2.5" width="1.2" height="1.2" fill="#00e676"><animate attributeName="opacity" values="0.1;1;1;1" dur="1.8s" repeatCount="indefinite"/></rect><rect x="16" y="2" width="28" height="7" fill="#0a1428" stroke="#1a3060" stroke-width="0.5" rx="1"/><rect x="17" y="3" width="26" height="5" fill="#0f2045"/><rect x="17" y="4.5" width="26" height="2" fill="#1a3870"><animate attributeName="fill" values="#0d2048;#29b6f6;#0d2048" dur="2.2s" repeatCount="indefinite"/></rect><circle cx="48.5" cy="5.5" r="1.4" fill="#00e676"><animate attributeName="opacity" values="1;0.3;1" dur="3.5s" repeatCount="indefinite"/></circle><circle cx="53" cy="5.5" r="1.1" fill="#ff9100"/><circle cx="62" cy="5.5" r="3" fill="#181818" stroke="#2d2d2d" stroke-width="0.5"/><circle cx="62" cy="5.5" r="1.4" fill="#0277bd"><animate attributeName="opacity" values="1;0.5;1" dur="3s" repeatCount="indefinite"/></circle></svg>
  <svg width="70" height="11" viewBox="0 0 70 11"><rect width="70" height="11" fill="#1e1e1e"/><rect width="70" height="0.5" fill="#353535"/><rect y="10.5" width="70" height="0.5" fill="#0d0d0d"/><rect x="1" y="2" width="5.5" height="7" fill="#141414" stroke="#272727" stroke-width="0.5" rx="0.3"/><rect x="8" y="2" width="5.5" height="7" fill="#141414" stroke="#272727" stroke-width="0.5" rx="0.3"/><rect x="1.5" y="2.5" width="1.2" height="1.2" fill="#00e676"><animate attributeName="opacity" values="1;0.1;1;1" dur="3s" repeatCount="indefinite"/></rect><rect x="8.5" y="2.5" width="1.2" height="1.2" fill="#2a2a2a"/><rect x="16" y="2" width="28" height="7" fill="#0a1428" stroke="#1a3060" stroke-width="0.5" rx="1"/><rect x="17" y="3" width="26" height="5" fill="#0f2045"/><rect x="17" y="4.5" width="26" height="2" fill="#1a3870"><animate attributeName="fill" values="#0d2048;#1a4090;#29b6f6;#1a4090;#0d2048" dur="3s" repeatCount="indefinite"/></rect><circle cx="48.5" cy="5.5" r="1.4" fill="#00e676"><animate attributeName="opacity" values="0.4;1;0.4" dur="4s" repeatCount="indefinite"/></circle><circle cx="53" cy="5.5" r="1.1" fill="#29b6f6"/><circle cx="62" cy="5.5" r="3" fill="#181818" stroke="#2d2d2d" stroke-width="0.5"/><circle cx="62" cy="5.5" r="1.4" fill="#0277bd"><animate attributeName="opacity" values="0.6;1;0.6" dur="2.5s" repeatCount="indefinite"/></circle></svg>
  <svg width="70" height="11" viewBox="0 0 70 11"><rect width="70" height="11" fill="#1e1e1e"/><rect width="70" height="0.5" fill="#353535"/><rect y="10.5" width="70" height="0.5" fill="#0d0d0d"/><rect x="1" y="2" width="5.5" height="7" fill="#141414" stroke="#272727" stroke-width="0.5" rx="0.3"/><rect x="8" y="2" width="5.5" height="7" fill="#141414" stroke="#272727" stroke-width="0.5" rx="0.3"/><rect x="1.5" y="2.5" width="1.2" height="1.2" fill="#00e676"><animate attributeName="opacity" values="0.1;1;1;1" dur="2.1s" repeatCount="indefinite"/></rect><rect x="8.5" y="2.5" width="1.2" height="1.2" fill="#00e676"><animate attributeName="opacity" values="1;1;0.1;1" dur="2.7s" repeatCount="indefinite"/></rect><rect x="16" y="2" width="28" height="7" fill="#0a1428" stroke="#1a3060" stroke-width="0.5" rx="1"/><rect x="17" y="3" width="26" height="5" fill="#0f2045"/><rect x="17" y="4.5" width="26" height="2" fill="#1a3870"><animate attributeName="fill" values="#29b6f6;#0d2048;#29b6f6" dur="1.8s" repeatCount="indefinite"/></rect><circle cx="48.5" cy="5.5" r="1.4" fill="#00e676"><animate attributeName="opacity" values="1;0.3;1" dur="3.2s" repeatCount="indefinite"/></circle><circle cx="53" cy="5.5" r="1.1" fill="#ff9100"/><circle cx="62" cy="5.5" r="3" fill="#181818" stroke="#2d2d2d" stroke-width="0.5"/><circle cx="62" cy="5.5" r="1.4" fill="#0277bd"><animate attributeName="opacity" values="1;0.4;1" dur="3.8s" repeatCount="indefinite"/></circle></svg>
  <svg width="70" height="11" viewBox="0 0 70 11"><rect width="70" height="11" fill="#1e1e1e"/><rect width="70" height="0.5" fill="#353535"/><rect y="10.5" width="70" height="0.5" fill="#0d0d0d"/><rect x="1" y="2" width="5.5" height="7" fill="#141414" stroke="#272727" stroke-width="0.5" rx="0.3"/><rect x="8" y="2" width="5.5" height="7" fill="#141414" stroke="#272727" stroke-width="0.5" rx="0.3"/><rect x="1.5" y="2.5" width="1.2" height="1.2" fill="#2a2a2a"/><rect x="8.5" y="2.5" width="1.2" height="1.2" fill="#00e676"><animate attributeName="opacity" values="1;0.1;1;1" dur="2.3s" repeatCount="indefinite"/></rect><rect x="16" y="2" width="28" height="7" fill="#0a1428" stroke="#1a3060" stroke-width="0.5" rx="1"/><rect x="17" y="3" width="26" height="5" fill="#0f2045"/><rect x="17" y="4.5" width="26" height="2" fill="#1a3870"><animate attributeName="fill" values="#0d2048;#29b6f6;#1a4090;#0d2048" dur="2.5s" repeatCount="indefinite"/></rect><circle cx="48.5" cy="5.5" r="1.4" fill="#00e676"><animate attributeName="opacity" values="0.5;1;0.5" dur="4.3s" repeatCount="indefinite"/></circle><circle cx="53" cy="5.5" r="1.1" fill="#29b6f6"/><circle cx="62" cy="5.5" r="3" fill="#181818" stroke="#2d2d2d" stroke-width="0.5"/><circle cx="62" cy="5.5" r="1.4" fill="#0277bd"><animate attributeName="opacity" values="1;0.5;1" dur="2.8s" repeatCount="indefinite"/></circle></svg>
  <svg width="70" height="11" viewBox="0 0 70 11"><rect width="70" height="11" fill="#1e1e1e"/><rect width="70" height="0.5" fill="#353535"/><rect y="10.5" width="70" height="0.5" fill="#0d0d0d"/><rect x="1" y="2" width="5.5" height="7" fill="#141414" stroke="#272727" stroke-width="0.5" rx="0.3"/><rect x="8" y="2" width="5.5" height="7" fill="#141414" stroke="#272727" stroke-width="0.5" rx="0.3"/><rect x="1.5" y="2.5" width="1.2" height="1.2" fill="#00e676"><animate attributeName="opacity" values="1;1;0.1;1" dur="1.9s" repeatCount="indefinite"/></rect><rect x="8.5" y="2.5" width="1.2" height="1.2" fill="#00e676"><animate attributeName="opacity" values="0.1;1;1;0.1" dur="2.6s" repeatCount="indefinite"/></rect><rect x="16" y="2" width="28" height="7" fill="#0a1428" stroke="#1a3060" stroke-width="0.5" rx="1"/><rect x="17" y="3" width="26" height="5" fill="#0f2045"/><rect x="17" y="4.5" width="26" height="2" fill="#1a3870"><animate attributeName="fill" values="#1a4090;#29b6f6;#0d2048;#1a4090" dur="2s" repeatCount="indefinite"/></rect><circle cx="48.5" cy="5.5" r="1.4" fill="#00e676"><animate attributeName="opacity" values="1;0.4;1" dur="3.6s" repeatCount="indefinite"/></circle><circle cx="53" cy="5.5" r="1.1" fill="#ff9100"/><circle cx="62" cy="5.5" r="3" fill="#181818" stroke="#2d2d2d" stroke-width="0.5"/><circle cx="62" cy="5.5" r="1.4" fill="#0277bd"><animate attributeName="opacity" values="0.7;1;0.7" dur="3.4s" repeatCount="indefinite"/></circle></svg>
  <svg width="70" height="11" viewBox="0 0 70 11"><rect width="70" height="11" fill="#1e1e1e"/><rect width="70" height="0.5" fill="#353535"/><rect y="10.5" width="70" height="0.5" fill="#0d0d0d"/><rect x="1" y="2" width="5.5" height="7" fill="#141414" stroke="#272727" stroke-width="0.5" rx="0.3"/><rect x="8" y="2" width="5.5" height="7" fill="#141414" stroke="#272727" stroke-width="0.5" rx="0.3"/><rect x="1.5" y="2.5" width="1.2" height="1.2" fill="#00e676"><animate attributeName="opacity" values="1;0.1;1;1;1" dur="2.4s" repeatCount="indefinite"/></rect><rect x="8.5" y="2.5" width="1.2" height="1.2" fill="#00e676"><animate attributeName="opacity" values="1;1;1;0.1;1" dur="3.1s" repeatCount="indefinite"/></rect><rect x="16" y="2" width="28" height="7" fill="#0a1428" stroke="#1a3060" stroke-width="0.5" rx="1"/><rect x="17" y="3" width="26" height="5" fill="#0f2045"/><rect x="17" y="4.5" width="26" height="2" fill="#1a3870"><animate attributeName="fill" values="#0d2048;#1a4090;#29b6f6;#1a4090;#0d2048" dur="2.8s" repeatCount="indefinite"/></rect><circle cx="48.5" cy="5.5" r="1.4" fill="#00e676"><animate attributeName="opacity" values="0.3;1;0.3" dur="4.2s" repeatCount="indefinite"/></circle><circle cx="53" cy="5.5" r="1.1" fill="#29b6f6"/><circle cx="62" cy="5.5" r="3" fill="#181818" stroke="#2d2d2d" stroke-width="0.5"/><circle cx="62" cy="5.5" r="1.4" fill="#0277bd"><animate attributeName="opacity" values="1;0.6;1" dur="2.6s" repeatCount="indefinite"/></circle></svg>
  <svg width="70" height="11" viewBox="0 0 70 11"><rect width="70" height="11" fill="#1e1e1e"/><rect width="70" height="0.5" fill="#353535"/><rect y="10.5" width="70" height="0.5" fill="#0d0d0d"/><rect x="1" y="2" width="5.5" height="7" fill="#141414" stroke="#272727" stroke-width="0.5" rx="0.3"/><rect x="8" y="2" width="5.5" height="7" fill="#141414" stroke="#272727" stroke-width="0.5" rx="0.3"/><rect x="1.5" y="2.5" width="1.2" height="1.2" fill="#2a2a2a"/><rect x="8.5" y="2.5" width="1.2" height="1.2" fill="#00e676"><animate attributeName="opacity" values="1;1;0.1;1" dur="2.2s" repeatCount="indefinite"/></rect><rect x="16" y="2" width="28" height="7" fill="#0a1428" stroke="#1a3060" stroke-width="0.5" rx="1"/><rect x="17" y="3" width="26" height="5" fill="#0f2045"/><rect x="17" y="4.5" width="26" height="2" fill="#1a3870"><animate attributeName="fill" values="#29b6f6;#1a4090;#0d2048;#29b6f6" dur="1.6s" repeatCount="indefinite"/></rect><circle cx="48.5" cy="5.5" r="1.4" fill="#00e676"><animate attributeName="opacity" values="1;0.3;1" dur="3.7s" repeatCount="indefinite"/></circle><circle cx="53" cy="5.5" r="1.1" fill="#ff9100"/><circle cx="62" cy="5.5" r="3" fill="#181818" stroke="#2d2d2d" stroke-width="0.5"/><circle cx="62" cy="5.5" r="1.4" fill="#0277bd"><animate attributeName="opacity" values="1;0.5;1" dur="3.2s" repeatCount="indefinite"/></circle></svg>
  <svg width="70" height="11" viewBox="0 0 70 11"><rect width="70" height="11" fill="#1e1e1e"/><rect width="70" height="0.5" fill="#353535"/><rect y="10.5" width="70" height="0.5" fill="#0d0d0d"/><rect x="1" y="2" width="5.5" height="7" fill="#141414" stroke="#272727" stroke-width="0.5" rx="0.3"/><rect x="8" y="2" width="5.5" height="7" fill="#141414" stroke="#272727" stroke-width="0.5" rx="0.3"/><rect x="1.5" y="2.5" width="1.2" height="1.2" fill="#00e676"><animate attributeName="opacity" values="1;0.1;1;1" dur="2.8s" repeatCount="indefinite"/></rect><rect x="8.5" y="2.5" width="1.2" height="1.2" fill="#00e676"><animate attributeName="opacity" values="0.1;1;1;0.1" dur="2s" repeatCount="indefinite"/></rect><rect x="16" y="2" width="28" height="7" fill="#0a1428" stroke="#1a3060" stroke-width="0.5" rx="1"/><rect x="17" y="3" width="26" height="5" fill="#0f2045"/><rect x="17" y="4.5" width="26" height="2" fill="#1a3870"><animate attributeName="fill" values="#0d2048;#29b6f6;#0d2048" dur="2.4s" repeatCount="indefinite"/></rect><circle cx="48.5" cy="5.5" r="1.4" fill="#00e676"><animate attributeName="opacity" values="0.4;1;0.4" dur="4.4s" repeatCount="indefinite"/></circle><circle cx="53" cy="5.5" r="1.1" fill="#29b6f6"/><circle cx="62" cy="5.5" r="3" fill="#181818" stroke="#2d2d2d" stroke-width="0.5"/><circle cx="62" cy="5.5" r="1.4" fill="#0277bd"><animate attributeName="opacity" values="0.8;1;0.8" dur="2.3s" repeatCount="indefinite"/></circle></svg>
  </div>
  <div class="rs"></div>
  </div>
  <div class="rack3d">
  <div class="rt" style="width:60px"></div>
  <div class="rf" style="width:60px">
  <svg width="52" height="11" viewBox="0 0 52 11"><rect width="52" height="11" fill="#1e1e1e"/><rect width="52" height="0.5" fill="#353535"/><rect y="10.5" width="52" height="0.5" fill="#0d0d0d"/><rect x="1" y="2" width="5.5" height="7" fill="#141414" stroke="#272727" stroke-width="0.5" rx="0.3"/><rect x="8" y="2" width="5.5" height="7" fill="#141414" stroke="#272727" stroke-width="0.5" rx="0.3"/><rect x="15" y="2" width="5.5" height="7" fill="#141414" stroke="#272727" stroke-width="0.5" rx="0.3"/><rect x="22" y="2" width="5.5" height="7" fill="#141414" stroke="#272727" stroke-width="0.5" rx="0.3"/><rect x="1.5" y="2.5" width="1.2" height="1.2" fill="#00e676"><animate attributeName="opacity" values="1;1;0.1;1" dur="2.7s" repeatCount="indefinite"/></rect><rect x="8.5" y="2.5" width="1.2" height="1.2" fill="#2a2a2a"/><rect x="15.5" y="2.5" width="1.2" height="1.2" fill="#00e676"><animate attributeName="opacity" values="1;0.1;1;1" dur="3.3s" repeatCount="indefinite"/></rect><rect x="22.5" y="2.5" width="1.2" height="1.2" fill="#00e676"><animate attributeName="opacity" values="0.1;1;1;0.1" dur="2s" repeatCount="indefinite"/></rect><circle cx="31" cy="5.5" r="1.4" fill="#00e676"><animate attributeName="opacity" values="1;0.3;1" dur="3.6s" repeatCount="indefinite"/></circle><circle cx="35.5" cy="5.5" r="1.1" fill="#29b6f6"/><circle cx="44" cy="5.5" r="3" fill="#181818" stroke="#2d2d2d" stroke-width="0.5"/><circle cx="44" cy="5.5" r="1.4" fill="#0277bd"><animate attributeName="opacity" values="1;0.4;1" dur="4.1s" repeatCount="indefinite"/></circle></svg>
  <svg width="52" height="11" viewBox="0 0 52 11"><rect width="52" height="11" fill="#1e1e1e"/><rect width="52" height="0.5" fill="#353535"/><rect y="10.5" width="52" height="0.5" fill="#0d0d0d"/><rect x="1" y="2" width="5.5" height="7" fill="#141414" stroke="#272727" stroke-width="0.5" rx="0.3"/><rect x="8" y="2" width="5.5" height="7" fill="#141414" stroke="#272727" stroke-width="0.5" rx="0.3"/><rect x="15" y="2" width="5.5" height="7" fill="#141414" stroke="#272727" stroke-width="0.5" rx="0.3"/><rect x="22" y="2" width="5.5" height="7" fill="#141414" stroke="#272727" stroke-width="0.5" rx="0.3"/><rect x="1.5" y="2.5" width="1.2" height="1.2" fill="#00e676"><animate attributeName="opacity" values="1;0.1;1;1;1" dur="2.2s" repeatCount="indefinite"/></rect><rect x="8.5" y="2.5" width="1.2" height="1.2" fill="#00e676"><animate attributeName="opacity" values="1;1;0.1;1;1" dur="3s" repeatCount="indefinite"/></rect><rect x="15.5" y="2.5" width="1.2" height="1.2" fill="#2a2a2a"/><rect x="22.5" y="2.5" width="1.2" height="1.2" fill="#00e676"><animate attributeName="opacity" values="0.1;0.1;1;1;1" dur="1.8s" repeatCount="indefinite"/></rect><circle cx="31" cy="5.5" r="1.4" fill="#00e676"><animate attributeName="opacity" values="0.5;1;0.5" dur="4s" repeatCount="indefinite"/></circle><circle cx="35.5" cy="5.5" r="1.1" fill="#29b6f6"/><circle cx="44" cy="5.5" r="3" fill="#181818" stroke="#2d2d2d" stroke-width="0.5"/><circle cx="44" cy="5.5" r="1.4" fill="#0277bd"><animate attributeName="opacity" values="1;0.6;1" dur="3.4s" repeatCount="indefinite"/></circle></svg>
  <svg width="52" height="11" viewBox="0 0 52 11"><rect width="52" height="11" fill="#1e1e1e"/><rect width="52" height="0.5" fill="#353535"/><rect y="10.5" width="52" height="0.5" fill="#0d0d0d"/><rect x="1" y="2" width="5.5" height="7" fill="#141414" stroke="#272727" stroke-width="0.5" rx="0.3"/><rect x="8" y="2" width="5.5" height="7" fill="#141414" stroke="#272727" stroke-width="0.5" rx="0.3"/><rect x="15" y="2" width="5.5" height="7" fill="#141414" stroke="#272727" stroke-width="0.5" rx="0.3"/><rect x="22" y="2" width="5.5" height="7" fill="#141414" stroke="#272727" stroke-width="0.5" rx="0.3"/><rect x="1.5" y="2.5" width="1.2" height="1.2" fill="#00e676"><animate attributeName="opacity" values="0.1;1;1;1;0.1" dur="1.6s" repeatCount="indefinite"/></rect><rect x="8.5" y="2.5" width="1.2" height="1.2" fill="#00e676"><animate attributeName="opacity" values="1;1;1;0.1;1" dur="2.8s" repeatCount="indefinite"/></rect><rect x="15.5" y="2.5" width="1.2" height="1.2" fill="#00e676"><animate attributeName="opacity" values="1;0.1;1;1;1" dur="2.3s" repeatCount="indefinite"/></rect><rect x="22.5" y="2.5" width="1.2" height="1.2" fill="#2a2a2a"/><circle cx="31" cy="5.5" r="1.4" fill="#00e676"><animate attributeName="opacity" values="1;0.4;1" dur="3.8s" repeatCount="indefinite"/></circle><circle cx="35.5" cy="5.5" r="1.1" fill="#29b6f6"/><circle cx="44" cy="5.5" r="3" fill="#181818" stroke="#2d2d2d" stroke-width="0.5"/><circle cx="44" cy="5.5" r="1.4" fill="#0277bd"><animate attributeName="opacity" values="0.9;1;0.9" dur="3s" repeatCount="indefinite"/></circle></svg>
  <svg width="52" height="11" viewBox="0 0 52 11"><rect width="52" height="11" fill="#1e1e1e"/><rect width="52" height="0.5" fill="#353535"/><rect y="10.5" width="52" height="0.5" fill="#0d0d0d"/><rect x="1" y="2" width="5.5" height="7" fill="#141414" stroke="#272727" stroke-width="0.5" rx="0.3"/><rect x="8" y="2" width="5.5" height="7" fill="#141414" stroke="#272727" stroke-width="0.5" rx="0.3"/><rect x="15" y="2" width="5.5" height="7" fill="#141414" stroke="#272727" stroke-width="0.5" rx="0.3"/><rect x="22" y="2" width="5.5" height="7" fill="#141414" stroke="#272727" stroke-width="0.5" rx="0.3"/><rect x="1.5" y="2.5" width="1.2" height="1.2" fill="#2a2a2a"/><rect x="8.5" y="2.5" width="1.2" height="1.2" fill="#00e676"><animate attributeName="opacity" values="1;0.1;1;1" dur="3.1s" repeatCount="indefinite"/></rect><rect x="15.5" y="2.5" width="1.2" height="1.2" fill="#2a2a2a"/><rect x="22.5" y="2.5" width="1.2" height="1.2" fill="#00e676"><animate attributeName="opacity" values="0.1;1;0.1;1" dur="1.5s" repeatCount="indefinite"/></rect><circle cx="31" cy="5.5" r="1.4" fill="#00e676"><animate attributeName="opacity" values="1;0.3;1" dur="4.4s" repeatCount="indefinite"/></circle><circle cx="35.5" cy="5.5" r="1.1" fill="#29b6f6"/><circle cx="44" cy="5.5" r="3" fill="#181818" stroke="#2d2d2d" stroke-width="0.5"/><circle cx="44" cy="5.5" r="1.4" fill="#0277bd"><animate attributeName="opacity" values="1;0.5;1" dur="2.7s" repeatCount="indefinite"/></circle></svg>
  <svg width="52" height="11" viewBox="0 0 52 11"><rect width="52" height="11" fill="#1e1e1e"/><rect width="52" height="0.5" fill="#353535"/><rect y="10.5" width="52" height="0.5" fill="#0d0d0d"/><rect x="1" y="2" width="5.5" height="7" fill="#141414" stroke="#272727" stroke-width="0.5" rx="0.3"/><rect x="8" y="2" width="5.5" height="7" fill="#141414" stroke="#272727" stroke-width="0.5" rx="0.3"/><rect x="15" y="2" width="5.5" height="7" fill="#141414" stroke="#272727" stroke-width="0.5" rx="0.3"/><rect x="22" y="2" width="5.5" height="7" fill="#141414" stroke="#272727" stroke-width="0.5" rx="0.3"/><rect x="1.5" y="2.5" width="1.2" height="1.2" fill="#00e676"><animate attributeName="opacity" values="1;1;0.1;1" dur="2.5s" repeatCount="indefinite"/></rect><rect x="8.5" y="2.5" width="1.2" height="1.2" fill="#00e676"><animate attributeName="opacity" values="0.1;1;1;0.1" dur="2.9s" repeatCount="indefinite"/></rect><rect x="15.5" y="2.5" width="1.2" height="1.2" fill="#00e676"><animate attributeName="opacity" values="1;0.1;1;1" dur="1.9s" repeatCount="indefinite"/></rect><rect x="22.5" y="2.5" width="1.2" height="1.2" fill="#00e676"><animate attributeName="opacity" values="1;1;0.1;1" dur="3.2s" repeatCount="indefinite"/></rect><circle cx="31" cy="5.5" r="1.4" fill="#00e676"><animate attributeName="opacity" values="0.4;1;0.4" dur="3.4s" repeatCount="indefinite"/></circle><circle cx="35.5" cy="5.5" r="1.1" fill="#29b6f6"/><circle cx="44" cy="5.5" r="3" fill="#181818" stroke="#2d2d2d" stroke-width="0.5"/><circle cx="44" cy="5.5" r="1.4" fill="#0277bd"><animate attributeName="opacity" values="0.8;1;0.8" dur="3.7s" repeatCount="indefinite"/></circle></svg>
  <svg width="52" height="11" viewBox="0 0 52 11"><rect width="52" height="11" fill="#1e1e1e"/><rect width="52" height="0.5" fill="#353535"/><rect y="10.5" width="52" height="0.5" fill="#0d0d0d"/><rect x="1" y="2" width="5.5" height="7" fill="#141414" stroke="#272727" stroke-width="0.5" rx="0.3"/><rect x="8" y="2" width="5.5" height="7" fill="#141414" stroke="#272727" stroke-width="0.5" rx="0.3"/><rect x="15" y="2" width="5.5" height="7" fill="#141414" stroke="#272727" stroke-width="0.5" rx="0.3"/><rect x="22" y="2" width="5.5" height="7" fill="#141414" stroke="#272727" stroke-width="0.5" rx="0.3"/><rect x="1.5" y="2.5" width="1.2" height="1.2" fill="#00e676"><animate attributeName="opacity" values="0.1;1;1;0.1" dur="2.1s" repeatCount="indefinite"/></rect><rect x="8.5" y="2.5" width="1.2" height="1.2" fill="#2a2a2a"/><rect x="15.5" y="2.5" width="1.2" height="1.2" fill="#00e676"><animate attributeName="opacity" values="1;1;0.1;1" dur="2.6s" repeatCount="indefinite"/></rect><rect x="22.5" y="2.5" width="1.2" height="1.2" fill="#00e676"><animate attributeName="opacity" values="1;0.1;1;1" dur="3.3s" repeatCount="indefinite"/></rect><circle cx="31" cy="5.5" r="1.4" fill="#00e676"><animate attributeName="opacity" values="1;0.3;1" dur="4.3s" repeatCount="indefinite"/></circle><circle cx="35.5" cy="5.5" r="1.1" fill="#29b6f6"/><circle cx="44" cy="5.5" r="3" fill="#181818" stroke="#2d2d2d" stroke-width="0.5"/><circle cx="44" cy="5.5" r="1.4" fill="#0277bd"><animate attributeName="opacity" values="1;0.6;1" dur="2.9s" repeatCount="indefinite"/></circle></svg>
  </div>
  <div class="rs"></div>
  </div>
  <div class="rack3d">
  <div class="rt" style="width:60px"></div>
  <div class="rf" style="width:60px">
  <svg width="52" height="11" viewBox="0 0 52 11"><rect width="52" height="11" fill="#1e1e1e"/><rect width="52" height="0.5" fill="#353535"/><rect y="10.5" width="52" height="0.5" fill="#0d0d0d"/><rect x="1" y="2" width="5.5" height="7" fill="#141414" stroke="#272727" stroke-width="0.5" rx="0.3"/><rect x="8" y="2" width="5.5" height="7" fill="#141414" stroke="#272727" stroke-width="0.5" rx="0.3"/><rect x="15" y="2" width="5.5" height="7" fill="#141414" stroke="#272727" stroke-width="0.5" rx="0.3"/><rect x="22" y="2" width="5.5" height="7" fill="#141414" stroke="#272727" stroke-width="0.5" rx="0.3"/><rect x="1.5" y="2.5" width="1.2" height="1.2" fill="#2a2a2a"/><rect x="8.5" y="2.5" width="1.2" height="1.2" fill="#00e676"><animate attributeName="opacity" values="1;0.1;1;1" dur="2.4s" repeatCount="indefinite"/></rect><rect x="15.5" y="2.5" width="1.2" height="1.2" fill="#00e676"><animate attributeName="opacity" values="0.1;1;1;0.1" dur="3.1s" repeatCount="indefinite"/></rect><rect x="22.5" y="2.5" width="1.2" height="1.2" fill="#00e676"><animate attributeName="opacity" values="1;1;0.1;1" dur="1.7s" repeatCount="indefinite"/></rect><circle cx="31" cy="5.5" r="1.4" fill="#00e676"><animate attributeName="opacity" values="1;0.4;1" dur="3.9s" repeatCount="indefinite"/></circle><circle cx="35.5" cy="5.5" r="1.1" fill="#29b6f6"/><circle cx="44" cy="5.5" r="3" fill="#181818" stroke="#2d2d2d" stroke-width="0.5"/><circle cx="44" cy="5.5" r="1.4" fill="#0277bd"><animate attributeName="opacity" values="0.7;1;0.7" dur="3.5s" repeatCount="indefinite"/></circle></svg>
  <svg width="52" height="11" viewBox="0 0 52 11"><rect width="52" height="11" fill="#1e1e1e"/><rect width="52" height="0.5" fill="#353535"/><rect y="10.5" width="52" height="0.5" fill="#0d0d0d"/><rect x="1" y="2" width="5.5" height="7" fill="#141414" stroke="#272727" stroke-width="0.5" rx="0.3"/><rect x="8" y="2" width="5.5" height="7" fill="#141414" stroke="#272727" stroke-width="0.5" rx="0.3"/><rect x="15" y="2" width="5.5" height="7" fill="#141414" stroke="#272727" stroke-width="0.5" rx="0.3"/><rect x="22" y="2" width="5.5" height="7" fill="#141414" stroke="#272727" stroke-width="0.5" rx="0.3"/><rect x="1.5" y="2.5" width="1.2" height="1.2" fill="#00e676"><animate attributeName="opacity" values="1;1;1;0.1;1" dur="2.6s" repeatCount="indefinite"/></rect><rect x="8.5" y="2.5" width="1.2" height="1.2" fill="#00e676"><animate attributeName="opacity" values="0.1;1;1;1;0.1" dur="2s" repeatCount="indefinite"/></rect><rect x="15.5" y="2.5" width="1.2" height="1.2" fill="#2a2a2a"/><rect x="22.5" y="2.5" width="1.2" height="1.2" fill="#00e676"><animate attributeName="opacity" values="1;0.1;1;1;1" dur="3.4s" repeatCount="indefinite"/></rect><circle cx="31" cy="5.5" r="1.4" fill="#00e676"><animate attributeName="opacity" values="0.3;1;0.3" dur="4.2s" repeatCount="indefinite"/></circle><circle cx="35.5" cy="5.5" r="1.1" fill="#29b6f6"/><circle cx="44" cy="5.5" r="3" fill="#181818" stroke="#2d2d2d" stroke-width="0.5"/><circle cx="44" cy="5.5" r="1.4" fill="#0277bd"><animate attributeName="opacity" values="1;0.5;1" dur="3.6s" repeatCount="indefinite"/></circle></svg>
  <svg width="52" height="11" viewBox="0 0 52 11"><rect width="52" height="11" fill="#1e1e1e"/><rect width="52" height="0.5" fill="#353535"/><rect y="10.5" width="52" height="0.5" fill="#0d0d0d"/><rect x="1" y="2" width="5.5" height="7" fill="#141414" stroke="#272727" stroke-width="0.5" rx="0.3"/><rect x="8" y="2" width="5.5" height="7" fill="#141414" stroke="#272727" stroke-width="0.5" rx="0.3"/><rect x="15" y="2" width="5.5" height="7" fill="#141414" stroke="#272727" stroke-width="0.5" rx="0.3"/><rect x="22" y="2" width="5.5" height="7" fill="#141414" stroke="#272727" stroke-width="0.5" rx="0.3"/><rect x="1.5" y="2.5" width="1.2" height="1.2" fill="#00e676"><animate attributeName="opacity" values="1;0.1;1;1" dur="3.2s" repeatCount="indefinite"/></rect><rect x="8.5" y="2.5" width="1.2" height="1.2" fill="#2a2a2a"/><rect x="15.5" y="2.5" width="1.2" height="1.2" fill="#00e676"><animate attributeName="opacity" values="1;1;0.1;1" dur="2.7s" repeatCount="indefinite"/></rect><rect x="22.5" y="2.5" width="1.2" height="1.2" fill="#00e676"><animate attributeName="opacity" values="0.1;1;1;0.1" dur="1.9s" repeatCount="indefinite"/></rect><circle cx="31" cy="5.5" r="1.4" fill="#00e676"><animate attributeName="opacity" values="1;0.3;1" dur="3.5s" repeatCount="indefinite"/></circle><circle cx="35.5" cy="5.5" r="1.1" fill="#29b6f6"/><circle cx="44" cy="5.5" r="3" fill="#181818" stroke="#2d2d2d" stroke-width="0.5"/><circle cx="44" cy="5.5" r="1.4" fill="#0277bd"><animate attributeName="opacity" values="0.8;1;0.8" dur="4s" repeatCount="indefinite"/></circle></svg>
  <svg width="52" height="11" viewBox="0 0 52 11"><rect width="52" height="11" fill="#1e1e1e"/><rect width="52" height="0.5" fill="#353535"/><rect y="10.5" width="52" height="0.5" fill="#0d0d0d"/><rect x="1" y="2" width="5.5" height="7" fill="#141414" stroke="#272727" stroke-width="0.5" rx="0.3"/><rect x="8" y="2" width="5.5" height="7" fill="#141414" stroke="#272727" stroke-width="0.5" rx="0.3"/><rect x="15" y="2" width="5.5" height="7" fill="#141414" stroke="#272727" stroke-width="0.5" rx="0.3"/><rect x="22" y="2" width="5.5" height="7" fill="#141414" stroke="#272727" stroke-width="0.5" rx="0.3"/><rect x="1.5" y="2.5" width="1.2" height="1.2" fill="#00e676"><animate attributeName="opacity" values="0.1;1;1;1;0.1" dur="2.3s" repeatCount="indefinite"/></rect><rect x="8.5" y="2.5" width="1.2" height="1.2" fill="#00e676"><animate attributeName="opacity" values="1;1;0.1;1;1" dur="3s" repeatCount="indefinite"/></rect><rect x="15.5" y="2.5" width="1.2" height="1.2" fill="#00e676"><animate attributeName="opacity" values="1;0.1;1;1;1" dur="2.5s" repeatCount="indefinite"/></rect><rect x="22.5" y="2.5" width="1.2" height="1.2" fill="#2a2a2a"/><circle cx="31" cy="5.5" r="1.4" fill="#00e676"><animate attributeName="opacity" values="1;0.4;1" dur="4.1s" repeatCount="indefinite"/></circle><circle cx="35.5" cy="5.5" r="1.1" fill="#29b6f6"/><circle cx="44" cy="5.5" r="3" fill="#181818" stroke="#2d2d2d" stroke-width="0.5"/><circle cx="44" cy="5.5" r="1.4" fill="#0277bd"><animate attributeName="opacity" values="1;0.6;1" dur="3.3s" repeatCount="indefinite"/></circle></svg>
  <svg width="52" height="11" viewBox="0 0 52 11"><rect width="52" height="11" fill="#1e1e1e"/><rect width="52" height="0.5" fill="#353535"/><rect y="10.5" width="52" height="0.5" fill="#0d0d0d"/><rect x="1" y="2" width="5.5" height="7" fill="#141414" stroke="#272727" stroke-width="0.5" rx="0.3"/><rect x="8" y="2" width="5.5" height="7" fill="#141414" stroke="#272727" stroke-width="0.5" rx="0.3"/><rect x="15" y="2" width="5.5" height="7" fill="#141414" stroke="#272727" stroke-width="0.5" rx="0.3"/><rect x="22" y="2" width="5.5" height="7" fill="#141414" stroke="#272727" stroke-width="0.5" rx="0.3"/><rect x="1.5" y="2.5" width="1.2" height="1.2" fill="#2a2a2a"/><rect x="8.5" y="2.5" width="1.2" height="1.2" fill="#00e676"><animate attributeName="opacity" values="1;0.1;1;1" dur="1.8s" repeatCount="indefinite"/></rect><rect x="15.5" y="2.5" width="1.2" height="1.2" fill="#00e676"><animate attributeName="opacity" values="0.1;1;0.1;1" dur="2.9s" repeatCount="indefinite"/></rect><rect x="22.5" y="2.5" width="1.2" height="1.2" fill="#00e676"><animate attributeName="opacity" values="1;1;1;0.1" dur="3.5s" repeatCount="indefinite"/></rect><circle cx="31" cy="5.5" r="1.4" fill="#00e676"><animate attributeName="opacity" values="0.5;1;0.5" dur="3.7s" repeatCount="indefinite"/></circle><circle cx="35.5" cy="5.5" r="1.1" fill="#29b6f6"/><circle cx="44" cy="5.5" r="3" fill="#181818" stroke="#2d2d2d" stroke-width="0.5"/><circle cx="44" cy="5.5" r="1.4" fill="#0277bd"><animate attributeName="opacity" values="0.9;1;0.9" dur="2.6s" repeatCount="indefinite"/></circle></svg>
  <svg width="52" height="11" viewBox="0 0 52 11"><rect width="52" height="11" fill="#1e1e1e"/><rect width="52" height="0.5" fill="#353535"/><rect y="10.5" width="52" height="0.5" fill="#0d0d0d"/><rect x="1" y="2" width="5.5" height="7" fill="#141414" stroke="#272727" stroke-width="0.5" rx="0.3"/><rect x="8" y="2" width="5.5" height="7" fill="#141414" stroke="#272727" stroke-width="0.5" rx="0.3"/><rect x="15" y="2" width="5.5" height="7" fill="#141414" stroke="#272727" stroke-width="0.5" rx="0.3"/><rect x="22" y="2" width="5.5" height="7" fill="#141414" stroke="#272727" stroke-width="0.5" rx="0.3"/><rect x="1.5" y="2.5" width="1.2" height="1.2" fill="#00e676"><animate attributeName="opacity" values="1;1;0.1;1;1" dur="2.8s" repeatCount="indefinite"/></rect><rect x="8.5" y="2.5" width="1.2" height="1.2" fill="#00e676"><animate attributeName="opacity" values="1;0.1;1;1;1" dur="2.1s" repeatCount="indefinite"/></rect><rect x="15.5" y="2.5" width="1.2" height="1.2" fill="#2a2a2a"/><rect x="22.5" y="2.5" width="1.2" height="1.2" fill="#00e676"><animate attributeName="opacity" values="0.1;1;1;0.1;1" dur="3.6s" repeatCount="indefinite"/></rect><circle cx="31" cy="5.5" r="1.4" fill="#00e676"><animate attributeName="opacity" values="1;0.3;1" dur="4s" repeatCount="indefinite"/></circle><circle cx="35.5" cy="5.5" r="1.1" fill="#29b6f6"/><circle cx="44" cy="5.5" r="3" fill="#181818" stroke="#2d2d2d" stroke-width="0.5"/><circle cx="44" cy="5.5" r="1.4" fill="#0277bd"><animate attributeName="opacity" values="1;0.5;1" dur="3.8s" repeatCount="indefinite"/></circle></svg>
  </div>
  <div class="rs"></div>
  </div>
  </div>
</div>
<div class="lhdr">
  <div class="uni-name">Groningen University</div>
  <div class="uni-sub">◈ &nbsp; A I &nbsp; C o m p u t e &nbsp; D e p o t &nbsp; ◈</div>
  <div class="scanline"></div>
</div>
""", unsafe_allow_html=True)

    _, centre, _ = st.columns([3, 2, 3])
    with centre:
        st.markdown('<div class="access-label">⬡ &nbsp; Secure Access Terminal &nbsp; ⬡</div>',
                    unsafe_allow_html=True)
        platform_url_in = st.text_input("Platform URL", value=st.session_state.platform_url)
        token_in = st.text_input("User Token", type="password", placeholder="Enter your user token")
        if st.button("⚡  authenticate", type="primary", use_container_width=True):
            if not token_in.strip():
                st.error("Please enter your user token.")
            else:
                with st.spinner("Verifying…"):
                    try:
                        import gridweave as _gw_check
                        _gw_check.auth(token_in.strip(), platform_url=platform_url_in)
                        _gw_check.endpoints()
                        st.session_state.user_token   = token_in.strip()
                        st.session_state.platform_url  = platform_url_in
                        st.session_state.authenticated = True
                        st.rerun()
                    except ImportError:
                        st.session_state.user_token   = token_in.strip()
                        st.session_state.platform_url  = platform_url_in
                        st.session_state.authenticated = True
                        st.rerun()
                    except Exception as e:
                        st.error(f"Authentication failed: {e}")

    st.markdown("""
<div style="text-align:center;margin-top:1.8rem;color:#81d4fa;
            font-family:'Courier New',monospace;font-size:0.74rem;
            letter-spacing:0.14em;opacity:0.88;">
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
    _gw.auth(st.session_state.user_token, platform_url=st.session_state.platform_url)
    _gw_available = True
except ImportError:
    _gw_available = False

_token_warn = _token_warning(st.session_state.user_token)
if _token_warn:
    st.warning(_token_warn)

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
        for k in ["authenticated", "user_token", "model_list", "endpoint", "chat_history",
                  "action_state", "action_log", "action_error", "_result_queue",
                  "gpu_cache", "_gpu_detecting", "_gpu_queue"]:
            st.session_state[k] = (
                False  if k == "authenticated" else
                ""     if k == "user_token"    else
                None   if k in ("model_list", "endpoint", "action_error", "_result_queue", "_gpu_queue") else
                "idle" if k == "action_state"  else
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
            _gw.auth(st.session_state.user_token, platform_url=st.session_state.platform_url)
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
        st.session_state["_cached_uid"] = _uid

        if _uid:
            # Endpoints with no "/" are bare/unqualified names — treat as yours.
            # Only put in Other Endpoints if it clearly carries a different user's prefix.
            my_eps    = [e for e in eps if
                         "/" not in e.get("name", "") or
                         e.get("name", "").startswith(f"{_uid}/")]
            other_eps = [e for e in eps if
                         "/" in e.get("name", "") and
                         not e.get("name", "").startswith(f"{_uid}/")]
        else:
            my_eps    = eps
            other_eps = []

        _CW = [2, 2, 2, 2, 2, 1, 1, 1, 1]

        # ── Your Endpoints ──────────────────────────────────────────────────
        if my_eps:
            h1,h2,h3,h4,h5,h6,_,_,_ = st.columns(_CW)
            h1.caption("Endpoint"); h2.caption("User"); h3.caption("Status")
            h4.caption("Model");    h5.caption("Server"); h6.caption("×GPU")

            for ep_info in my_eps:
                name         = ep_info.get("name", "")
                display_name = ep_info.get("display_name") or _ep_short_name(name)
                user_part    = ep_info.get("user_id") or _split_ep_name(name)[1]
                status = ep_info.get("status", "")
                model  = ep_info.get("model", "—")
                gpus   = ep_info.get("gpus", "?")
                icon   = "🟢" if status == "running" else ("🟡" if status in ("deploying","allocating") else "🔴")
                host, gpu, vendor, vram_gb = _hw(ep_info, hw)

                c1,c2,c3,c4,c5,c6,c7,c8,c9 = st.columns(_CW)
                c1.write(f"**{display_name}**")
                c2.write(user_part)
                c3.write(f"{icon} {status}")
                c4.write(model.split("/")[-1])
                c5.write(host)
                c6.write(str(gpus))

                with c7:
                    if status == "running" and st.button("Chat", key=f"chat_{name}", use_container_width=True):
                        from gridweave.serve import Endpoint
                        st.session_state.endpoint = Endpoint(
                            name=display_name, status="running",
                            endpoint_type=ep_info.get("endpoint_type", "vllm"),
                            model=ep_info.get("model", ""), image=ep_info.get("image", ""),
                            gpus=ep_info.get("gpus", 1), vendor=ep_info.get("vendor"),
                            spec=ep_info.get("spec"),
                        )
                        st.session_state.chat_history = []
                        st.rerun()
                with c8:
                    if status == "running":
                        if st.button("Stop", key=f"stop_{name}", use_container_width=True):
                            _gw.auth(st.session_state.user_token, platform_url=st.session_state.platform_url)
                            _gw_direct("post", f"/v1/endpoints/{display_name}/stop")
                            if st.session_state.endpoint and st.session_state.endpoint.name in (name, display_name):
                                st.session_state.endpoint = None
                            st.rerun()
                    elif status == "stopped":
                        if st.button("Start", key=f"start_{name}", use_container_width=True):
                            _gw.auth(st.session_state.user_token, platform_url=st.session_state.platform_url)
                            st.session_state.action_state = "busy"
                            st.session_state.action_label = f"Starting '{display_name}'…"
                            st.session_state.action_log   = []
                            st.session_state.action_error = None
                            _launch(_start_worker, (display_name, st.session_state.user_token,
                                                    st.session_state.platform_url, _uid))
                            st.rerun()
                with c9:
                    if st.button("Delete", key=f"del_{name}", use_container_width=True):
                        _gw.auth(st.session_state.user_token, platform_url=st.session_state.platform_url)
                        _gw_direct("delete", f"/v1/endpoints/{display_name}")
                        if st.session_state.endpoint and st.session_state.endpoint.name in (name, display_name):
                            st.session_state.endpoint = None
                        st.rerun()
        else:
            st.info("No endpoints found.")

        # ── Other Endpoints ─────────────────────────────────────────────────
        if other_eps:
            st.divider()
            st.subheader("Other Endpoints")
            h1,h2,h3,h4,h5,h6,_,_,_ = st.columns(_CW)
            h1.caption("Endpoint"); h2.caption("User"); h3.caption("Status")
            h4.caption("Model");    h5.caption("Server"); h6.caption("×GPU")

            for ep_info in other_eps:
                name         = ep_info.get("name", "")
                display_name = ep_info.get("display_name") or _ep_short_name(name)
                user_part    = ep_info.get("user_id") or _split_ep_name(name)[1]
                status = ep_info.get("status", "")
                model  = ep_info.get("model", "—")
                gpus   = ep_info.get("gpus", "?")
                icon   = "🟢" if status == "running" else ("🟡" if status in ("deploying","allocating") else "🔴")
                host, gpu, vendor, vram_gb = _hw(ep_info, hw)

                c1,c2,c3,c4,c5,c6,c7,_,_ = st.columns(_CW)
                c1.write(f"**{display_name}**")
                c2.write(user_part)
                c3.write(f"{icon} {status}")
                c4.write(model.split("/")[-1])
                c5.write(host)
                c6.write(str(gpus))

                with c7:
                    if status == "running" and st.button("Chat", key=f"chat_{name}", use_container_width=True):
                        from gridweave.serve import Endpoint
                        st.session_state.endpoint = Endpoint(
                            name=display_name, status="running",
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
        cc1, cc2 = st.columns(2)
        with cc1:
            hf_token = st.text_input(
                "HuggingFace Token", value="",
                type="password",
                key="hf_token_input",
                help="Optional. Required for gated models such as Llama. Leave empty for public models like Qwen. Generate one at huggingface.co/settings/tokens.",
            )
        with cc2:
            s3_endpoint   = st.text_input(
                "S3/R2 Endpoint", placeholder="https://<account>.r2.cloudflarestorage.com",
                help="URL of your S3-compatible storage endpoint (e.g. Cloudflare R2 or AWS S3).",
            )
            s3_access_key = st.text_input(
                "S3/R2 Access Key", placeholder="your-access-key-id", type="password",
                help="Access key ID for your S3/R2 bucket.",
            )
            s3_secret_key = st.text_input(
                "S3/R2 Secret Key", placeholder="your-secret-access-key", type="password",
                help="Secret access key for your S3/R2 bucket.",
            )
            s3_bucket = st.text_input(
                "S3/R2 Bucket", placeholder="my-bucket",
                help="Name of the bucket where your model files are stored.",
            )

    ma, mb, mc = st.columns(3)
    with ma:
        _uid_ml = st.session_state.get("_cached_uid")
        if st.session_state.model_list is None:
            st.session_state.model_list = _load_llm_list(_uid_ml)
        _llm_opts = st.session_state.model_list

        _inp_c, _btn_c = st.columns([5, 1])
        with _inp_c:
            _new_model = st.text_input("Model ID", placeholder="org/model-name", key="new_model_input")
        with _btn_c:
            st.write("")
            if st.button("Add", key="add_model_btn", use_container_width=True):
                if _new_model.strip() and _new_model.strip() not in _llm_opts:
                    st.session_state.model_list = _llm_opts + [_new_model.strip()]
                    _save_llm_list(st.session_state.model_list, _uid_ml)
                    st.rerun()
        if _llm_opts:
            model_id = st.selectbox("Saved models", _llm_opts, label_visibility="collapsed")
            for _m in _llm_opts:
                _mc, _md = st.columns([5, 1])
                _mc.caption(_m)
                with _md:
                    if st.button("✕", key=f"rm_{_m}", use_container_width=True):
                        st.session_state.model_list = [m for m in _llm_opts if m != _m]
                        _save_llm_list(st.session_state.model_list, _uid_ml)
                        st.rerun()
        else:
            model_id = _new_model.strip()
            st.caption("Type a model ID above and click **Add** to save it, or type one and deploy directly.")
    with mb: vram          = st.selectbox("VRAM", ["4GB","8GB","16GB","24GB","40GB","80GB"])
    with mc: endpoint_name = st.text_input("Endpoint Name", value="llama-eric")

    action = st.session_state.action_state
    if action == "idle":
        if st.button("🚀 Deploy", type="primary", use_container_width=True):
            if not model_id:
                st.error("Please enter a Model ID.")
            elif not _token_ok(st.session_state.user_token):
                st.error("Your GridWeave session has expired. Please log out and log in again.")
            else:
                st.session_state.action_state = "busy"
                st.session_state.action_label = f"Deploying {model_id}…"
                st.session_state.action_log   = []
                st.session_state.action_error = None
                _launch(_deploy_worker, (dict(
                    platform_url=st.session_state.platform_url,
                    user_token=st.session_state.user_token,
                    hf_token=hf_token,
                    s3_endpoint=s3_endpoint,
                    s3_access_key=s3_access_key,
                    s3_secret_key=s3_secret_key,
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
        ep_bare = ep.name  # full API name e.g. "User--endpoint-name"
        _cu = st.session_state.get("_cached_uid")
        for _e in _gw.endpoints():
            if _e.get("display_name") == ep_bare or _e.get("name") == ep_bare:
                _ep_info = _e; break
        _host, _gpu, _vendor, _vgb = _hw(_ep_info, _active_hw)
        _gpus = _ep_info.get("gpus", ep.gpus)
        if _vendor != "—" and _vgb:
            _start_gpu_detect(_vendor, _vgb)
    except Exception:
        _host, _gpu, _vendor, _vgb, _gpus = "—", ep.vendor or "—", ep.vendor or "—", 0, ep.gpus

    short_model = ep.model.split("/")[-1]
    ep_display, _ = _split_ep_name(ep.name)
    st.success(f"✅ **{ep_display}**")
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
    _m = ep.model.lower().split("/")[-1]
    _is_instruct = "instruct" in _m or "chat" in _m

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
                            data = _gw_direct(
                                "post",
                                "/v1/chat/completions",
                                json={"model": ep_bare,
                                      "messages": [{"role": m["role"], "content": m["content"]}
                                                   for m in st.session_state.chat_history],
                                      "max_tokens": max_tokens, "temperature": temperature},
                            )
                            response = data["choices"][0]["message"]["content"]
                        else:
                            prompt_text = "\n".join(
                                f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
                                for m in st.session_state.chat_history
                            ) + "\nAssistant:"
                            data = _gw_direct(
                                "post",
                                "/v1/completions",
                                json={"model": ep_bare,
                                      "prompt": prompt_text, "max_tokens": max_tokens,
                                      "temperature": temperature},
                            )
                            raw = data["choices"][0]["text"]
                            response = re.split(r"\n(User|Assistant):", raw)[0].strip()
                    except Exception as exc:
                        response = f"⚠️ Error: {exc}\n\n(ep.name={ep.name!r}, ep_bare={ep_bare!r})"
                st.write(response)
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            st.rerun()

else:
    if st.session_state.action_state == "idle":
        st.info("Deploy a new endpoint or click **Chat** on an existing one above to start chatting.")
