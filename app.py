import queue
import subprocess
import sys
import threading
import time

import streamlit as st

st.set_page_config(page_title="Gridweave LLM Launcher", page_icon="🦙", layout="wide")

WHL_PATH = "/home/jacovandijk/Projects/personal-load-llm/gridweave_sdk-0.2.0-py3-none-any.whl"
WHL_URL  = "https://pub-cbb8992ad1bd437b81d58d5b2da09787.r2.dev/tarball/gridweave_sdk-0.2.0-py3-none-any.whl"

# ── Session state ─────────────────────────────────────────────────────────────
for _k, _v in [
    ("action_state", "idle"),   # idle | busy | error
    ("action_label", ""),
    ("action_log",   []),
    ("action_error", None),
    ("endpoint",     None),     # Endpoint object selected for chat
    ("chat_history", []),
    ("_result_queue", None),
    ("platform_url", "https://platform.gridweave.io"),
    ("admin_token",  "25d7bcb8f31bb67ef3edfbcd1c15a9d53c2fb1773a23e6abbd5beb2814519ee7"),
]:
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ── Background helpers ────────────────────────────────────────────────────────
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
    """Forwards SDK print() lines into the result queue."""
    def __init__(self, orig, q): self._orig, self._q = orig, q
    def write(self, s):
        self._orig.write(s); self._orig.flush()
        if s.strip(): self._q.put(("log", s.strip()))
    def flush(self): self._orig.flush()


def _deploy_worker(cfg: dict, q: queue.Queue):
    try:
        q.put(("log", "Installing SDK…"))
        _install_deps()
        import importlib, gridweave
        importlib.reload(gridweave)
        q.put(("log", f"Authenticating with {cfg['platform_url']}…"))
        gridweave.auth(cfg["admin_token"], platform_url=cfg["platform_url"])
        q.put(("log", f"Deploying {cfg['model_id']} ({cfg['vram']}) as '{cfg['endpoint_name']}'…"))
        old = sys.stdout; sys.stdout = _Tee(old, q)
        try:
            ep = gridweave.serve(
                model=cfg["model_id"], hf_token=cfg["hf_token"],
                vram=cfg["vram"], name=cfg["endpoint_name"],
            )
        finally:
            sys.stdout = old
        q.put(("done", ep))
    except Exception as exc:
        q.put(("error", str(exc)))


def _start_worker(name: str, q: queue.Queue):
    try:
        import gridweave
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


def _launch(target, args):
    q = queue.Queue()
    st.session_state._result_queue = q
    threading.Thread(target=target, args=(*args, q), daemon=True).start()


# ── Poll queue on every rerun ─────────────────────────────────────────────────
def _poll():
    q = st.session_state._result_queue
    if q is None:
        return
    while True:
        try:
            kind, value = q.get_nowait()
        except queue.Empty:
            break
        if kind == "log":
            st.session_state.action_log.append(value)
        elif kind == "done":
            st.session_state.action_state = "idle"
            st.session_state.endpoint = value
            st.session_state.chat_history = []
            st.session_state._result_queue = None
        elif kind == "error":
            st.session_state.action_state = "error"
            st.session_state.action_error = value
            st.session_state._result_queue = None

_poll()

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🦙 Gridweave LLM Launcher")

try:
    import gridweave as _gw
    _gw.auth(st.session_state.admin_token, platform_url=st.session_state.platform_url)
    _gw_available = True
except ImportError:
    _gw_available = False

# ══════════════════════════════════════════════════════════════════════════════
# 1. Endpoint Manager
# ══════════════════════════════════════════════════════════════════════════════
with st.expander("📡 Endpoint Manager", expanded=(st.session_state.endpoint is None)):

    # ── Endpoints list ────────────────────────────────────────────────────────
    if _gw_available:
        hdr, _, ref_col = st.columns([4, 3, 1])
        with hdr:
            st.subheader("Your Endpoints")
        with ref_col:
            do_refresh = st.button("🔄 Refresh", use_container_width=True)

        try:
            eps = _gw.endpoints()
        except Exception as e:
            eps = []; st.warning(f"Could not load endpoints: {e}")

        if eps:
            for ep_info in eps:
                name   = ep_info.get("name", "")
                status = ep_info.get("status", "")
                model  = ep_info.get("model", "—")
                gpus   = ep_info.get("gpus", "?")
                icon   = "🟢" if status == "running" else ("🟡" if status in ("deploying", "allocating") else "🔴")

                c1, c2, c3, c4, c5, c6, c7, c8 = st.columns([3, 2, 2, 1, 1, 1, 1, 1])
                c1.write(f"**{name}**")
                c2.write(f"{icon} {status}")
                c3.write(model.split("/")[-1])
                c4.write(f"{gpus} GPU")

                with c5:
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
                with c6:
                    if status == "running":
                        if st.button("Stop", key=f"stop_{name}", use_container_width=True):
                            _gw.stop(name)
                            if st.session_state.endpoint and st.session_state.endpoint.name == name:
                                st.session_state.endpoint = None
                            st.rerun()
                    elif status == "stopped":
                        if st.button("Start", key=f"start_{name}", use_container_width=True):
                            st.session_state.action_state = "busy"
                            st.session_state.action_label = f"Starting '{name}'…"
                            st.session_state.action_log = []
                            st.session_state.action_error = None
                            _launch(_start_worker, (name,))
                            st.rerun()
                with c7:
                    if st.button("Delete", key=f"del_{name}", use_container_width=True):
                        _gw.delete(name)
                        if st.session_state.endpoint and st.session_state.endpoint.name == name:
                            st.session_state.endpoint = None
                        st.rerun()
        else:
            st.info("No endpoints found.")
    else:
        st.info("Deploy a model below to install the SDK and create your first endpoint.")

    st.divider()

    # ── Deploy form ───────────────────────────────────────────────────────────
    st.subheader("Deploy New Endpoint")

    with st.expander("🔑 Credentials", expanded=not _gw_available):
        cc1, cc2 = st.columns(2)
        with cc1:
            platform_url = st.text_input("Platform URL", key="platform_url")
            admin_token  = st.text_input("Admin Token",  key="admin_token", type="password")
            hf_token     = st.text_input("HuggingFace Token",
                value="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", type="password")
        with cc2:
            st.text_input("R2 Endpoint",
                value="https://d97bc2f3151f58bc38c26d9da78c21e9.r2.cloudflarestorage.com")
            st.text_input("R2 Access Key", value="06506278cfb40d0777bd9d2f0d63076b", type="password")
            st.text_input("R2 Secret Key",
                value="28f84de15ee32e539a6f21020d413cec7cc57398968e11fbc3efbb2978889026", type="password")
            st.text_input("R2 Bucket", value="gridweave")

    ma, mb, mc = st.columns(3)
    with ma: model_id      = st.text_input("Model ID",       value="meta-llama/Llama-3.2-1B")
    with mb: vram          = st.selectbox("VRAM", ["4GB", "8GB", "16GB", "24GB", "40GB", "80GB"])
    with mc: endpoint_name = st.text_input("Endpoint Name",  value="llama-eric")

    action = st.session_state.action_state

    if action == "idle":
        if st.button("🚀 Deploy", type="primary", use_container_width=True):
            if not admin_token or not hf_token:
                st.error("Admin Token and HuggingFace Token are required.")
            else:
                st.session_state.action_state = "busy"
                st.session_state.action_label = f"Deploying {model_id}…"
                st.session_state.action_log   = []
                st.session_state.action_error = None
                _launch(_deploy_worker, (dict(
                    platform_url=platform_url, admin_token=admin_token,
                    hf_token=hf_token, model_id=model_id, vram=vram,
                    endpoint_name=endpoint_name,
                ),))
                st.rerun()

    elif action == "busy":
        st.info(st.session_state.action_label)
        st.code("\n".join(st.session_state.action_log) or "Starting…", language=None)
        time.sleep(2)
        st.rerun()

    elif action == "error":
        st.error(f"Failed: {st.session_state.action_error}")
        if st.button("↩ Retry"):
            st.session_state.action_state = "idle"
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# 2. Chat
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.endpoint:
    ep = st.session_state.endpoint
    st.divider()

    c1, c2, c3, c4, c5 = st.columns([4, 2, 2, 2, 2])
    with c1: st.success(f"✅ Chatting with **{ep.name}**")
    with c2: st.metric("Model",  ep.model.split("/")[-1])
    with c3: st.metric("GPUs",   ep.gpus)
    with c4: st.metric("Vendor", ep.vendor or "auto")
    with c5:
        if st.button("✖ Disconnect", use_container_width=True):
            st.session_state.endpoint = None
            st.session_state.chat_history = []
            st.rerun()

    left, right = st.columns([3, 1])

    with right:
        st.caption("Settings")
        max_tokens  = st.slider("Max tokens",  64, 2048, 512, step=64)
        temperature = st.slider("Temperature", 0.0, 2.0,  0.7, step=0.05)
        if st.button("🗑 Clear chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    with left:
        st.subheader(f"Chat — {ep.model.split('/')[-1]}")

        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        if prompt := st.chat_input(f"Message {ep.model.split('/')[-1]}…"):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)
            with st.chat_message("assistant"):
                with st.spinner("Thinking…"):
                    try:
                        response = ep.chat(
                            [{"role": m["role"], "content": m["content"]}
                             for m in st.session_state.chat_history],
                            max_tokens=max_tokens, temperature=temperature,
                        )
                    except Exception as exc:
                        response = f"⚠️ Error: {exc}"
                st.write(response)
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            st.rerun()

else:
    if st.session_state.action_state == "idle":
        st.info("Deploy a new endpoint or click **Chat** on an existing one above to start chatting.")
