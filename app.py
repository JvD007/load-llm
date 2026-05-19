import queue
import subprocess
import sys
import threading
import time

import streamlit as st

st.set_page_config(page_title="Gridweave LLM Launcher", page_icon="🦙", layout="wide")

WHL_PATH = "/home/jacovandijk/Projects/personal-load-llm/gridweave_sdk-0.2.0-py3-none-any.whl"
WHL_URL = "https://pub-cbb8992ad1bd437b81d58d5b2da09787.r2.dev/tarball/gridweave_sdk-0.2.0-py3-none-any.whl"

# ── Session state ─────────────────────────────────────────────────────────────
for _k, _v in [
    ("deploy_state", "idle"),   # idle | deploying | running | error
    ("endpoint", None),
    ("deploy_log", []),
    ("deploy_error", None),
    ("chat_history", []),
    ("_result_queue", None),
]:
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ── Background worker ─────────────────────────────────────────────────────────
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


def _deploy_worker(cfg: dict, q: queue.Queue):
    try:
        q.put(("log", "Installing SDK dependencies…"))
        _install_deps()

        import importlib
        import gridweave
        importlib.reload(gridweave)

        q.put(("log", f"Authenticating with {cfg['platform_url']}…"))
        gridweave.auth(cfg["admin_token"], platform_url=cfg["platform_url"])

        q.put(("log", f"Requesting deployment of {cfg['model_id']} ({cfg['vram']})…"))

        # Forward SDK's print() progress lines into the queue
        class _Tee:
            def __init__(self, orig): self._orig = orig
            def write(self, s):
                self._orig.write(s); self._orig.flush()
                if s.strip():
                    q.put(("log", s.strip()))
            def flush(self): self._orig.flush()

        old_stdout = sys.stdout
        sys.stdout = _Tee(old_stdout)
        try:
            ep = gridweave.serve(
                model=cfg["model_id"],
                hf_token=cfg["hf_token"],
                vram=cfg["vram"],
                name=cfg["endpoint_name"],
            )
        finally:
            sys.stdout = old_stdout

        q.put(("done", ep))
    except Exception as exc:
        q.put(("error", str(exc)))


# ── Drain queue on every rerun ────────────────────────────────────────────────
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
            st.session_state.deploy_log.append(value)
        elif kind == "done":
            st.session_state.deploy_state = "running"
            st.session_state.endpoint = value
            st.session_state._result_queue = None
        elif kind == "error":
            st.session_state.deploy_state = "error"
            st.session_state.deploy_error = value
            st.session_state._result_queue = None


_poll()

# ── Layout ────────────────────────────────────────────────────────────────────
st.title("🦙 Gridweave LLM Launcher")

state = st.session_state.deploy_state

# ─── Configuration (shown when not yet running) ───────────────────────────────
if state in ("idle", "deploying", "error"):
    with st.expander("🔑 Credentials", expanded=(state == "idle")):
        c1, c2 = st.columns(2)
        with c1:
            platform_url  = st.text_input("Platform URL", value="https://platform.gridweave.io")
            admin_token   = st.text_input("Admin Token",
                value="25d7bcb8f31bb67ef3edfbcd1c15a9d53c2fb1773a23e6abbd5beb2814519ee7",
                type="password")
            hf_token      = st.text_input("HuggingFace Token",
                value="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", type="password")
        with c2:
            st.text_input("R2 Endpoint",
                value="https://d97bc2f3151f58bc38c26d9da78c21e9.r2.cloudflarestorage.com")
            st.text_input("R2 Access Key",
                value="06506278cfb40d0777bd9d2f0d63076b", type="password")
            st.text_input("R2 Secret Key",
                value="28f84de15ee32e539a6f21020d413cec7cc57398968e11fbc3efbb2978889026",
                type="password")
            st.text_input("R2 Bucket", value="gridweave")

    st.subheader("Model Configuration")
    ca, cb, cc = st.columns(3)
    with ca:
        model_id = st.text_input("Model ID", value="meta-llama/Llama-3.2-1B")
    with cb:
        vram = st.selectbox("VRAM", ["4GB", "8GB", "16GB", "24GB", "40GB", "80GB"])
    with cc:
        endpoint_name = st.text_input("Endpoint Name", value="llama-eric")

# ─── Idle: deploy button ──────────────────────────────────────────────────────
if state == "idle":
    st.divider()
    if st.button("🚀 Deploy Model", type="primary", use_container_width=True):
        if not admin_token or not hf_token:
            st.error("Admin Token and HuggingFace Token are required.")
        else:
            st.session_state.deploy_state = "deploying"
            st.session_state.deploy_log = []
            st.session_state.deploy_error = None
            q = queue.Queue()
            st.session_state._result_queue = q
            threading.Thread(
                target=_deploy_worker,
                args=(dict(
                    platform_url=platform_url,
                    admin_token=admin_token,
                    hf_token=hf_token,
                    model_id=model_id,
                    vram=vram,
                    endpoint_name=endpoint_name,
                ), q),
                daemon=True,
            ).start()
            st.rerun()

# ─── Deploying: live log ──────────────────────────────────────────────────────
elif state == "deploying":
    st.info("Deploying — this may take a few minutes on a cold start.")
    st.code("\n".join(st.session_state.deploy_log) or "Starting…", language=None)
    time.sleep(2)
    st.rerun()

# ─── Error ────────────────────────────────────────────────────────────────────
elif state == "error":
    st.error(f"Deployment failed: {st.session_state.deploy_error}")
    if st.button("↩ Retry"):
        st.session_state.deploy_state = "idle"
        st.rerun()

# ─── Running: endpoint info + chat ───────────────────────────────────────────
elif state == "running":
    ep = st.session_state.endpoint

    # Status bar
    c1, c2, c3, c4, c5 = st.columns([4, 2, 2, 2, 2])
    with c1:
        st.success(f"✅ **{ep.name}**  |  {ep.status}")
    with c2:
        st.metric("Model", ep.model.split("/")[-1])
    with c3:
        st.metric("GPUs", ep.gpus)
    with c4:
        st.metric("Vendor", ep.vendor or "auto")
    with c5:
        if st.button("⏹ Stop", type="secondary", use_container_width=True):
            try:
                import gridweave
                gridweave.stop(ep.name)
            except Exception as e:
                st.warning(f"Stop request: {e}")
            st.session_state.deploy_state = "idle"
            st.session_state.endpoint = None
            st.session_state.chat_history = []
            st.rerun()

    st.divider()

    # ── Chat ──────────────────────────────────────────────────────────────────
    left, right = st.columns([3, 1])

    with right:
        st.caption("Settings")
        max_tokens = st.slider("Max tokens", 64, 2048, 512, step=64)
        temperature = st.slider("Temperature", 0.0, 2.0, 0.7, step=0.05)
        if st.button("🗑 Clear chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    with left:
        st.subheader(f"Chat with {ep.model.split('/')[-1]}")

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
                            max_tokens=max_tokens,
                            temperature=temperature,
                        )
                    except Exception as exc:
                        response = f"⚠️ Error: {exc}"
                st.write(response)
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            st.rerun()
