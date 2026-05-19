import subprocess
import sys
import streamlit as st

st.set_page_config(page_title="Gridweave LLM Launcher", page_icon="🦙", layout="centered")

st.title("🦙 Gridweave LLM Launcher")
st.caption("Configure credentials and deploy a model endpoint")

# ── Credentials ──────────────────────────────────────────────────────────────
with st.expander("🔑 Credentials", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        platform_url = st.text_input(
            "Platform URL",
            value="https://platform.gridweave.io",
        )
        admin_token = st.text_input(
            "Admin Token",
            value="25d7bcb8f31bb67ef3edfbcd1c15a9d53c2fb1773a23e6abbd5beb2814519ee7",
            type="password",
        )
        hf_token = st.text_input(
            "HuggingFace Token",
            value="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",
            type="password",
        )
    with col2:
        r2_endpoint = st.text_input(
            "R2 Endpoint",
            value="https://d97bc2f3151f58bc38c26d9da78c21e9.r2.cloudflarestorage.com",
        )
        r2_access_key = st.text_input(
            "R2 Access Key",
            value="06506278cfb40d0777bd9d2f0d63076b",
            type="password",
        )
        r2_secret_key = st.text_input(
            "R2 Secret Key",
            value="28f84de15ee32e539a6f21020d413cec7cc57398968e11fbc3efbb2978889026",
            type="password",
        )
        r2_bucket = st.text_input("R2 Bucket", value="gridweave")

# ── Model config ──────────────────────────────────────────────────────────────
st.divider()
st.subheader("Model Configuration")

col_a, col_b, col_c = st.columns(3)
with col_a:
    model_id = st.text_input("Model ID", value="meta-llama/Llama-3.2-1B")
with col_b:
    vram = st.selectbox("VRAM", ["4GB", "8GB", "16GB", "24GB", "40GB", "80GB"], index=0)
with col_c:
    endpoint_name = st.text_input("Endpoint Name", value="llama-eric")

# ── Deploy ────────────────────────────────────────────────────────────────────
st.divider()

if st.button("🚀 Deploy Model", type="primary", use_container_width=True):
    if not admin_token or not hf_token:
        st.error("Admin Token and HuggingFace Token are required.")
    else:
        whl_path = "/home/jacovandijk/Projects/personal-load-llm/gridweave_sdk-0.2.0-py3-none-any.whl"
        whl_url = "https://pub-cbb8992ad1bd437b81d58d5b2da09787.r2.dev/tarball/gridweave_sdk-0.2.0-py3-none-any.whl"

        with st.spinner("Installing gridweave SDK…"):
            import os, urllib.request
            if not os.path.exists(whl_path):
                with st.spinner("Downloading gridweave SDK wheel…"):
                    urllib.request.urlretrieve(whl_url, whl_path)
            try:
                result = subprocess.run(
                    [
                        sys.executable, "-m", "pip", "install",
                        "--quiet", "--force-reinstall", "--no-deps",
                        "--break-system-packages",
                        whl_path,
                    ],
                    capture_output=True,
                    text=True,
                )
                if result.returncode != 0:
                    st.warning(f"SDK install warning: {result.stderr[:300]}")
            except Exception as e:
                st.warning(f"SDK install skipped: {e}")

        with st.spinner(f"Deploying `{model_id}` as `{endpoint_name}`…"):
            try:
                import gridweave

                st.info(f"gridweave v{gridweave.__version__}")

                gridweave.auth(admin_token, platform_url=platform_url)

                ep = gridweave.serve(
                    model=model_id,
                    hf_token=hf_token,
                    vram=vram,
                    name=endpoint_name,
                )

                st.success("✅ Model deployed successfully!")
                st.json(ep if isinstance(ep, dict) else {"endpoint": str(ep)})

            except ModuleNotFoundError:
                st.error(
                    "gridweave SDK not found. Make sure `gridweave_sdk-0.2.0-py3-none-any.whl` "
                    "is in the project directory and re-click Deploy."
                )
            except Exception as e:
                st.error(f"Deployment failed: {e}")
