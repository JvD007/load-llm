!export V=$(curl -s https://pub-cbb8992ad1bd437b81d58d5b2da09787.r2.dev/tarball/sdk-latest-version.txt) && pip install --quiet --force-reinstall --no-deps https://pub-cbb8992ad1bd437b81d58d5b2da09787.r2.dev/tarball/gridweave_sdk-${V}-py3-none-any.whl
import gridweave
print(f"gridweave v{gridweave.__version__}")

PLATFORM_URL = "https://platform.gridweave.io"
ADMIN_TOKEN = "25d7bcb8f31bb67ef3edfbcd1c15a9d53c2fb1773a23e6abbd5beb2814519ee7"

# R2 credentials (S3-compatible)
R2_ENDPOINT = "https://d97bc2f3151f58bc38c26d9da78c21e9.r2.cloudflarestorage.com"
R2_ACCESS_KEY = "06506278cfb40d0777bd9d2f0d63076b"
R2_SECRET_KEY = "28f84de15ee32e539a6f21020d413cec7cc57398968e11fbc3efbb2978889026"
R2_BUCKET = "gridweave"

# HuggingFace (gated models)
HF_TOKEN = "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"

endpoint_name = "llama-eric"
model = "meta-llama/Llama-3.2-1B"
vram = "4GB"

gridweave.auth(ADMIN_TOKEN, platform_url=PLATFORM_URL)

ep = gridweave.serve(
    model=model,
    hf_token=HF_TOKEN,
    vram=vram,
    name=endpoint_name,
)

