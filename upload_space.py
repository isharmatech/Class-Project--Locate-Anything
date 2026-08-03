"""One-shot script: create the Hugging Face Space and upload the demo files.

Run:
    python upload_space.py
It will prompt for your HF token (get one at https://huggingface.co/settings/tokens
with "write" access). Re-running is safe — it only uploads changed files.
"""
import getpass
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo

OWNER = "cyberfrost7"
SPACE_NAME = "locate-anything-demo"
SPACE_REPO_ID = f"{OWNER}/{SPACE_NAME}"
SPACE_DIR = Path(__file__).parent / "hf-space"


def main() -> int:
    token = os.getenv("HF_TOKEN") or getpass.getpass(
        "Paste your Hugging Face token (write access): "
    ).strip()
    if not token:
        print("No token provided — aborting.")
        return 1

    api = HfApi(token=token)

    # Only call create_repo on the first run. Re-calling create_repo with
    # hardware/sdk params on an already-existing Space returns an empty 400
    # Bad Request, so we probe with repo_info first and skip creation when
    # the Space already exists.
    space_exists = False
    try:
        api.repo_info(repo_id=SPACE_REPO_ID, repo_type="space", token=token)
        space_exists = True
    except Exception:
        pass

    if space_exists:
        print(f"Space {SPACE_REPO_ID} already exists — skipping creation, will just upload.")
    else:
        print(f"Creating Space {SPACE_REPO_ID} (Gradio SDK, ZeroGPU) ...")
        # Python version is not settable via create_repo() in this huggingface_hub
        # version — it's declared in hf-space/README.md's YAML frontmatter instead
        # (python_version: "3.12"), which Hugging Face reads on its own.
        create_repo(
            repo_id=SPACE_REPO_ID,
            repo_type="space",
            space_sdk="gradio",
            # The installed huggingface_hub's SpaceHardware enum predates ZeroGPU
            # and doesn't include it, but the server API accepts the raw string
            # "zerogpu" (no hyphen).
            space_hardware="zerogpu",
            private=False,
            exist_ok=True,
            token=token,
        )

    print(f"Uploading contents of {SPACE_DIR} to {SPACE_REPO_ID} ...")
    api.upload_folder(
        repo_id=SPACE_REPO_ID,
        repo_type="space",
        folder_path=str(SPACE_DIR),
        commit_message="Locate Anything 3B ZeroGPU demo",
        token=token,
    )

    # Set HF_TOKEN as a Space Secret so ZeroGPU attributes GPU-minute usage to
    # this PRO account instead of falling back to anonymous IP-based quotas.
    # Secrets are stored server-side only (never written to the repo files) and
    # are redacted in the HF UI. Setting/changing a secret triggers a Space
    # restart, which is expected.
    print("Setting HF_TOKEN Space secret (attributes ZeroGPU quota to your PRO account) ...")
    api.add_space_secret(
        repo_id=SPACE_REPO_ID,
        key="HF_TOKEN",
        value=token,
        token=token,
    )

    print("\nDone. Watch the build log at:")
    print(f"  https://huggingface.co/spaces/{SPACE_REPO_ID}")
    print("Once the status reads 'Running', the demo is live at:")
    print(f"  https://{OWNER}-{SPACE_NAME}.hf.space")
    return 0


if __name__ == "__main__":
    sys.exit(main())
