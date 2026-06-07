"""Download the SAEmnesia checkpoint from HuggingFace Hub."""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser(
        description="Download SAEmnesia checkpoint from HuggingFace Hub."
    )
    parser.add_argument(
        "save_dir",
        type=str,
        help="Directory where the checkpoint will be saved.",
    )
    args = parser.parse_args()

    HOOKPOINT = "unet.up_blocks.1.attentions.1"

    save_dir = Path(args.save_dir) / HOOKPOINT
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading leno3003/SAEmnesia to {save_dir} ...")
    snapshot_download(
        repo_id="leno3003/SAEmnesia",
        repo_type="model",
        local_dir=str(save_dir),
    )
    print(f"Checkpoint saved to {save_dir}")


if __name__ == "__main__":
    main()
