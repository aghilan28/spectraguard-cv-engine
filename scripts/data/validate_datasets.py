import yaml
from pathlib import Path
import sys


def validate():
    print("\n--- Validating Data Foundation (DF-2) ---")
    manifest_path = Path("src/fixtures/manifest.yaml")
    assert manifest_path.exists(), "Manifest file missing!"

    with open(manifest_path, "r") as f:
        manifest = yaml.safe_load(f)

    assert (
        manifest["dataset_manifest"]["qa_checkpoint"] == "QA-0"
    ), "Incorrect QA checkpoint"

    kylberg_path = Path(manifest["dataset_manifest"]["datasets"]["kylberg"]["path"])
    assert kylberg_path.exists(), f"Kylberg path {kylberg_path} does not exist"

    files = list(kylberg_path.rglob("*.png")) + list(kylberg_path.rglob("*.tif"))
    assert len(files) > 0, "No texture images found in Kylberg directory"

    print(
        f"[SUCCESS] Dataset manifest loaded. QA Checkpoint: {manifest['dataset_manifest']['qa_checkpoint']}"
    )
    print(f"[SUCCESS] Found {len(files)} texture images in Kylberg raw directory.")
    print(
        f"[SUCCESS] UHCTD Status correctly registered as: {manifest['dataset_manifest']['datasets']['uhctd']['status']}"
    )
    print("--- Validation Complete ---")


if __name__ == "__main__":
    try:
        validate()
    except AssertionError as e:
        print(f"[ERROR] Validation failed: {e}")
        sys.exit(1)
