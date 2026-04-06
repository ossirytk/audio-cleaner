"""Copy patches and Hydra configuration into the active virtual environment.

Two things are required after ``uv sync --extra training``:

1. **conf/** — demucs/train.py uses ``config_path="../conf"`` which resolves at
   runtime to ``site-packages/conf/``.  Since demucs is installed as a wheel
   the conf/ directory is not placed there automatically.

2. **patches/** — small Windows-compatibility fixes that cannot be handled as
   version pins.  Currently only ``demucs/repitch.py`` (NamedTemporaryFile
   fix for Windows file-locking behaviour).

Run once after ``uv sync --extra training``, and again whenever conf/ or
patches/ change.

Usage::

    uv run python scripts/apply_patches.py
"""

import shutil
import site
import sys
from pathlib import Path


def main() -> None:
    site_dirs = [Path(p) for p in site.getsitepackages() if "site-packages" in p]
    if not site_dirs:
        print("ERROR: could not locate site-packages", file=sys.stderr)
        sys.exit(1)
    site_packages = site_dirs[0]

    repo_root = Path(__file__).parent.parent

    # Copy conf/ so Hydra can find it at site-packages/conf/
    conf_src = repo_root / "conf"
    if not conf_src.is_dir():
        print(f"ERROR: conf/ directory not found at {conf_src}", file=sys.stderr)
        sys.exit(1)
    shutil.copytree(conf_src, site_packages / "conf", dirs_exist_ok=True)
    print(f"Copied conf/ → {site_packages / 'conf'}")

    # Copy patches/ over the installed packages
    patches_root = repo_root / "patches"
    applied = 0
    if patches_root.is_dir():
        for patch_file in sorted(patches_root.rglob("*.py")):
            rel = patch_file.relative_to(patches_root)
            dest = site_packages / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(patch_file, dest)
            print(f"Patched: {rel}")
            applied += 1

    print(f"\napply-patches complete — {applied} patch(es) + conf/ copied to {site_packages}")


if __name__ == "__main__":
    main()

