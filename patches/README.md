# Patches

This directory contains minimal Windows-compatibility fixes for installed packages.

Run after `uv sync --extra training`:

```powershell
uv run python scripts/apply_patches.py
```

---

## `patches/demucs/repitch.py`

> **License note:** This file is vendored from the upstream
> [Demucs](https://github.com/facebookresearch/demucs) project by Meta Platforms, Inc.
> and affiliates, and is covered by the upstream Demucs license (MIT).
> The original LICENSE file is not included alongside this patch; consult the
> [upstream repository](https://github.com/facebookresearch/demucs/blob/main/LICENSE) for
> full license terms.

**Problem:** On Windows, `tempfile.NamedTemporaryFile()` holds an exclusive lock on the
file while it is open. When `soundstretch` and `soundfile` then try to open the same
file by name, they get "System error: Access denied".

**Fix:** Use `delete=False`, close the handle immediately after creation, then delete
the files manually in a `finally` block.

