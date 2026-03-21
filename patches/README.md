# Patches

This directory contains minimal Windows-compatibility fixes for installed packages.

Run after `uv sync --extra training`:

```powershell
uv run python scripts/apply_patches.py
```

---

## `patches/demucs/repitch.py`

**Problem:** On Windows, `tempfile.NamedTemporaryFile()` holds an exclusive lock on the
file while it is open. When `soundstretch` and `soundfile` then try to open the same
file by name, they get "System error: Access denied".

**Fix:** Use `delete=False`, close the handle immediately after creation, then delete
the files manually in a `finally` block.

