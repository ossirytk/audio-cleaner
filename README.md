# audio-cleaner

Train and apply a custom **[HDemucs](https://github.com/facebookresearch/demucs)** model to
automatically remove radio station idents, jingles, and advertisements from audio recordings.

## How it works

HDemucs is a **source separation** model — it learns to split a mixed audio signal into separate
components called *stems*. This project repurposes it by training it to recognise jingles as one
stem and background music as another.

For training, the model needs paired examples of:
- **Background music** — clean music clips without any announcements
- **Jingle audio** — isolated recordings of just the idents or announcements you want removed

From these two ingredients the training pipeline synthetically creates thousands of *mixtures*
(music with a jingle overlaid at a random position and volume), then trains the model to undo the
mixing. Once trained, you can feed it any new recording and it will output a clean version with
jingles removed.

> **Don't have isolated jingle recordings?** See
> [Getting isolated jingle stems](#getting-isolated-jingle-stems) below.

---

## Quickstart for new users

> This section is written for people who are not very familiar with programming or the command
> line. It walks through everything from scratch. Experienced users can skip straight to
> [Prerequisites](#prerequisites).

### What you will need

- A **Windows PC** (Windows 10 or 11)
- An internet connection to download tools
- Your audio files — music clips and jingle recordings

### Step A — Install the required tools

You need to install three small programs before you can use audio-cleaner. Each installation takes
about a minute.

**1. Install Python**

Go to [python.org/downloads](https://www.python.org/downloads/) and download the latest
**Python 3.12** installer. Run it, and on the first screen make sure to tick
**"Add Python to PATH"** before clicking Install.

To check it worked, open **PowerShell** (press `Win + R`, type `powershell`, press Enter) and type:
```
python --version
```
You should see something like `Python 3.12.x`.

**2. Install uv** (the package manager this project uses)

In the same PowerShell window, paste this line and press Enter:
```powershell
powershell -ExecutionPolicy Bypass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
Close and re-open PowerShell when it finishes so the new tool is available.

**3. Install FFmpeg** (needed for reading audio files)

In PowerShell, paste and press Enter:
```powershell
winget install --id Gyan.FFmpeg
```
If `winget` is not available, download the FFmpeg build from
[ffmpeg.org/download.html](https://ffmpeg.org/download.html), extract it somewhere (e.g.
`C:\ffmpeg`), and add its `bin` folder to your system PATH.

---

### Step B — Get the audio-cleaner files

If you received audio-cleaner as a ZIP file, extract it to a folder on your computer — for
example `C:\audio-cleaner`.

If you use Git, clone the repository instead:
```powershell
git clone https://github.com/your-org/audio-cleaner.git C:\audio-cleaner
```

---

### Step C — Open a terminal in the project folder

Open **PowerShell** and navigate to the folder where you extracted or cloned audio-cleaner:
```powershell
cd C:\audio-cleaner
```
*(Replace `C:\audio-cleaner` with the actual path if you used a different location.)*

---

### Step D — Install project dependencies

Run these two commands one after the other. Each one may take a few minutes the first time:
```powershell
uv sync --extra dev --extra training --extra web
uv run python scripts/apply_patches.py
```
You should see a message saying the patches were applied. If you see any red error text, check
that Python and uv were installed correctly in Step A.

---

### Step E — Start the web server

In the same PowerShell window, run:
```powershell
uv run uvicorn web_app:app --host 127.0.0.1 --port 8000 --reload
```

You will see some lines of text ending with something like:
```
INFO:     Application startup complete.
```
**Leave this window open.** The server keeps running for as long as this window is open. If you
close it, the web interface will stop working.

---

### Step F — Open the web interface

Open your web browser (Chrome, Edge, Firefox — any will work) and go to:
```
http://127.0.0.1:8000
```

You should see the audio-cleaner dashboard. 🎉

---

### Step G — Tell the app where your data is

The first time you open the app you will see a message saying the data directory is not
configured. Click **Settings** in the left sidebar and set the **Data directory** to the folder
where you want to keep your audio files — for example `D:\jingle_removal`. Click **Save**, and
the page will reload automatically.

---

### Step H — Use the pipeline

The left sidebar has five sections:

| Section | What it does |
|---------|-------------|
| **Dashboard** | Overview of your data and any running job |
| **Dataset** | Prepare music clips and jingle files; build the training dataset |
| **Training** | Start/stop model training; watch live progress |
| **Inference** | Upload a recording and run it through your trained model to clean it |
| **Settings** | Change the data directory path |

Work through **Dataset → Training → Inference** in order. Each section has buttons and forms —
you do not need to use the command line again unless something goes wrong.

---

### Stopping and restarting the server

To stop the server, go back to the PowerShell window and press `Ctrl + C`.

To start it again, run the same command as Step E:
```powershell
uv run uvicorn web_app:app --host 127.0.0.1 --port 8000 --reload
```

---

## Web UI

The web interface (`web_app.py`) is a browser-based frontend for the full pipeline. It provides
forms and live log streaming for all pipeline steps so you do not need to remember long command
lines.

### Starting the server

**From the terminal (any platform):**
```powershell
uv run uvicorn web_app:app --host 127.0.0.1 --port 8000 --reload
```
Then open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser. The `--reload` flag
means the server restarts automatically whenever you edit Python or template files — useful
during development.

**From VS Code:**

Open the Command Palette (`Ctrl + Shift + P`) → **Tasks: Run Task** → pick one of:

| Task | What it does |
|------|-------------|
| **Start web server (Windows)** | Launches uvicorn with `--reload` in a dedicated terminal panel |
| **Stop web server (Windows)** | Kills the process listening on port 8000 |
| **Restart web server (Windows)** | Stop + Start in sequence |

These tasks are defined in `.vscode/tasks.json` and work without any extra setup once the
dependencies are installed.

### Installing web dependencies

The web server requires a small set of extra packages. Install them with:
```powershell
uv sync --extra web
```
Or install everything at once (recommended):
```powershell
uv sync --extra dev --extra training --extra web
```

### Configuration

The app reads `JINGLE_BASE_DIR` from a `.env` file in the project root (or from the environment).
You can set this through the **Settings** page in the UI — it writes `.env` automatically and
reloads the server. Alternatively, create `.env` by hand:
```
JINGLE_BASE_DIR=D:\my_audio_data
```

---

## Prerequisites

- Windows (primary), Linux, or macOS
- Python 3.12
- [uv](https://github.com/astral-sh/uv) package manager (installed in Step 1)
- [FFmpeg](https://ffmpeg.org/) — required for audio decoding. Install via:
  ```powershell
  winget install --id Gyan.FFmpeg
  ```
- An NVIDIA GPU with at least 8 GB VRAM is strongly recommended — CPU training works but is
  extremely slow

## Data layout

All data lives under a single base directory, defaulting to `I:\jingle_removal\`. You can change
this by setting the `JINGLE_BASE_DIR` environment variable before running any command.
```powershell
$env:JINGLE_BASE_DIR
```

```
I:\jingle_removal\
├── music_sources\             # Raw music files (FLAC) used as background content
├── music_sources_cassettes\   # Optional second folder of raw music files
├── music_clips\               # 40-second WAV clips prepared from the sources above
├── jingles_original\          # Raw jingle / ident recordings (unprocessed)
├── jingles_processed\         # Normalised versions of the same jingles
├── training_dataset\          # Assembled training data (created automatically)
│   ├── train\<track>\         # drums.wav  bass.wav  other.wav  vocals.wav  mixture.wav
│   └── valid\<track>\         # same structure, used for validation during training
├── outputs\                   # Model checkpoints saved during training
├── test_audio\mixture.wav     # A recording you want to clean (used in Step 6)
└── separation_results\        # Cleaned audio output from Step 6
```

## Step-by-step workflow

### Step 1 — Install dependencies

```powershell
# Install uv (skip if already installed)
powershell -ExecutionPolicy Bypass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Install all project dependencies and apply the required patches
uv sync --extra dev --extra training
uv run python scripts/apply_patches.py
```

**For GPU-accelerated training** (strongly recommended), replace the default CPU version of
PyTorch, then re-apply the patches:

```powershell
uv pip install torch torchaudio --extra-index-url https://download.pytorch.org/whl/cu126
uv run python scripts/apply_patches.py
```

> Re-run `uv run python scripts/apply_patches.py` any time you run `uv sync`, as syncing can
> overwrite the patched files.

### Step 2 — Prepare music clips

Place your source music files (FLAC format) into `I:\jingle_removal\music_sources\`, then run:

```powershell
uv run python -m scripts.create_samples
```

This slices each file into a random 40-second clip, resamples to 44.1 kHz stereo, normalises
the level, and saves the result to `music_clips\`.

> **Already done?** If you already have 40-second WAV clips ready, place them directly in
> `music_clips\` and skip this step.

### Step 3 — Prepare isolated jingle audio

You need recordings of just the jingle or announcement on its own — without music playing
underneath. See [Getting isolated jingle stems](#getting-isolated-jingle-stems) if you only have
a mixed broadcast recording.

Once you have isolated jingle files:

- Copy them to `jingles_original\` (raw, unprocessed versions)
- Copy them to `jingles_processed\` (or a normalised/cleaned-up version if you have one)

Using the same files in both folders is fine as a starting point.

### Step 4 — Build the training dataset

```powershell
uv run python -m scripts.generate_dataset
```

This mixes the jingles into the music clips at random positions and volumes, producing the
`training_dataset\` folder with train and validation splits.

### Step 5 — Train the model

Before training, install the CUDA-enabled version of PyTorch for GPU acceleration
(replace `cu126` with your CUDA version if different):

```powershell
uv pip install torch==2.8.0+cu126 torchaudio==2.8.0+cu126 `
    --extra-index-url https://download.pytorch.org/whl/cu126
```

Then start training:

```powershell
uv run dora --package demucs run model=hdemucs dset=musdb44 `
    epochs=200 `
    ++batch_size=2 `
    ++misc.num_workers=0 `
    ++optim.lr=0.0002 `
    "++weights=[0.1,0.1,1.0,5.0]" `
    ++augment.remix.group_size=2 `
    +name=HIFI_HDEMUCS `
    ++test.every=999 `
    ++dset.use_musdb=false `
    dset.wav=D:\data\training_dataset
```

Training progress is printed to the console. Checkpoints are saved to `outputs\`. Expect this to
take several hours on a GPU depending on dataset size.

> **Note on `++test.every=999`:** the test evaluation step requires the official musdb
> benchmark dataset, which we don't use. Setting this to a value larger than `epochs`
> disables it. Validation (loss on your own held-out clip) still runs every epoch.

**Resuming a stopped run:** at the start of training, a short signature code is printed (e.g.
`f88645aa`). Pass it to resume from the last checkpoint by appending `continue_from=<code>`:

```powershell
uv run dora --package demucs run model=hdemucs dset=musdb44 `
    epochs=200 `
    ++batch_size=2 `
    ++misc.num_workers=0 `
    ++optim.lr=0.0002 `
    "++weights=[0.1,0.1,1.0,5.0]" `
    ++augment.remix.group_size=2 `
    +name=HIFI_HDEMUCS `
    ++test.every=999 `
    ++dset.use_musdb=false `
    dset.wav=D:\data\training_dataset `
    continue_from=f88645aa
```

### Step 6 — Clean a recording

Place the audio file you want to clean at `I:\jingle_removal\test_audio\mixture.wav`, then run:

```powershell
uv run python -m scripts.separate_audio
```

Results are saved to `separation_results\`. The file you want is **`result_other.wav`** — the
background music with jingles removed. `result_vocals.wav` contains the isolated jingle audio if
you want to inspect what was removed.

---

## Getting isolated jingle stems

The training pipeline requires isolated jingle audio — just the announcement sound on its own,
without background music underneath. If you only have a broadcast recording where the jingle is
mixed into music, you need to extract it first. There are three approaches:

### Option A — Use a pre-trained separation model (recommended starting point)

Run your broadcast recording through the stock HDemucs model to get a rough vocal/speech stem:

```powershell
uv run python -m demucs.separate --name htdemucs "I:\path\to\your\broadcast.wav"
```

Output is saved to a `separated\htdemucs\` folder next to the input file. The `vocals.wav` stem
will contain most of the speech and announcements. The separation will not be perfect, but you
can manually clean it up in an audio editor.

### Option B — Manual editing

Open your broadcast recording in **[Audacity](https://www.audacityteam.org/)** (free) or a DAW
such as Reaper. Find moments where only the announcement plays — for example, over a pause or a
fade — and export those segments as a new file. Even a few seconds of clean announcement audio
is usable.

### Option C — Find or re-record the original jingle

Radio station idents are often professionally produced audio packages. Check whether the station
publishes them, or see if a clean version is available elsewhere. This gives the highest-quality
stems and the best training results.

---

## Patches

`patches/` contains small fixes to the upstream `demucs` package that are required for this
project to work correctly on Windows. They are applied automatically as part of Step 1 above
(`uv run python scripts/apply_patches.py`).

See [`patches/README.md`](patches/README.md) for the full change list.

## Development

```powershell
uv run ruff check scripts patches           # check code style
uv run ruff check --fix scripts patches     # auto-fix style issues
uv run ruff format scripts patches          # format files
```

See [AGENTS.md](AGENTS.md) for AI coding-agent guidelines.
