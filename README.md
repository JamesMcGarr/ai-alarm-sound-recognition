# AI Alarm Sound Recognition

A Python application that listens to a USB microphone, uses a trained neural network to recognise an X-Sense security alarm sound, and fires a callback when the alarm is detected.

## How it works

The app is split into three logical modules:

| Module | Purpose |
|---|---|
| `src/audio/` | USB mic capture and mel-spectrogram feature extraction |
| `src/training/` | Sample collection, dataset, model definition, training loop |
| `src/inference/` | Live inference loop and alarm callback |

**Architecture:** A compact CNN (`AlarmCNN`, ~618k parameters) classifies 1-second overlapping audio windows as *alarm* or *not alarm*.  Each window is converted to a 64-band log-mel-spectrogram before being fed into the model.  The model fires when confidence ≥ **99.9%**.

```
USB mic
  └─ AudioCapture (sounddevice, 22 050 Hz, 1-second windows, 0.5 s hop)
       └─ features.extract() → log-mel-spectrogram tensor (1 × 64 × T)
            └─ AlarmCNN.predict_proba()
                 ├─ ≥ 0.999 → on_alarm_detected() + save to data/positive_captures/
                 └─ < 0.999 (but interesting) → save to data/negative_captures/
```

**Want to learn how this works?**  See [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md) for a beginner-friendly guide to every AI/ML concept used in this project — from audio feature extraction and neural networks to training, inference, and the continuous improvement loop.

---

## Project layout

```
.
├── train.py                    # Entrypoint: collect samples + train
├── listen.py                   # Entrypoint: live inference
├── src/
│   ├── audio/
│   │   ├── capture.py          # Microphone streaming
│   │   └── features.py         # Audio → mel-spectrogram
│   ├── training/
│   │   ├── collector.py        # Interactive sample recorder
│   │   ├── dataset.py          # PyTorch Dataset
│   │   ├── model.py            # AlarmCNN definition
│   │   └── trainer.py          # Training loop
│   └── inference/
│       └── listener.py         # Live detection loop + on_alarm_detected()
├── data/
│   ├── positive/               # Alarm .wav clips (training)
│   ├── negative/               # Background .wav clips (training)
│   ├── positive_captures/      # Alarm triggers saved during inference
│   └── negative_captures/      # Interesting non-alarm frames saved during inference
├── models/
│   ├── alarm_model.pt          # Symlink → latest dated model
│   ├── alarm_model_YYYY-MM-DD.pt       # Saved model weights
│   ├── alarm_model_YYYY-MM-DD_meta.json # Training metadata
│   └── archive/                # Previous models + metadata
├── tests/
│   ├── test_features.py
│   ├── test_dataset.py
│   └── test_model.py
└── pyproject.toml
```

---

## Requirements

- Python 3.13+
- A USB microphone (the default ALSA device used in `record_sample.sh` is `plughw:2,0`)
- PortAudio (required by `sounddevice`) — install with `sudo apt install portaudio19-dev` on Debian/Ubuntu

---

## Setup

```bash
sudo apt-get install libportaudio2
python -m venv .venv
source .venv/bin/activate
pip install torch torchaudio numpy sounddevice scipy
```

---

## Usage

### 1. Collect samples and train

```bash
python train.py
```

This runs the full workflow:

1. **Positive samples** — you are prompted to trigger your X-Sense alarm.  The app records 60 one-second clips and saves them to `data/positive/`.
2. **Negative samples** — you are prompted to let the alarm stop.  The app records 60 clips of ambient background sounds and saves them to `data/negative/`.
3. **Training** — the model trains until validation accuracy reaches 99.9%, then saves a dated model (e.g. `models/alarm_model_2026-03-17.pt`) and updates the `models/alarm_model.pt` symlink.

#### Flags

| Flag | Description |
|---|---|
| `--collect` | Collect samples only (skip training) |
| `--train` | Train only (samples already collected) |
| `--positive N` | Number of positive clips to record (default: 60) |
| `--negative N` | Number of negative clips to record (default: 60) |
| `--epochs N` | Maximum training epochs (default: 200) |
| `--device INDEX\|NAME` | sounddevice device index or name fragment |

If 99.9% accuracy is not reached, record more samples with `python train.py --collect` and retrain with `python train.py --train`.

---

### 2. Live inference

```bash
python listen.py
```

Loads `models/alarm_model.pt` and streams audio continuously.  When the alarm is detected at ≥ 99.9% confidence, `on_alarm_detected()` is called.

#### Implement your response

Open `src/inference/listener.py` and fill in the `on_alarm_detected` function:

```python
def on_alarm_detected(audio_window: np.ndarray, confidence: float) -> None:
    # TODO: implement your response here
    # e.g. send a notification, trigger a relay, log to a file, etc.
    pass
```

#### Flags

| Flag | Description |
|---|---|
| `--device INDEX\|NAME` | sounddevice device index or name fragment |
| `--threshold FLOAT` | Detection confidence threshold (default: 0.999) |
| `--model PATH` | Path to a `.pt` model file (default: `models/alarm_model.pt`) |
| `--no-save-triggers` | Disable saving alarm triggers to `data/positive_captures/` |
| `--no-save-negatives` | Disable saving interesting non-alarm frames to `data/negative_captures/` |
| `--verbose` | Enable DEBUG logging |

---

## Retraining with captured samples

During live inference the app automatically saves:

- **`data/positive_captures/`** — every window that triggers the alarm at ≥ 99.9% confidence.
- **`data/negative_captures/`** — non-silent, interesting windows that scored below the threshold (useful hard negatives).

Copy these into the training directories and retrain:

```bash
cp data/positive_captures/*.wav data/positive/
cp data/negative_captures/*.wav data/negative/
python train.py --train
```

---

## Changing the USB microphone

By default `sounddevice` uses the system default input device.  To target a specific USB microphone, pass its device index or name fragment:

```bash
python train.py --device 2
python listen.py --device 2
```

List available devices:

```bash
python -c "import sounddevice; print(sounddevice.query_devices())"
```

---

## Privacy

The trained model (`.pt` file) and its metadata (`.json`) are version controlled so others can use this project without re-recording samples and retraining from scratch.

**Are my recordings recoverable from the model?** No. The `.pt` file contains only ~618k learned weight values — abstract numbers representing patterns, not audio. While *model inversion attacks* are a known area of ML research, extracting meaningful audio from a small binary CNN like `AlarmCNN` is not practically feasible. At best an attacker could recover a vague spectral pattern ("something that sweeps between certain frequencies"), not intelligible speech or identifiable home sounds. The risk is accepted.

The metadata JSON includes the **filenames** of the training clips (e.g. `data/positive/positive_0001.wav`) but **not their audio content**. The actual `.wav` files in `data/` are git-ignored and never committed.

---

## Tests

```bash
python -m pytest tests/ -v
```

All tests run without a microphone or a trained model.
