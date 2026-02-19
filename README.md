# AI Alarm Sound Recognition

A Python application that listens to a USB microphone, uses a trained neural network to recognise an X-Sense security alarm sound, and fires a callback when the alarm is detected.

## How it works

The app is split into three logical modules:

| Module | Purpose |
|---|---|
| `src/audio/` | USB mic capture and mel-spectrogram feature extraction |
| `src/training/` | Sample collection, dataset, model definition, training loop |
| `src/inference/` | Live inference loop and alarm callback |

**Architecture:** A compact CNN (`AlarmCNN`, ~618k parameters) classifies 1-second overlapping audio windows as *alarm* or *not alarm*.  Each window is converted to a 64-band log-mel-spectrogram before being fed into the model.  The model fires when confidence ≥ **99%**.

```
USB mic
  └─ AudioCapture (sounddevice, 22 050 Hz, 1-second windows, 0.5 s hop)
       └─ features.extract() → log-mel-spectrogram tensor (1 × 64 × T)
            └─ AlarmCNN.predict_proba()
                 ├─ ≥ 0.99 → on_alarm_detected() + save to data/positive_captures/
                 └─ < 0.99 (but interesting) → save to data/negative_captures/
```

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
│   └── alarm_model.pt          # Saved model weights (created after training)
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
3. **Training** — the model trains until validation accuracy reaches 99%, then saves weights to `models/alarm_model.pt`.

#### Flags

| Flag | Description |
|---|---|
| `--collect` | Collect samples only (skip training) |
| `--train` | Train only (samples already collected) |
| `--positive N` | Number of positive clips to record (default: 60) |
| `--negative N` | Number of negative clips to record (default: 60) |
| `--epochs N` | Maximum training epochs (default: 200) |
| `--device INDEX\|NAME` | sounddevice device index or name fragment |

If 99% accuracy is not reached, record more samples with `python train.py --collect` and retrain with `python train.py --train`.

---

### 2. Live inference

```bash
python listen.py
```

Loads `models/alarm_model.pt` and streams audio continuously.  When the alarm is detected at ≥ 99% confidence, `on_alarm_detected()` is called.

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
| `--threshold FLOAT` | Detection confidence threshold (default: 0.99) |
| `--model PATH` | Path to a `.pt` model file (default: `models/alarm_model.pt`) |
| `--no-save-triggers` | Disable saving alarm triggers to `data/positive_captures/` |
| `--no-save-negatives` | Disable saving interesting non-alarm frames to `data/negative_captures/` |
| `--verbose` | Enable DEBUG logging |

---

## Retraining with captured samples

During live inference the app automatically saves:

- **`data/positive_captures/`** — every window that triggers the alarm at ≥ 99% confidence.
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

## Tests

```bash
python -m pytest tests/ -v
```

All tests run without a microphone or a trained model.
