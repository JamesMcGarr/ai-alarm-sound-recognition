# How This Project Works — A Beginner's Guide to AI

This document walks through how this alarm sound recognition system works, and uses it as a practical vehicle for learning **Artificial Intelligence (AI)** and **Machine Learning (ML)** from the ground up.

You do not need any prior AI knowledge. Each section introduces a concept in plain English first, then shows you exactly where it appears in the code. Optional **Deep Dive** sections are available if you want to go further.

---

## The Big Picture

In plain English, this project does the following:

> *A microphone listens continuously. When it hears an X-Sense security alarm, it turns on a smart plug (which has a siren connected to it). When the alarm stops, the siren turns off.*

The clever part is that **no one wrote rules to describe what an alarm sounds like**. Instead, the system *learned* what an alarm sounds like by listening to many examples. That learning process is what AI is all about.

The full journey from sound to action looks like this:

```
USB Microphone
  └─ Capture overlapping 1-second audio windows         [src/audio/capture.py]
       └─ Convert audio to a visual "fingerprint"       [src/audio/features.py]
            └─ Feed fingerprint to the neural network   [src/training/model.py]
                 ├─ Confidence ≥ 99.9% → ALARM!
                 │    └─ Turn on smart plug siren       [src/siren/controller.py]
                 └─ Confidence < 99.9% → Not alarm
                      └─ Turn off smart plug siren
```

---

## Section 1: What Is AI / Machine Learning?

### The core idea

Traditional programming works like this:
> *You write rules → the computer follows them.*

For example, you could try: *"If the audio frequency is between 3000 Hz and 4000 Hz for more than 0.5 seconds, it's an alarm."*

The problem is that alarms vary, microphones vary, rooms vary, and background noise varies. Writing rules that cover every case is extremely hard — or even impossible.

Machine Learning flips this around:
> *You show the computer examples → the computer figures out the rules.*

You give it many recordings of the alarm and many recordings of non-alarm sounds, and the algorithm finds patterns you never explicitly wrote down.

### What type of AI is this?

This project uses **supervised learning** — the most common and practical form of ML:

- **Supervised** means the training examples are *labelled*. Each recording is tagged as either "alarm" (label = 1) or "not alarm" (label = 0).
- The task is called **binary classification** — the model produces a single answer: *alarm* or *not alarm*.

<details>
<summary>🔬 Deep Dive: Supervised learning and why binary classification is a good starting point</summary>

### The taxonomy of machine learning

There are three broad flavours of machine learning:

| Type | What it needs | Example use |
|---|---|---|
| **Supervised** | Labelled examples (input + correct answer) | Spam detection, image classification, this project |
| **Unsupervised** | Unlabelled data only | Clustering customers by behaviour, anomaly detection |
| **Reinforcement** | An environment and a reward signal | Game-playing AIs (chess, Go), robotics |

This project uses **supervised learning** because:
1. We know exactly what we want to detect (a specific alarm sound).
2. We can easily create labelled examples (record the alarm, label it "1"; record silence or traffic, label it "0").
3. Supervised classification is the most mature, well-understood area of ML with reliable results.

### Binary vs multi-class classification

Binary classification (2 classes: yes/no) is the simplest classification problem. A single output neuron with a sigmoid activation is all you need — it outputs a probability between 0 and 1. The project's threshold of 0.999 means: only fire the alarm if the model is *at least 99.9% confident* it heard the alarm.

Multi-class classification (e.g., distinguishing between 10 different alarm types) would require 10 output neurons and a softmax activation instead.

### Why not just use a threshold on a specific frequency?

Frequency-based rule systems fail because:
- The same alarm sounds different from different distances, in different rooms, with different microphones.
- Background noise overlaps with alarm frequencies.
- The alarm sweep covers many frequencies over time — no single bin captures it.

A trained CNN, by contrast, learns to recognise the *pattern* of how frequencies change over time — something no simple rule can express.

</details>

---

## Section 2: Collecting Training Data

### Why data matters

An AI model is only as good as the data it learns from. Before any training can happen, you need a **dataset** — a collection of labelled examples.

For this project, the dataset consists of short audio clips:
- **Positive examples** (label = 1): 1-second recordings of the alarm actually sounding.
- **Negative examples** (label = 0): 1-second recordings of everything *else* — silence, speech, traffic, music, domestic noise.

These are stored in:
```
data/
  positive/    ← alarm clips, used for training
  negative/    ← non-alarm clips, used for training
```

### How collection works

The script [`src/training/collector.py`](../src/training/collector.py) handles this. When you run `python train.py`, it:

1. Prompts you to trigger your alarm.
2. Records 60 one-second clips and saves them to `data/positive/`.
3. Prompts you to stop the alarm.
4. Records 60 clips of background sounds and saves them to `data/negative/`.

Each clip is a `.wav` file — a standard audio format that stores uncompressed sound.

### Why equal numbers matter

The default is 60 positive and 60 negative clips. This **balanced dataset** is important. If you trained with 600 alarm clips and only 10 non-alarm clips, the model would learn to always predict "alarm" just to get most answers right — not because it genuinely recognised the sound.

<details>
<summary>🔬 Deep Dive: Data quality, hard negatives, and the retraining loop</summary>

### Data quality > model complexity

A common beginner mistake is to focus on making the model bigger or more complex. In practice, improving your dataset almost always gives better results than tweaking the model.

For this project, the most important data quality questions are:
- **Diversity of negatives**: Do the negative clips cover all the sounds the system will encounter in production? Speech, TV, appliances, outdoor noise? A model that only trains on silence will fail when the TV is on.
- **Diversity of positives**: Does the alarm sound different at different distances? From different angles? The more variation in your positives, the more robust the model.
- **Clip quality**: Are clips recorded at the same sample rate as inference? A mismatch between training audio (44100 Hz) and inference audio would degrade accuracy.

### Hard negatives

A **hard negative** is a non-alarm sound that the model incorrectly classifies as an alarm (a false positive). These are the most valuable training examples — they expose exactly where the model's decision boundary is wrong.

During live inference, [`src/inference/listener.py`](../src/inference/listener.py) automatically saves hard negatives:
- Any non-alarm window with confidence ≥ 10% (`INTERESTING_MIN_CONFIDENCE = 0.1`) is saved to `data/negative_captures/`.
- Any alarm trigger is saved to `data/positive_captures/`.

You can copy these into the training directories and retrain:
```bash
cp data/positive_captures/*.wav data/positive/
cp data/negative_captures/*.wav data/negative/
python train.py --train
```

This **retraining loop** is how the model improves over time in its actual deployment environment. It's a simplified version of a practice called **active learning**.

### Class imbalance and what to do about it

In production, alarm events are rare. If you want to add 1000 hours of background audio to your negative set but only have 60 seconds of alarm recordings, the dataset becomes imbalanced. Strategies include:
- **Oversample the minority class**: Repeat positive clips multiple times.
- **Data augmentation**: Generate new positive examples by modifying existing ones (pitch shifting, time stretching, adding target noise).
- **Weighted loss**: Tell the loss function to penalise false negatives more than false positives.

This project's augmentation pipeline (noise injection + amplitude scaling in [`src/training/dataset.py`](../src/training/dataset.py)) provides a light version of this.

</details>

---

## Section 3: Turning Sound Into Numbers

### The problem: computers cannot hear

A neural network is, at its core, a mathematical function. It takes in numbers and outputs numbers. It cannot process raw audio directly — or rather, it *could*, but doing so would be painfully inefficient and slow to learn.

The key question is: **what numerical representation of sound is most useful for recognising an alarm?**

### Step 1: Raw audio — a sequence of pressure readings

When a microphone records sound, it samples the air pressure thousands of times per second. Each sample is a number (roughly between -1 and +1). A 1-second clip at 44100 Hz is simply an array of 44100 numbers.

This is handled by [`src/audio/capture.py`](../src/audio/capture.py), which streams 1-second windows from the USB microphone with a 0.5-second hop (50% overlap between consecutive windows).

### Step 2: From waveform to spectrogram

A raw waveform is hard to learn from — an alarm sounds the same whether it is slightly louder or quieter, faster or slower. Instead, we transform it into a **spectrogram**: a 2D image showing *which frequencies are present* at *which points in time*.

Think of it like sheet music: instead of a wiggly line, you get a picture where:
- The **horizontal axis** is time.
- The **vertical axis** is frequency (pitch).
- The **brightness** at each point shows how loud that frequency is at that moment.

An alarm has a very distinctive spectrogram pattern — a rising/falling sweep across a specific range of frequencies, repeating rhythmically. That pattern is easy for a CNN to learn.

### Step 3: Mel scale — match human hearing

Not all frequencies are equally important. Human hearing (and most sounds that matter to us) is more sensitive to lower frequencies. The **mel scale** compresses high frequencies and expands low frequencies to match this.

Instead of evenly spaced frequency bins, a mel spectrogram uses 64 unevenly spaced bins (`N_MELS = 64` in [`src/audio/features.py`](../src/audio/features.py)).

### Step 4: Log scale (decibels)

Sound intensity varies enormously — from a whisper to a jet engine. Converting amplitude to **decibels (dB)** using a logarithmic scale compresses this range and better matches how we perceive loudness. The project uses an 80 dB dynamic range (`top_db=80`).

### Step 5: Normalisation

The final step scales the values to approximately the range [-1, 1] using:

```
normalised = (log_mel_dB / 40.0) - 1.0
```

This ensures the input to the neural network is always in a consistent numerical range, which helps training converge faster and more stably.

### The result: a tensor

After all these steps, a 1-second audio clip becomes a PyTorch **tensor** of shape `(1, 64, T)`:
- `1` = one channel (like a greyscale image)
- `64` = number of mel frequency bins (vertical axis)
- `T` = number of time frames (horizontal axis)

This is exactly the format a convolutional neural network expects — just like a single-channel image.

```python
# src/audio/features.py — the full pipeline in ~10 lines
def extract(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> torch.Tensor:
    waveform = torch.from_numpy(audio).unsqueeze(0)   # (1, samples)
    mel = _mel_transform(waveform)                     # (1, 64, T)
    log_mel = _amplitude_to_db(mel)                    # convert to dB
    return log_mel / 40.0 - 1.0                        # normalise to [-1, 1]
```

<details>
<summary>🔬 Deep Dive: FFT, hop length, mel scale mathematics, and why settings must match</summary>

### The Short-Time Fourier Transform (STFT)

A mel spectrogram is built on top of the **Short-Time Fourier Transform (STFT)**. The STFT works by:

1. Taking a short slice ("window") of the audio signal — `N_FFT = 1024` samples wide.
2. Applying a Fourier Transform to that slice to get the frequency content.
3. Sliding the window forward by `HOP_LENGTH = 512` samples (50% overlap).
4. Repeating until the full audio is covered.

The result is a 2D array: (frequency bins) × (time frames).

At 44100 Hz:
- `N_FFT = 1024` → window duration = 1024/44100 ≈ 23 ms (short enough to capture transient detail)
- `HOP_LENGTH = 512` → step = 512/44100 ≈ 11.6 ms between frames

### The mel filterbank

The raw STFT produces 513 frequency bins (N_FFT/2 + 1). A mel filterbank applies 64 overlapping triangular filters (`N_MELS = 64`) to these bins, summing the energy in each. The filters are spaced on the mel scale:

```
mel = 2595 × log₁₀(1 + frequency / 700)
```

This maps 50 Hz–8000 Hz (`F_MIN=50.0`, `F_MAX=8000.0`) onto a perceptually uniform scale.

### Why do these settings need to match between training and inference?

If you train with `N_MELS=64` and then run inference with `N_MELS=128`, the model sees tensors of a completely different shape — it would crash or produce garbage output. The same is true for `SAMPLE_RATE`, `N_FFT`, `HOP_LENGTH`, `F_MIN`, and `F_MAX`. These are **hyperparameters fixed at training time**. Changing them requires retraining from scratch.

### What the normalisation formula does

`torchaudio.transforms.AmplitudeToDB` with `top_db=80` produces values in the range [-80, 0] dB (where 0 dB is the loudest sound in the window). Dividing by 40 and subtracting 1:

```
input range:  [-80, 0]
÷ 40:         [-2, 0]
- 1:          [-3, -1]
```

In practice the useful signal sits closer to the upper end, giving rough coverage of [-1, 1] for typical inputs. The normalisation is not perfect (very quiet sounds can push values below -2), but it is good enough for robust training.

</details>

---

## Section 4: The Neural Network (AlarmCNN)

### What is a neural network?

A neural network is a mathematical function built from layers of simpler operations. Each layer transforms its input slightly, and the layers together learn complex, high-level patterns.

The name comes from a loose analogy with biological neurons in the brain — each artificial "neuron" receives some inputs, combines them, applies a non-linear function, and passes the result on.

The model is defined in [`src/training/model.py`](../src/training/model.py) as `AlarmCNN`.

### Why a CNN for audio?

**Convolutional Neural Networks (CNNs)** were originally designed for images. They work by sliding small filters (kernels) across the input and learning what those filters should detect.

Since our input is a mel spectrogram (effectively a greyscale image), a CNN is a natural choice. Just as a CNN recognises a face whether it's in the top-left or centre of an image, it can recognise an alarm's frequency sweep whether it starts at the beginning or middle of the 1-second window.

### The AlarmCNN architecture, layer by layer

```
Input: (batch, 1, 64, T)   ← mel spectrogram, shape like a greyscale image
  │
  ├─ Conv Block 1: Conv2d(1→32, 3×3) → BatchNorm → ReLU → MaxPool(2×2)
  │    ↓ output: (batch, 32, 32, T/2)
  │
  ├─ Conv Block 2: Conv2d(32→64, 3×3) → BatchNorm → ReLU → MaxPool(2×2)
  │    ↓ output: (batch, 64, 16, T/4)
  │
  ├─ Conv Block 3: Conv2d(64→128, 3×3) → BatchNorm → ReLU → AdaptiveAvgPool(4×4)
  │    ↓ output: (batch, 128, 4, 4)
  │
  ├─ Flatten: (batch, 128×4×4) = (batch, 2048)
  │
  ├─ Linear(2048 → 256) → ReLU → Dropout(0.3)
  │
  ├─ Linear(256 → 1)
  │
  └─ Sigmoid
       ↓
Output: (batch, 1)   ← probability that the audio contains an alarm, e.g. 0.9993
```

### What each layer type does

| Layer | What it does |
|---|---|
| **Conv2d** | Scans the input with small learned filters to detect local patterns (edges, textures, frequency sweeps) |
| **BatchNorm** | Normalises activations within each mini-batch to keep training stable |
| **ReLU** | Sets negative values to zero — introduces non-linearity so the model can learn complex patterns |
| **MaxPool2d** | Shrinks the feature map by taking the maximum value in each 2×2 block — reduces computation and adds some shift-invariance |
| **AdaptiveAvgPool2d(4×4)** | Resizes any feature map to exactly 4×4, regardless of input size — allows variable-length audio |
| **Flatten** | Converts the 3D feature map to a 1D vector so it can be fed into fully-connected layers |
| **Linear** | A traditional "fully connected" layer — every input connected to every output, learns high-level combinations |
| **Dropout(0.3)** | Randomly zeros 30% of neurons during training to prevent over-relying on any single feature (regularisation) |
| **Sigmoid** | Squashes the final number to the range [0, 1], turning it into a probability |

### Reading the output

```python
model.predict_proba(spectrogram)  # → e.g. 0.9997

# Is it an alarm?
model.is_alarm(spectrogram, threshold=0.999)  # → True if prob >= 0.999
```

A value of 0.9997 means: *"I am 99.97% confident this is an alarm."*

<details>
<summary>🔬 Deep Dive: Why these design choices were made</summary>

### Why three convolutional blocks?

Three blocks give the network enough depth to learn hierarchical features:
- **Block 1** learns low-level patterns: edges and simple frequency shapes.
- **Block 2** combines those into mid-level patterns: frequency bands, rhythmic transitions.
- **Block 3** combines those into high-level patterns: the overall alarm signature.

More blocks would increase capacity but risk overfitting on a small dataset (120 training clips). Fewer blocks might not have enough capacity to distinguish the alarm from similar sounds.

### Batch normalisation: the training stabiliser

Without batch normalisation, activations deepen through layers can grow or shrink dramatically, causing training to diverge or stall. BatchNorm rescales activations to have zero mean and unit variance within each mini-batch, then applies learned scaling and shifting. It is now standard in almost every modern CNN.

### Dropout: preventing overfitting

A model with 618k parameters trained on only ~96 clips (80% of 120) could easily **overfit** — memorise the exact training clips rather than learn general alarm features. Dropout helps by randomly disabling 30% of the neurons in the fully-connected layer during each training step, forcing the network to learn redundant representations that generalise better.

### Sigmoid vs Softmax

For **binary** classification (2 classes), a single sigmoid output is equivalent to a 2-class softmax. Sigmoid is simpler: a single output neuron produces one probability `p` (alarm), and `1 - p` is the implied probability of "not alarm."

### Adaptive average pooling

`AdaptiveAvgPool2d(4, 4)` always produces a 4×4 output regardless of input size. This is important here because different sample rates or `HOP_LENGTH` values produce spectrograms of slightly different widths (the time dimension `T`). The adaptive pooling makes the model input-size agnostic.

### Parameter count

The model has ~618k parameters (learnable numbers). This is very small by modern standards — GPT-4 has ~1 trillion. But for a binary audio classifier running in real-time on a Raspberry Pi, it is exactly right: fast enough for live inference, large enough to learn the task.

</details>

---

## Section 5: Training — How the Model Learns

### Learning as repeated refinement

Before training starts, the model's parameters (the numbers inside each layer) are random. The model initially produces random, useless predictions. Training is the process of iteratively adjusting those parameters so predictions improve.

The training loop in [`src/training/trainer.py`](../src/training/trainer.py) repeats the following cycle thousands of times:

```
1. Take a small batch of labelled spectrograms (16 at a time)
2. Pass them through the model → get predicted probabilities
3. Measure how wrong the predictions are (the "loss")
4. Calculate how each parameter contributed to the error (backpropagation)
5. Nudge each parameter slightly in the direction that reduces the loss
6. Repeat
```

### The loss function: Binary Cross-Entropy (BCE)

The **loss function** measures how wrong the model's predictions are. For binary classification, the standard choice is **Binary Cross-Entropy**:

```
BCE = -(y × log(p) + (1-y) × log(1-p))
```

Where:
- `y` is the true label (0 or 1)
- `p` is the model's predicted probability

If the label is 1 (alarm) and the model predicts 0.999, the loss is small. If the model predicts 0.01, the loss is large. This creates a strong gradient pushing the model to predict higher probabilities for alarms.

### The optimiser: Adam

The **Adam optimiser** uses the calculated gradients to update the model parameters. Adam is popular because it adapts the learning rate separately for each parameter — parameters that have been updated a lot get smaller updates, preventing them from overshooting.

The learning rate (`LEARNING_RATE = 0.001`) controls how big each update step is. Too large → training is unstable. Too small → training is painfully slow.

### Train vs validation split

The dataset is split 80% train / 20% validation:
- **Training set** (80%): the model sees these and updates its parameters.
- **Validation set** (20%): the model sees these but does *not* update — used to check if the model generalises.

If training accuracy is 99% but validation accuracy is 60%, the model has **overfit** — it memorised the training clips instead of learning the real pattern.

### Reading the training log

Each epoch prints three numbers. Here is a representative line from a real run:

```
Epoch  3/20 | train_loss=0.0234  train_acc=98.43%  val_acc=100.00% ✓ saved
```

| Metric | Where it comes from | What it tells you |
|---|---|---|
| **train_loss** | Average BCE error across all training batches | How confident and correct the model is on data it *has learned from*. Lower is better; 0.0 would be perfect. |
| **train_acc** | % of training clips classified correctly | The model's score on its own "homework". High early in training is normal; very high very fast may mean the model is memorising. |
| **val_acc** | % of validation clips classified correctly | The model's score on data it has *never seen*. This is the honest measure of whether learning has actually generalised. |

The `✓ saved` marker means this epoch produced the best `val_acc` so far and the model was checkpointed.

**How to interpret the relationship between the three numbers:**

| Scenario | Typical numbers | What it means |
|---|---|---|
| Both train and val accuracy are high | train 99%, val 99% | ✅ Model is learning real patterns |
| Train high, val much lower | train 99%, val 60% | ⚠️ Overfitting — memorised training data; needs more varied samples or dropout |
| Both low | train 60%, val 58% | ⚠️ Underfitting — model isn't learning; may need more epochs or a larger model |
| Val higher than train | train 92%, val 97% | Normal early in training when the val set happens to be easier; usually resolves after more epochs |

Loss and accuracy move in opposite directions: as the model improves, loss goes down and accuracy goes up. If you see loss going *up* while accuracy goes *up*, something is wrong (this rarely happens with a well-configured run).

### Early stopping and model checkpointing

Training stops automatically when validation accuracy reaches `TARGET_ACCURACY = 0.999` (99.9%). The best model seen during training is saved as a dated file (e.g. `models/alarm_model_2026-03-17.pt`) whenever validation accuracy improves. A symlink `models/alarm_model.pt` always points to the latest model so the listener works without reconfiguration.

### Model metadata

Each training run produces a companion JSON file (e.g. `models/alarm_model_2026-03-17_meta.json`) alongside the model weights. This metadata captures everything needed to reproduce or compare models:

| Category | Example fields |
|---|---|
| **Identity** | `model_filename`, `created_date` |
| **Training** | `epochs`, `best_epoch`, `training_duration_seconds`, `target_accuracy_reached` |
| **Data** | `training_samples.total`, `.positive`, `.negative`, `.positive_files`, `.negative_files` |
| **Performance** | `val_accuracy`, `val_loss`, `precision`, `recall`, `f1`, `false_positive_rate`, `false_negative_rate` |
| **Config** | `learning_rate`, `batch_size`, `optimizer`, `loss_function`, `scheduler`, `train_val_split` |
| **Features** | `sample_rate`, `n_mels`, `n_fft`, `hop_length`, `f_min`, `f_max`, `clip_duration_seconds` |
| **Architecture** | `name`, `detection_threshold`, `total_parameters` |
| **Environment** | `python_version`, `torch_version`, `platform` |
| **Free text** | `notes` — editable after training to record context like *"added kitchen samples"* |

### Archiving models

When you train a new model, archive the previous one by copying both files into a dated folder:

```bash
mkdir -p models/archive/2026-03-17
mv models/alarm_model_2026-03-17.pt models/archive/2026-03-17/
mv models/alarm_model_2026-03-17_meta.json models/archive/2026-03-17/
```

The dated folder name matches the model filename, making it easy to find later. Comparing the metadata JSON files between two models reveals exactly what changed — more samples, different hyperparameters, or a different environment.

<details>
<summary>🔬 Deep Dive: Learning rate scheduling, data augmentation, and why 99.9% is the target</summary>

### Learning rate scheduling with ReduceLROnPlateau

Adam starts with `lr = 0.001`. If the validation loss stops improving for 10 consecutive epochs (`patience=10`), the learning rate is halved (`factor=0.5`). This **learning rate decay** helps the model make finer adjustments in later training, squeezing out the last few percentage points of accuracy.

Tracking in the training output:
```
Epoch 45: train_acc=0.9999, val_acc=0.9875, lr=0.0005
```
A reduced LR indicates the model is in its fine-tuning phase.

### Data augmentation: affordable synthetic data

[`src/training/dataset.py`](../src/training/dataset.py) applies random augmentation to every training clip before extracting features:

```python
def _augment(audio: np.ndarray) -> np.ndarray:
    noise = np.random.randn(len(audio)) * np.random.uniform(0.0, 0.005)
    audio = audio + noise                                  # additive noise
    audio = audio * np.random.uniform(0.7, 1.3)           # amplitude scaling
    return audio
```

- **Additive Gaussian noise**: simulates different microphone noise floors and acoustic environments.
- **Amplitude scaling**: simulates the alarm at different distances (closer = louder, further = quieter).

Augmentation is only applied to the training set, not the validation set — the validation set must reflect real-world conditions without artificial modification.

### Why train on the spectrogram, not the raw waveform?

The augmentation is applied to the raw audio *before* feature extraction. This means each epoch produces slightly different spectrograms from the same original clips — effectively tripling or quadrupling your effective dataset size at zero cost.

### Why 99.9% accuracy as the target?

The model runs approximately twice per second (2 Hz, due to the 0.5s hop). Over a 10-minute period it runs ~1200 times. At 99% accuracy, expected false positives in 10 minutes: 1200 × 0.01 = **12 false alarms**. At 99.9%: 1200 × 0.001 = **1.2**. The 99.9% validation accuracy target reduces false alarms by 10× compared to 99%.

Note that validation accuracy is measured on a small set (24 clips). Real-world accuracy depends heavily on deployment conditions. The 99.9% threshold at *inference time* provides an additional layer of safety on top of the trained accuracy.

### The PyTorch DataLoader

The `DataLoader` wraps the dataset and handles:
- **Shuffling**: randomises the order each epoch so the model sees clips in different sequences, preventing it from learning order-based patterns.
- **Batching**: groups 16 clips into a mini-batch for GPU-parallel processing.
- **Collation**: stacks the individual tensors into a batch tensor.

</details>

---

## Section 6: Making Predictions (Inference)

### From training to deployment

After training, the model's parameters are saved to `models/alarm_model.pt`. This file contains the learned "knowledge" — you can load it on any machine and make predictions without retraining.

```python
# src/training/trainer.py
torch.save(model.state_dict(), MODEL_PATH)   # save

# Loading back:
model = AlarmCNN()
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()                                  # switch to inference mode
```

### The live inference loop

[`src/inference/listener.py`](../src/inference/listener.py) runs forever, processing audio windows as they arrive:

```python
for window in audio_capture.stream():          # receive 1-second audio windows
    spectrogram = extract(window)              # convert to mel spectrogram
    confidence = model.predict_proba(spec)     # run the neural network
    if confidence >= DETECTION_THRESHOLD:      # 0.999
        on_alarm_detected(window, confidence)
    else:
        on_alarm_not_detected(window, confidence)
```

### `torch.no_grad()` and eval mode

Two important things happen during inference that do not happen during training:

1. **`model.eval()`** — switches BatchNorm and Dropout to their inference behaviour. BatchNorm uses stored running statistics instead of mini-batch statistics. Dropout is disabled (all neurons are always active).

2. **`torch.no_grad()`** — tells PyTorch not to track gradients (the mathematical machinery needed for backpropagation). This saves memory and makes inference faster.

### The sliding window

Audio is captured in overlapping 1-second windows with a 0.5-second hop. This means:
- **2 predictions per second**: at 0.5s intervals, the system responds to an alarm within ~0.5 seconds.
- **Context continuity**: each window shares 0.5 seconds with the previous one, so a sound that starts near the end of one window is fully captured in the next.

<details>
<summary>🔬 Deep Dive: Confidence thresholding, silence filtering, and the 99.9% decision</summary>

### Confidence thresholding

The model's sigmoid output is a probability — technically the probability that the window contains an alarm, given everything the model has learned. The **detection threshold** (`DETECTION_THRESHOLD = 0.999`) converts this into a binary decision.

Why 99.9% and not 50%? This relates to the **precision/recall trade-off**:

- **Precision**: of all the windows we call "alarm", what fraction really are alarms? (False positives are bad — they cause the siren to fire when there is no alarm.)
- **Recall**: of all the alarm windows, what fraction do we detect? (False negatives are bad — we miss a real alarm.)

A higher threshold improves precision (fewer false positives) at the cost of recall (might miss the very first second of an alarm). For a home security system where false positives are disruptive, this is the right trade-off. The system will catch the alarm after 1–2 windows anyway.

### Silence filtering

Windows with very low amplitude (RMS < `SILENCE_RMS_THRESHOLD = 0.005`) are detected but not saved as "interesting" negatives. This prevents filling the disk with empty silence clips that add no training value.

### The interesting negatives threshold

Windows that score between 0.1 and 0.999 (`INTERESTING_MIN_CONFIDENCE = 0.1`) are saved as negative captures. These represent sounds that *resembled* the alarm enough to register, making them valuable hard negatives for future retraining. Windows below 0.1 are confident non-alarms — not useful.

### Running on CPU vs GPU

PyTorch automatically uses a GPU if one is available (`--device cuda`). For a Raspberry Pi or typical laptop, `--device cpu` is fine — inference on a tiny CNN is fast enough with no GPU needed. A single forward pass through AlarmCNN takes ~1–5 ms on a modern CPU.

</details>

---

## Section 7: Taking Action

### The actuator layer

Once the model says "alarm", *something* must happen. In this project, the action is turning on a TP-Link Tapo smart plug that has a siren plugged into it.

[`src/siren/controller.py`](../src/siren/controller.py) manages this:

```python
siren = SirenController()

siren.turn_on()   # called when alarm detected
siren.turn_off()  # called when no alarm
```

### State tracking: avoid flooding the network

The system detects audio windows at 2 Hz. Without state tracking, it would send a `turn_on` command to the Tapo plug *twice per second* for the entire duration of the alarm — potentially hundreds of HTTP requests.

`SirenController` tracks whether the plug is already on (`_siren_on` flag):

```python
def turn_on(self) -> None:
    if self._siren_on:
        return   ← already on, do nothing
    asyncio.run(self._device.on())
    self._siren_on = True
```

This is a simple but essential design pattern: **idempotent state management**. The real world has state; good software tracks it.

<details>
<summary>🔬 Deep Dive: Async/await, retry logic, and network resilience</summary>

### Why async?

The Tapo library uses Python's `asyncio` framework — it sends HTTP requests to the smart plug and waits for responses without blocking the thread. [`src/siren/tapo_client.py`](../src/siren/tapo_client.py) bridges the synchronous world of the inference loop and the asynchronous world of the network library using `asyncio.run()`.

### The retry strategy

Network requests can fail for many reasons: the plug is momentarily busy, the Wi-Fi drops, the session expires. `TapoDevice._execute_with_retry()` implements a 5-attempt retry strategy with escalating recovery:

| Attempt | Recovery action |
|---|---|
| 1 | Direct call |
| 2 | Refresh Tapo session, retry |
| 3 | Refresh again |
| 4 | Reinitialise device connection, retry |
| 5 | Reinitialise again |

Between attempts: 5-second sleep. This "cascade of recovery" handles transient errors gracefully without crashing the whole inference loop.

### Graceful degradation

In `listen.py`, creating the `SirenController` is wrapped in a `try/except`. If the credentials are missing or the Tapo plug is unreachable, the system continues running without the siren — it still logs alarms, still captures clips, it just cannot actuate. This is good engineering: don't crash the whole system because one component is unavailable.

</details>

---

## Section 7.1: When the Actuator Interferes With the Sensor

### The feedback-loop problem

This project has a classic engineering problem: **the actuator interferes with the sensor**.

The microphone (sensor) detects the X-Sense alarm and turns on the siren (actuator). But the siren is deliberately very loud — loud enough to scare off intruders. That loudness completely overwhelms the microphone. It can no longer hear the quiet X-Sense alarm over the deafening siren.

So what happens?

1. Microphone hears the X-Sense alarm → **siren turns ON.**
2. Siren is so loud the microphone can't hear the X-Sense alarm anymore → model says "no alarm" → **siren turns OFF.**
3. With the siren off, the microphone can hear again → it picks up the X-Sense alarm → **siren turns ON.**
4. Repeat.

The result is rapid on-off-on-off oscillation — the siren flickers rather than holding steady. This is not effective as a deterrent.

This is not unique to this project. It appears in many real-world systems where an actuator's output pollutes the sensor's input:

| Domain | Sensor | Actuator | Interference |
|---|---|---|---|
| This project | Microphone | Loud siren | Siren drowns out the alarm sound |
| Audio systems | Microphone | Speaker | Speaker output feeds back into mic (acoustic feedback / "howling") |
| Heating control | Thermostat | Heater | Heater's heat reaches thermostat before room warms evenly |
| Robotics | Camera | Headlights | Headlights cause glare in camera image |

The general principle: when the thing you *do* (actuate) corrupts the thing you *measure* (sense), naive closed-loop control breaks down.

### The duty-cycle solution

The fix used in this project is a **duty cycle** — a pattern where the actuator operates in timed pulses rather than continuously.

Instead of keeping the siren on indefinitely (which deafens the microphone forever), the listener:

1. **Detects the alarm** → turns the siren **ON**.
2. **Holds the siren on** for a fixed duration (default: 5 seconds) — long enough to be effective.
3. **Turns the siren OFF** → normal listening resumes.
4. If the X-Sense alarm is still going, the microphone detects it again on the next clean audio window and **re-enters the cycle**.
5. If the alarm has stopped, the microphone hears silence and **does nothing** — the system is back to passive monitoring.

```
 ┌──── Alarm detected ◄──────────────────────────┐
 │                                                │
 ▼                                                │
 Siren ON for 5 seconds                           │
 │  (microphone can't hear alarm — that's OK,     │
 │   audio windows naturally score < 0.999)       │
 ▼                                                │
 Siren OFF                                        │
 │                                                │
 ▼                                                │
 Normal listening resumes                         │
 ├── Alarm still audible? ── YES ─────────────────┘
 └── Alarm stopped? ── NO action, keep listening
```

The key insight is that **no explicit "gap" or queue-draining is needed**. While the siren is blaring, the sounddevice callback continues pushing audio windows into the queue. Those siren-contaminated windows are processed as usual, but they don't match the X-Sense alarm's acoustic signature — the model scores them well below the 99.9% threshold. They are simply ignored. Once the siren turns off and the microphone can hear normally again, the very next clean window either re-triggers the cycle or doesn't.

This is controlled by a single configurable value:

```bash
# CLI argument
python listen.py --siren-on-duration 5

# Or via .env
SIREN_ON_DURATION=5.0
```

### Where it lives in the code

[`src/inference/listener.py`](../src/inference/listener.py) — the `run()` method orchestrates the duty cycle:

```python
for window in self.capture.stream():
    alarm_detected = self._process_window(window)
    if alarm_detected and self.siren is not None:
        self._siren_duty_cycle()
```

And `_siren_duty_cycle()` is intentionally simple:

```python
def _siren_duty_cycle(self) -> None:
    self.siren.turn_on()
    time.sleep(self.siren_on_duration)
    self.siren.turn_off()
```

After `_siren_duty_cycle()` returns, the `for` loop picks up the next audio window from the stream and the cycle either repeats or doesn't — no special logic required.

<details>
<summary>🔬 Deep Dive: Duty cycles in engineering and alternative approaches</summary>

### Duty cycles

A **duty cycle** is the fraction of time a system is in an active state. In electronics, a 50% duty cycle means a signal is on half the time and off half the time (like a square wave). PWM (Pulse-Width Modulation) — used in LED dimming, motor speed control, and power supplies — is a direct application of duty cycling.

In this project, with a 5-second on-duration and roughly 1 second for detection after the siren turns off, the effective duty cycle is approximately 5/6 ≈ 83%. The siren is sounding 83% of the time during an active alarm — more than enough to be effective as a deterrent.

### Alternative approaches considered

**1. Acoustic Echo Cancellation (AEC)**
Professional audio systems (speakerphones, hearing aids) use AEC algorithms to subtract the known speaker output from the microphone input. This would let the microphone "hear through" the siren. However, AEC requires precise knowledge of the siren's audio signal and the room's acoustic response — far more complexity than this project warrants.

**2. Separate frequency bands**
If the siren operated exclusively in frequencies the X-Sense alarm doesn't use, a bandpass filter could isolate the alarm. In practice, both the siren and the X-Sense alarm cover broad frequency ranges that overlap significantly.

**3. A second microphone further from the siren**
Physical isolation can reduce interference, but adds hardware complexity and still doesn't fully solve the problem — the siren is designed to fill the entire space with sound.

**4. Fixed minimum on-time with hysteresis**
Rather than a simple timer, you could require N consecutive "no alarm" windows before turning the siren off (hysteresis). This was considered but adds complexity without clear benefit — the duty-cycle approach is simpler and equally effective because the normal listening loop already provides the checking mechanism.

The duty-cycle approach was chosen for its simplicity: one parameter, four lines of code, no signal processing, and it leverages the existing ML model's natural inability to confuse siren noise with the X-Sense alarm.

</details>

---

## Section 8: The Continuous Improvement Loop

### AI models are not "finished"

A trained model is a snapshot of what it learned from a particular dataset at a particular time. In the real world:
- Background noise changes (new appliances, seasons, street activity).
- The microphone might be repositioned.
- A new type of alarm might be installed.

This is called **concept drift** — the relationship between the input data and the correct output changes over time.

### How this project handles it

The live inference loop automatically saves two types of audio captures:

| Directory | Contents | Purpose |
|---|---|---|
| `data/positive_captures/` | Every window that triggered the alarm (≥ 99.9% confidence) | Confirms the model is working; adds diversity to positives |
| `data/negative_captures/` | Interesting non-alarm windows (10%–99.9% confidence) | Hard negatives — sounds the model almost confused for an alarm |

**The retraining workflow:**
```bash
cp data/positive_captures/*.wav data/positive/
cp data/negative_captures/*.wav data/negative/
python train.py --train
sudo systemctl restart ai-alarm-listener
```

This is a simplified version of an **MLOps pipeline** — the practice of continuously maintaining ML models in production.

<details>
<summary>🔬 Deep Dive: Hard-negative mining, concept drift, and the ML lifecycle</summary>

### Hard-negative mining

In ML, a **hard negative** is a negative example (label = 0) that the model incorrectly or nearly-incorrectly classifies as positive. These examples represent the model's weakness — the boundary case where it almost gets confused.

Training on hard negatives is highly efficient: adding 20 hard negatives to your training set can do more good than adding 200 easy negatives (silence, white noise) because they directly target the model's failure modes.

This project automatically mines hard negatives during deployment. Every time the model hedges (confidence 10%–99.9%) on a non-alarm sound, that sound is saved for potential training use.

### Concept drift

Concept drift is when the real-world distribution of your data changes after deployment. Examples for this project:
- **Covariate drift**: the acoustic environment changes (new furniture, windows open in summer, moved microphone). Input audio looks different, but the labelling rule is the same.
- **Label drift**: unlikely here, but in general, when what "counts" as positive changes.

Monitoring for concept drift: if you notice the model triggers frequently at unusual times, or stops triggering when the alarm actually sounds, it is time to retrain.

### A real-world MLOps pipeline

This project demonstrates the core of the ML lifecycle:

```
Data Collection → Preprocessing → Training → Evaluation → Deployment
       ↑                                                        │
       └──────────────────── Monitoring & Retraining ──────────┘
```

Professional MLOps tools (MLflow, Weights & Biases, Kubeflow) automate and scale this loop, but the underlying cycle is identical.

</details>

---

## Section 9: Putting It All Together

### End-to-end data flow

```
USB Microphone
  │  (hardware: ~44100 samples/sec, float32, mono)
  ▼
AudioCapture.stream()                          src/audio/capture.py
  │  (1-second windows, 0.5s hop)
  ▼
features.extract()                             src/audio/features.py
  │  STFT → Mel filterbank → log scale → normalise
  │  Output: tensor (1, 64, ~87)  ← mel spectrogram
  ▼
AlarmCNN.predict_proba()                       src/training/model.py
  │  3× [Conv → BN → ReLU → Pool] → FC → Sigmoid
  │  Output: float in [0, 1]
  ▼
Threshold comparison                           src/inference/listener.py
  │  confidence >= 0.999?
  ├── YES → on_alarm_detected()
  │           ├─ Save .wav to data/positive_captures/
  │           └─ Siren duty cycle (if configured):
  │                SirenController.turn_on()   src/siren/controller.py
  │                  └─ TapoDevice.on()        src/siren/tapo_client.py
  │                       └─ HTTP to Tapo P110
  │                sleep(siren_on_duration)
  │                SirenController.turn_off()
  │                  └─ resume listening
  └── NO  → on_alarm_not_detected()
              └─ If interesting → save to data/negative_captures/
```

### AI/ML concepts in this project

| Concept | Where in the code |
|---|---|
| Supervised binary classification | Overall project goal |
| Labelled dataset collection | [src/training/collector.py](../src/training/collector.py) |
| Audio feature extraction (mel-spectrograms) | [src/audio/features.py](../src/audio/features.py) |
| PyTorch `Dataset` and `DataLoader` | [src/training/dataset.py](../src/training/dataset.py) |
| Data augmentation | [src/training/dataset.py](../src/training/dataset.py) — `_augment()` |
| Train/validation split | [src/training/dataset.py](../src/training/dataset.py) — `make_splits()` |
| Convolutional Neural Network | [src/training/model.py](../src/training/model.py) — `AlarmCNN` |
| Batch normalisation | [src/training/model.py](../src/training/model.py) |
| Dropout regularisation | [src/training/model.py](../src/training/model.py) |
| Binary Cross-Entropy loss | [src/training/trainer.py](../src/training/trainer.py) |
| Adam optimiser | [src/training/trainer.py](../src/training/trainer.py) |
| Learning rate scheduling | [src/training/trainer.py](../src/training/trainer.py) — `ReduceLROnPlateau` |
| Model checkpointing | [src/training/trainer.py](../src/training/trainer.py) |
| Inference with `torch.no_grad()` | [src/training/model.py](../src/training/model.py) — `predict_proba()` |
| Confidence thresholding | [src/inference/listener.py](../src/inference/listener.py) |
| Actuator-sensor interference / duty cycle | [src/inference/listener.py](../src/inference/listener.py) — `_siren_duty_cycle()` |
| Hard-negative mining | [src/inference/listener.py](../src/inference/listener.py) |
| Real-time streaming inference | [src/inference/listener.py](../src/inference/listener.py) |
| Production deployment (systemd) | [install.sh](../install.sh) |
| Continuous retraining loop | [README.md](../README.md) — retraining section |

### Suggested next steps for learning

1. **Visualise a spectrogram**: Load one of your `.wav` clips in Python and plot the mel spectrogram using `matplotlib`. See what the alarm "looks" like.

2. **Train on your own sound**: Replace the alarm clips with a doorbell, a baby cry, or your name being called. The code needs zero changes — just different `data/positive/` clips.

3. **Inspect the model's confidence histogram**: Log all confidence values during a listening session and plot a histogram. You will likely see a bimodal distribution: most values near 0 (definite non-alarms) and some near 1 (alarms), with very few in between. This is a well-trained classifier.

4. **Experiment with the threshold**: Lower `DETECTION_THRESHOLD` to 0.9 and observe whether false-positive rate increases. This directly demonstrates the precision/recall trade-off.

5. **Try multi-class classification**: Add a third class (e.g., smoke alarm) and modify the model output from `Linear(256→1) + Sigmoid` to `Linear(256→3) + Softmax`.

6. **Read the PyTorch documentation**: This project covers roughly 30% of what PyTorch can do. The [official PyTorch tutorials](https://pytorch.org/tutorials) are an excellent next step.

---

## Glossary

| Term | Definition |
|---|---|
| **Accuracy** | Percentage of predictions the model gets right |
| **Activation function** | A non-linear function (e.g., ReLU, Sigmoid) applied after a layer to let the network learn complex patterns |
| **Adam** | A popular gradient-based optimiser that adapts the learning rate for each parameter |
| **Augmentation** | Artificially modifying training data (noise, flipping, scaling) to increase effective dataset size and reduce overfitting |
| **Backpropagation** | The algorithm that calculates how much each model parameter contributed to the loss, enabling gradient-based updates |
| **Batch normalisation** | Normalises layer outputs within a mini-batch to stabilise and speed up training |
| **BCE (Binary Cross-Entropy)** | A loss function for binary classification problems |
| **Binary classification** | A task with exactly two possible output classes (alarm / not alarm) |
| **CNN (Convolutional Neural Network)** | A type of neural network designed for grid-structured inputs (images, spectrograms) using sliding filter operations |
| **Concept drift** | When the real-world data distribution changes after a model is deployed, degrading its performance |
| **Dataset** | A collection of labelled examples used for training and evaluation |
| **Duty cycle** | Operating an actuator in timed on/off pulses rather than continuously, often to allow a sensor to take readings between pulses |
| **DataLoader** | A PyTorch utility that batches, shuffles, and feeds a Dataset to the training loop |
| **Dropout** | A regularisation technique that randomly disables neurons during training to prevent overfitting |
| **Epoch** | One complete pass through the entire training dataset |
| **Feature extraction** | Converting raw data (audio waveform) into a numerical representation (mel spectrogram) useful for ML |
| **Gradient** | Measure of how much the loss would change if a parameter were nudged slightly — used to update parameters |
| **Hard negative** | A negative example that a model finds difficult to classify correctly — very valuable for training |
| **Hyperparameter** | A configuration value set before training (learning rate, batch size, N_MELS) as opposed to a parameter learned during training |
| **Inference** | Using a trained model to make predictions on new data (as opposed to training) |
| **Label** | The correct answer associated with a training example (1 = alarm, 0 = not alarm) |
| **Learning rate** | Controls how large each parameter update step is during optimisation |
| **Loss function** | A mathematical measure of how wrong the model's predictions are — minimised during training |
| **Mel spectrogram** | A 2D representation of audio showing frequency content over time, scaled to match human hearing |
| **Mini-batch** | A small subset of the training data (e.g., 16 examples) processed together per training step |
| **Model** | A mathematical function with learnable parameters; in this project, `AlarmCNN` |
| **Model checkpointing** | Saving the model weights whenever performance improves, to preserve the best version |
| **Neural network** | A layered mathematical function loosely inspired by biological neurons |
| **Overfitting** | When a model memorises training data rather than learning general patterns, causing poor performance on new data |
| **Parameter** | A learnable number inside the model (weights and biases) that is adjusted during training |
| **Precision** | Of all positive predictions, what fraction were correct |
| **Recall** | Of all actual positives, what fraction were detected |
| **Regularisation** | Techniques (dropout, weight decay) that prevent overfitting |
| **ReLU** | Rectified Linear Unit: `f(x) = max(0, x)` — a simple but effective activation function |
| **Sigmoid** | `f(x) = 1/(1+e^{-x})` — squashes any number to [0,1], used as final layer for binary classifiers |
| **Spectrogram** | A visual representation of the spectrum of frequencies in a signal as it varies over time |
| **Supervised learning** | ML where training examples are labelled with the correct answer |
| **Tensor** | A multi-dimensional array (generalisation of matrix) — the fundamental data structure in PyTorch |
| **Training** | The process of adjusting model parameters to minimise loss on labelled examples |
| **Validation set** | A held-out subset of data used to measure model performance during training without leaking into parameter updates |
