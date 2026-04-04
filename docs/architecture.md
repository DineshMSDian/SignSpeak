# System Architecture — SignSpeak

## Real-Time Continuous Hybrid Sign Language Translation System

### Architecture Overview

```
┌──────────────────────────────────────┐     ┌─────────────────────────────────────────────────────┐
│          Client (Flutter)            │     │                 Backend (FastAPI)                   │
│                                      │     │                                                     │
│  ┌─────────────┐     ┌────────────┐  │     │  ┌──────────────┐     ┌─────────────────────┐       │
│  │   Webcam    │────▶│ WebSocket  │──┼─────┼─▶│   OpenCV     │────▶│  MediaPipe Hands    │       │
│  │  (Camera 0) │     │   Sender   │  │     │  │  BGR Frame   │     │  0-2 Hand Detection │       │
│  └─────────────┘     └────────────┘  │     │  └──────────────┘     └─────────┬───────────┘       │
│                                      │     │                                 │                   │
│                                      │     │                       21 landmarks × 2 hands        │
│                                      │     │                           (42, 3) array             │
│                                      │     │                                 │                   │
│                                      │     │                                 ▼                   │
│                                      │     │                      ┌──────────────────┐           │
│                                      │     │                      │   Skeletal        │           │
│                                      │     │                      │   Normalization   │           │
│                                      │     │                      │   (Wrist-Rel.)    │           │
│                                      │     │                      └────────┬─────────┘           │
│                                      │     │                               │                   │
│                                      │     │                        126-dim vector               │
│                                      │     │                               │                   │
│                                      │     │                               ▼                   │
│                                      │     │                      ┌──────────────────┐           │
│                                      │     │                      │  Sequence Buffer  │           │
│                                      │     │                      │  (60 frames)      │           │
│                                      │     │                      │  Sliding Window   │           │
│                                      │     │                      └────────┬─────────┘           │
│                                      │     │                               │                   │
│                                      │     │                        (60, 126) tensor             │
│                                      │     │                               │                   │
│                                      │     │                               ▼                   │
│                                      │     │                      ┌──────────────────┐           │
│  ┌────────────┐      ┌────────────┐  │     │                      │   LSTM Model      │           │
│  │    UI      │      │ WebSocket  │  │     │                      │   LSTM(128)→64    │           │
│  │  (Flutter) │◀─────│  Receiver  │◀─┼─────┼──────────────────────│   Dense→Softmax   │           │
│  └────────────┘      └────────────┘  │     │                      └──────────────────┘           │
└──────────────────────────────────────┘     └─────────────────────────────────────────────────────┘
```

---

## Module Descriptions

### Module 1: Capture Engine (`capture_engine.py`)
- Wraps OpenCV + MediaPipe Hands
- Extracts 21 landmarks per hand (up to 2 hands)
- Zero-fills missing hand data
- Returns annotated frames with landmark overlay

### Module 2: Normalization Engine (`normalization.py`)
- Wrist-relative position normalization
- Flattens to 126-dim vector per frame
- SequenceBuffer manages 60-frame sliding window

### Module 3: Dataset Manager (`dataset_manager.py`)
- Structured storage: `data/raw/{gesture}/sequence_XXXX.npy`
- Stratified 80/20 train-test split
- Label map persistence (JSON)

### Module 4: LSTM Model (`model.py`)
- `Input(60, 126) → LSTM(128) → LSTM(64) → Dense(64) → Dense(N, softmax)`
- Dropout regularization (0.3)
- Callbacks: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

### Module 5: Prediction Engine (`predictor.py`)
- Confidence threshold filtering
- Debounce suppression for repeated labels
- SentenceBuilder accumulation

### Module 6: TTS Engine (`tts_engine.py`)
- pyttsx3 offline synthesis
- Thread-safe, non-blocking speak
- Toggle on/off, configurable rate

---

## Design Decisions

| Decision | Rationale |
|---|---|
| **Landmarks vs CNN** | 126-dim vectors are ~1000× smaller than image tensors; CPU-efficient |
| **Wrist normalization** | Position/translation invariance without data augmentation |
| **LSTM vs Transformer** | Lower parameter count, proven on short sequences, CPU-friendly |
| **60-frame window** | ~2 sec at 30 FPS — adequate for most dynamic gestures |
| **Debounce logic** | Prevents flooding sentence with repeated predictions |
| **FastAPI Backend** | Decouples heavy MediaPipe/LSTM workload from the mobile UI. |
| **Flutter Mobile App** | Cross-platform, native camera support, high performance UI. |

---

## Future Scalability

1. **Transformer backbone** — Replace LSTM with lightweight Transformer for longer sequences
2. **Multilingual TTS** — Integrate gTTS or Coqui for regional language output
3. **Pose + Face landmarks** — Extend to full upper-body sign language
4. **Cloud deployment** — Containerize with Docker, deploy FastAPI on AWS/GCP
5. **Edge deployment** — Export to TFLite for local Android/iOS inference without network overhead
6. **Federated learning** — Privacy-preserving distributed training
