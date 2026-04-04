# Data Flow — SignSpeak

## End-to-End Data Pipeline

### Stage 1: Raw Capture (Flutter Client)
```
Native Device Camera (Android/iOS/Web)
   ↓  Capture Frame
   ↓  Encode to JPEG (quality compressed)
   ↓  Send via WebSocket (binary)
```

### Stage 2: Backend Decoding & Pose Estimation (FastAPI)
```
WebSocket Receiver (server.py)
   ↓  cv2.imdecode → BGR uint8
   ↓  cv2.cvtColor → RGB
MediaPipe Hands Processing
   ↓  Hand detection + landmark regression
Landmarks: List[NormalizedLandmark] (21 per hand)
   ↓  Extract .x, .y, .z
Raw Array: np.float32 (42, 3) — 21 landmarks × 2 hands × 3 coords
```

### Stage 3: Normalization
```
Raw Landmarks (42, 3)
   ↓  Split by hand: hand_0 = [0:21], hand_1 = [21:42]
   ↓  Subtract wrist (landmark 0) from each hand's 21 landmarks
   ↓  hand_i[j] = hand_i[j] - hand_i[0]  ∀ j ∈ [0, 20]
Wrist-Relative (42, 3) — wrist is now (0, 0, 0)
   ↓  .flatten()
Normalized Vector: np.float32 (126,)
```

### Stage 4: Sequence Buffering
```
Frame vectors: (126,) × N frames
   ↓  SequenceBuffer.push() — appends to rolling list
   ↓  (Note: frames duplicated locally to match 30 FPS training temporal ratio)
   ↓  Sliding window keeps latest 60 frames
   ↓  Zero-pads if < 60 frames collected
Sequence Tensor: np.float32 (60, 126)
```

### Stage 5: Model Inference
```
Input: (1, 60, 126) — batch dimension added
   ↓  LSTM Layer 1: (1, 60, 128) — return_sequences=True
   ↓  Dropout(0.3)
   ↓  LSTM Layer 2: (1, 64) — return_sequences=False
   ↓  Dropout(0.3)
   ↓  Dense(64, ReLU): (1, 64)
   ↓  Dropout(0.3)
   ↓  Dense(N, Softmax): (1, N) — probability per class
Output: Probability vector (N,)
   ↓  argmax → predicted class
   ↓  max → confidence score
```

### Stage 6: Prediction Filtering
```
(predicted_class, confidence)
   ↓  Confidence ≥ threshold? (default 0.7)
   │     No → discard
   ↓  Same as last prediction?
   │     Yes → debounce counter < limit? → discard
   ↓  Accept prediction
   ↓  Bundle Result: { prediction: label, confidence: score, buffer_ready: true }
```

### Stage 7: Output (Flutter UI)
```
JSON Response received via WebSocket
   ↓  Flutter state updates
   ↓  Confidence bars animate
   ↓  Sentence builder accumulates predicted words
   ↓  Display in Material 3 UI
   ↓  (Optional) On-device TTS triggered
```

---

## Tensor Shape Summary

| Stage | Shape | Type | Size |
|---|---|---|---|
| JPEG Payload | N/A | bytes | ~10-30 KB |
| Decoded BGR | (480, 640, 3) | uint8 | ~900 KB |
| Raw landmarks | (42, 3) | float32 | 504 B |
| Normalized vector | (126,) | float32 | 504 B |
| Sequence buffer | (60, 126) | float32 | ~30 KB |
| Model input | (1, 60, 126) | float32 | ~30 KB |
| Model output | (1, N) | float32 | 4N B |

---

## Performance Optimization Strategies

1. **No image storage** — Only 504 bytes per frame vs ~900 KB pixel data
2. **MediaPipe GPU delegate** — Offloads hand detection to GPU if available
3. **Sliding window** — No redundant recomputation of entire sequences
4. **TF Lite conversion** — 2-4× inference speedup for deployment
5. **Batch normalization** — Stable gradient flow during training
6. **Frame skipping** — Process every 2nd frame if needed (30→15 FPS)

---

## Error Handling Strategy

| Scenario | Handling |
|---|---|
| WebSocket Disconnect | Reconnection logic in Flutter app, clear buffers on backend |
| No hands detected | Zero-fill landmarks, continue buffering |
| Model not found | Start backend in safe mode, return `/health` warning |
| Low confidence | Silently discard, do not send update over WebSocket |
| Camera Permission Denied | Show dialog in mobile app directing to system settings |
| Corrupt data file | Skip with warning during dataset build |
