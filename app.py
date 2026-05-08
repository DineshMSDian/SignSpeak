import os
import sys
import json
import time
import cv2
import numpy as np
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from modules.capture_engine import HandCapture
from modules.normalization import normalize_landmarks, SequenceBuffer
from modules.predictor import ContinuousPredictor
from modules.tts_engine import TTSEngine
from modules.model import load_trained_model

st.set_page_config(
    page_title="SignSpeak — Sign Language Translator",
    page_icon="🤟",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    .stApp { font-family: 'Inter', sans-serif; }

    .main-header {
        text-align: center; padding: 0.5rem 0 0.2rem;
        background: linear-gradient(135deg,
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        font-size: 2.4rem; font-weight: 800; letter-spacing: -0.5px;
    }
    .sub-header {
        text-align: center; color:
        margin-bottom: 1rem; font-weight: 400;
    }
    .mode-badge {
        display: inline-block; padding: 0.3rem 1rem; border-radius: 20px;
        font-size: 0.8rem; font-weight: 700; letter-spacing: 1px; text-transform: uppercase;
    }
    .mode-asl { background:
    .mode-isl { background:
    .prediction-box {
        background: linear-gradient(135deg,
        border: 2px solid
        text-align: center; margin: 0.5rem 0;
        box-shadow: 0 8px 32px rgba(99, 102, 241, 0.15);
    }
    .prediction-text {
        font-size: 2.5rem; font-weight: 800; color:
        text-transform: uppercase; letter-spacing: 2px;
        text-shadow: 0 0 20px rgba(99, 102, 241, 0.5);
    }
    .prediction-label {
        font-size: 0.75rem; color:
        letter-spacing: 3px; margin-bottom: 0.2rem; font-weight: 600;
    }
    .confidence-text { font-size: 0.85rem; color:
    .sentence-box {
        background: linear-gradient(135deg,
        border: 1px solid
        margin: 0.5rem 0; min-height: 50px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
    }
    .sentence-label {
        color:
        letter-spacing: 2px; font-weight: 600; margin-bottom: 0.3rem;
    }
    .sentence-text { color:
    .sentence-empty { color:
    .word-chip {
        display: inline-block; background:
        padding: 0.2rem 0.6rem; border-radius: 8px; margin: 0.1rem 0.15rem;
        font-size: 0.9rem; font-weight: 500; border: 1px solid
    }
    [data-testid="stSidebar"] { display: none; }
</style>
""", unsafe_allow_html=True)

def init_session_state():
    defaults = {
        "model": None,
        "label_map": None,
        "model_loaded": False,
        "loaded_mode": None,
        "tts_engine": None,
        "running": False,
        "current_prediction": "—",
        "current_confidence": 0.0,
        "sentence_words": [],
        "frames_processed": 0,
        "mode": "ASL",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session_state()

@st.cache_resource
def load_model_for_mode(mode):
    """Load the model and label map for the given mode."""
    if mode == "ASL":
        model_path = config.ASL_MODEL_PATH
        label_path = config.ASL_LABEL_MAP_PATH
    else:
        model_path = config.ISL_MODEL_PATH
        label_path = config.ISL_LABEL_MAP_PATH

    model = load_trained_model(model_path)
    label_map = None
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            label_map = json.load(f)
    return model, label_map

st.markdown('<h1 class="main-header">🤟 SignSpeak</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Real-Time Sign Language Translator</p>', unsafe_allow_html=True)

mode = st.radio(
    "Select Sign Language",
    options=["ASL", "ISL"],
    horizontal=True,
    index=0 if st.session_state.mode == "ASL" else 1,
    help="ASL = A-Z Letters  •  ISL = Gestures",
)

if mode != st.session_state.mode:
    st.session_state.mode = mode
    st.session_state.current_prediction = "—"
    st.session_state.current_confidence = 0.0
    st.session_state.loaded_mode = None

st.session_state.mode = mode

if mode == "ASL":
    st.markdown('<span class="mode-badge mode-asl">🔤 ASL — Alphabet A-Z</span>', unsafe_allow_html=True)
else:
    st.markdown('<span class="mode-badge mode-isl">🤲 ISL — Gesture Recognition</span>', unsafe_allow_html=True)

model, label_map = load_model_for_mode(mode)

if model is not None and label_map is not None:
    st.session_state.model = model
    st.session_state.label_map = label_map
    st.session_state.model_loaded = True
    st.session_state.loaded_mode = mode
    st.success(f"✅ {mode} model loaded — {len(label_map)} classes: {', '.join(sorted(label_map.keys()))}")
else:
    st.session_state.model_loaded = False
    st.error(f"❌ No {mode} model found. Run `python scripts/train_split.py --{mode.lower()}` first.")
    st.stop()

if st.session_state.tts_engine is None:
    st.session_state.tts_engine = TTSEngine(enabled=True)

st.markdown("---")
start_col, stop_col = st.columns(2)
with start_col:
    start_btn = st.button("▶️ Start Camera", use_container_width=True, type="primary")
with stop_col:
    stop_btn = st.button("⏹️ Stop Camera", use_container_width=True)

if stop_btn:
    st.session_state.running = False

camera_placeholder = st.empty()
prediction_placeholder = st.empty()

st.markdown("---")

words = st.session_state.sentence_words
if words:
    chips_html = " ".join(f'<span class="word-chip">{w}</span>' for w in words)
    sentence_str = " ".join(words)
    st.markdown(f"""
    <div class="sentence-box">
        <div class="sentence-label">📝 Your Sentence</div>
        <div>{chips_html}</div>
        <div class="sentence-text" style="margin-top: 0.4rem; border-top: 1px solid #334155; padding-top: 0.4rem;">
            "{sentence_str}"
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="sentence-box">
        <div class="sentence-label">📝 Your Sentence</div>
        <div class="sentence-empty">Words will appear here as you sign...</div>
    </div>
    """, unsafe_allow_html=True)

col_speak, col_undo, col_clear = st.columns(3)
with col_speak:
    speak_btn = st.button("🔊 Speak Sentence", use_container_width=True, type="primary",
                           disabled=(len(words) == 0))
with col_undo:
    undo_btn = st.button("↩️ Undo", use_container_width=True, disabled=(len(words) == 0))
with col_clear:
    clear_btn = st.button("🗑️ Clear", use_container_width=True, disabled=(len(words) == 0))

if speak_btn and words:
    sentence_str = " ".join(words)
    if st.session_state.tts_engine:
        st.session_state.tts_engine.speak(sentence_str)

if undo_btn and words:
    st.session_state.sentence_words.pop()
    st.rerun()

if clear_btn:
    st.session_state.sentence_words = []
    st.session_state.current_prediction = "—"
    st.session_state.current_confidence = 0.0
    st.rerun()

if not st.session_state.running and not start_btn:
    conf_pct = st.session_state.current_confidence * 100
    prediction_placeholder.markdown(f"""
    <div class="prediction-box">
        <div class="prediction-label">Last Prediction</div>
        <div class="prediction-text">{st.session_state.current_prediction}</div>
        <div class="confidence-text">Confidence: {conf_pct:.0f}%</div>
    </div>
    """, unsafe_allow_html=True)

if start_btn and st.session_state.model_loaded:
    st.session_state.running = True

    predictor = ContinuousPredictor(
        model=st.session_state.model,
        label_map=st.session_state.label_map,
        confidence_threshold=config.CONFIDENCE_THRESHOLD,
    )

    buf = SequenceBuffer()
    hc = HandCapture()

    if not hc.start():
        st.error("❌ Failed to open webcam!")
        st.session_state.running = False
    else:
        frame_count = 0
        DISPLAY_WIDTH = 640
        PREDICT_EVERY_N = 2
        JPEG_QUALITY = 70

        try:
            while st.session_state.running:
                landmarks, annotated, num_hands = hc.capture_and_extract()
                if annotated is None:
                    continue

                frame_count += 1

                normalized = normalize_landmarks(landmarks)
                buf.push(normalized)

                cv2.putText(annotated, f"Mode: {mode}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(annotated, f"Buffer: {buf.current_length}/{config.SEQUENCE_LENGTH}",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                if buf.is_ready() and frame_count % PREDICT_EVERY_N == 0:
                    seq = buf.get_sequence()
                    label, conf, raw_label = predictor.predict(seq)

                    st.session_state.current_confidence = conf

                    if label is not None:
                        st.session_state.current_prediction = label

                        st.session_state.sentence_words.append(label)

                        if st.session_state.tts_engine:
                            display_name = label.replace("_", " ")
                            st.session_state.tts_engine.speak(display_name)

                        cv2.putText(annotated, f">> {label.upper()} ({conf:.0%})",
                                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

                    conf_pct = conf * 100
                    display_label = label if label else raw_label
                    conf_color = "#34d399" if conf >= config.CONFIDENCE_THRESHOLD else "#f87171"
                    prediction_placeholder.markdown(f"""
                    <div class="prediction-box">
                        <div class="prediction-label">🔴 LIVE — {mode} Prediction</div>
                        <div class="prediction-text">{display_label}</div>
                        <div class="confidence-text" style="color: {conf_color}">Confidence: {conf_pct:.0f}%</div>
                    </div>
                    """, unsafe_allow_html=True)

                h, w = annotated.shape[:2]
                if w > DISPLAY_WIDTH:
                    scale = DISPLAY_WIDTH / w
                    annotated = cv2.resize(annotated, (DISPLAY_WIDTH, int(h * scale)))

                frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                _, jpeg_buf = cv2.imencode('.jpg', frame_rgb, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                camera_placeholder.image(jpeg_buf.tobytes(), channels="RGB", use_container_width=True)

        except Exception as e:
            st.error(f"Error: {e}")
        finally:
            hc.stop()
            st.session_state.running = False
