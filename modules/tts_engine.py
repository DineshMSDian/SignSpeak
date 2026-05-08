"""
Module 6: Text-to-Speech Engine
────────────────────────────────
Thread-safe TTS wrapper using pyttsx3 for offline speech synthesis.
"""

import threading
import pyttsx3

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

class TTSEngine:
    """
    Text-to-Speech engine with toggle, rate control, and thread safety.

    Uses pyttsx3 for fully offline, cross-platform speech synthesis.
    All speak calls are non-blocking (run in a background thread).
    """

    def __init__(
        self,
        enabled: bool = config.TTS_ENABLED_DEFAULT,
        rate: int = config.TTS_RATE,
    ):
        self._enabled = enabled
        self._rate = rate
        self._lock = threading.Lock()
        self._speaking = False

    def _create_engine(self):
        """Create a fresh pyttsx3 engine (must be done per-thread)."""
        engine = pyttsx3.init()
        engine.setProperty("rate", self._rate)
        return engine

    def speak(self, text: str):
        """
        Speak the given text in a background thread (non-blocking).

        Args:
            text: String to speak. Ignored if TTS is disabled or empty.
        """
        if not self._enabled or not text or not text.strip():
            return

        if self._speaking:
            return

        thread = threading.Thread(target=self._speak_worker, args=(text,), daemon=True)
        thread.start()

    def _speak_worker(self, text: str):
        """Worker thread for TTS — creates engine, speaks, then cleans up."""
        with self._lock:
            self._speaking = True
            try:
                engine = self._create_engine()
                engine.say(text)
                engine.runAndWait()
                engine.stop()
            except Exception as e:
                print(f"[TTS ERROR] {e}")
            finally:
                self._speaking = False

    def toggle(self) -> bool:
        """
        Toggle TTS on/off.

        Returns:
            New enabled state.
        """
        self._enabled = not self._enabled
        return self._enabled

    def set_enabled(self, enabled: bool):
        """Explicitly set enabled state."""
        self._enabled = enabled

    def set_rate(self, rate: int):
        """
        Set speech rate (words per minute).

        Args:
            rate: Speech speed, typically 100–200 WPM.
        """
        self._rate = max(50, min(rate, 300))

    @property
    def is_enabled(self) -> bool:
        """Whether TTS is currently enabled."""
        return self._enabled

    @property
    def is_speaking(self) -> bool:
        """Whether the engine is currently speaking."""
        return self._speaking

if __name__ == "__main__":
    print("[TEST] Text-to-Speech Engine")

    tts = TTSEngine(enabled=True, rate=150)
    assert tts.is_enabled is True
    print("  ✓ TTS initialized and enabled")

    tts.toggle()
    assert tts.is_enabled is False
    print("  ✓ Toggle OFF works")

    tts.toggle()
    assert tts.is_enabled is True
    print("  ✓ Toggle ON works")

    tts.set_rate(180)
    print("  ✓ Rate set to 180 WPM")

    print("  → Speaking test phrase...")
    tts.speak("Sign language translator initialized successfully")

    import time
    time.sleep(3)

    print("[PASS] TTS engine test complete!")
