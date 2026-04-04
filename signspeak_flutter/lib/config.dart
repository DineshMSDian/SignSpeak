/// SignSpeak Configuration Constants
library;

// ─── Sequence Parameters (for display only) ────────────────
const int sequenceLength = 60;

// ─── TTS ───────────────────────────────────────────────────
const double ttsRate = 0.5;
const bool ttsEnabledDefault = true;

// ─── Default Server ────────────────────────────────────────
const String defaultServerIp = '192.168.1.100';
const int defaultServerPort = 8000;

// ─── Supported Modes ──────────────────────────────────────
enum SignLanguageMode {
  asl('ASL', 'American Sign Language', '🔤 ASL — Alphabet A-Z'),
  isl('ISL', 'Indian Sign Language', '🤲 ISL — Gesture Recognition');

  final String code;
  final String fullName;
  final String badge;
  const SignLanguageMode(this.code, this.fullName, this.badge);
}
