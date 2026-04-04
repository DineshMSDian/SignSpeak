import 'package:flutter_tts/flutter_tts.dart';
import '../config.dart' as app_config;

class TtsService {
  final FlutterTts _tts = FlutterTts();
  bool _enabled = app_config.ttsEnabledDefault;
  bool _isSpeaking = false;

  bool get isEnabled => _enabled;
  bool get isSpeaking => _isSpeaking;

  TtsService() {
    _init();
  }

  Future<void> _init() async {
    await _tts.setLanguage('en-US');
    await _tts.setSpeechRate(app_config.ttsRate);
    await _tts.setVolume(1.0);
    await _tts.setPitch(1.0);

    _tts.setStartHandler(() => _isSpeaking = true);
    _tts.setCompletionHandler(() => _isSpeaking = false);
    _tts.setCancelHandler(() => _isSpeaking = false);
    _tts.setErrorHandler((msg) {
      _isSpeaking = false;
    });
  }

  Future<void> speak(String text, {String language = 'en-US'}) async {
    if (!_enabled || text.trim().isEmpty || _isSpeaking) return;
    await _tts.setLanguage(language);
    await _tts.speak(text);
  }

  bool toggle() {
    _enabled = !_enabled;
    if (!_enabled) _tts.stop();
    return _enabled;
  }

  void setEnabled(bool enabled) {
    _enabled = enabled;
    if (!_enabled) _tts.stop();
  }

  Future<void> dispose() async {
    await _tts.stop();
  }
}
