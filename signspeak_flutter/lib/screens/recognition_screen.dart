import 'dart:async';
import 'dart:typed_data';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:image/image.dart' as img;
import '../config.dart';
import '../services/backend_service.dart';
import '../services/gemini_service.dart';
import '../services/tts_service.dart';

const List<List<int>> kHandConnections = [
  [0, 1],
  [1, 2],
  [2, 3],
  [3, 4],
  [0, 5],
  [5, 6],
  [6, 7],
  [7, 8],
  [0, 9],
  [9, 10],
  [10, 11],
  [11, 12],
  [0, 13],
  [13, 14],
  [14, 15],
  [15, 16],
  [0, 17],
  [17, 18],
  [18, 19],
  [19, 20],
  [5, 9],
  [9, 13],
  [13, 17],
];
const Set<int> kFingertips = {4, 8, 12, 16, 20};

class HandSkeletonPainter extends CustomPainter {
  final List<List<Map<String, double>>> hands;
  final Size imageSize;

  const HandSkeletonPainter({required this.hands, this.imageSize = Size.zero});

  @override
  void paint(Canvas canvas, Size size) {
    if (hands.isEmpty) return;

    Offset toCanvas(Map<String, double> lm) {
      return Offset((1.0 - lm['y']!) * size.width, lm['x']! * size.height);
    }

    final bonePaint = Paint()
      ..color = Colors.white.withValues(alpha: 0.9)
      ..strokeWidth = 2.5
      ..strokeCap = StrokeCap.round
      ..style = PaintingStyle.stroke;

    final jointPaint = Paint()
      ..color = const Color(0xFFFF3C3C)
      ..style = PaintingStyle.fill;

    final tipPaint = Paint()
      ..color = const Color(0xFFFFD700)
      ..style = PaintingStyle.fill;

    final tipRing = Paint()
      ..color = Colors.white.withValues(alpha: 0.7)
      ..strokeWidth = 1.8
      ..style = PaintingStyle.stroke;

    for (final hand in hands) {
      if (hand.length < 21) continue;

      final wrist = toCanvas(hand[0]);
      final mid = toCanvas(hand[12]);
      final span = (mid - wrist).distance;
      final tipRadius = (span * 0.13).clamp(5.0, 14.0);
      final jointRadius = (span * 0.08).clamp(3.0, 9.0);

      for (final c in kHandConnections) {
        canvas.drawLine(toCanvas(hand[c[0]]), toCanvas(hand[c[1]]), bonePaint);
      }
      for (int i = 0; i < hand.length; i++) {
        final o = toCanvas(hand[i]);
        if (kFingertips.contains(i)) {
          canvas.drawCircle(o, tipRadius, tipPaint);
          canvas.drawCircle(o, tipRadius, tipRing);
        } else {
          canvas.drawCircle(o, jointRadius, jointPaint);
        }
      }
    }
  }

  @override
  bool shouldRepaint(HandSkeletonPainter old) => true;
}

class RecognitionScreen extends StatefulWidget {
  final SignLanguageMode mode;
  final BackendService backend;

  const RecognitionScreen({
    super.key,
    required this.mode,
    required this.backend,
  });

  @override
  State<RecognitionScreen> createState() => _RecognitionScreenState();
}

class _RecognitionScreenState extends State<RecognitionScreen>
    with WidgetsBindingObserver {
  late final BackendService _backend;
  final TtsService _tts = TtsService();

  CameraController? _cameraController;
  List<CameraDescription>? _cameras;
  bool _isCameraRunning = false;
  bool _isCameraInitializing = false;
  bool _isDisposing = false;

  late SignLanguageMode _mode;
  String _currentPrediction = '—';
  double _currentConfidence = 0.0;
  int _bufferProgress = 0;
  int _detectedHands = 0;
  String? _errorMessage;
  List<List<Map<String, double>>> _currentLandmarks = [];

  final List<String> _tokens = [];
  TranslationResult _translation = TranslationResult.empty;
  bool _isTranslating = false;
  bool _showTranslation = false;
  String _selectedTranslationLang = 'en';

  String? _stableLabel;
  int _stableCount = 0;
  bool _justLocked = false;
  static const int _stableThreshold = 10;



  int _frameSkip = 0;
  bool _isSending = false;

  StreamSubscription<PredictionResult>? _predSub;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _backend = widget.backend;
    _mode = widget.mode;
    _backend.setMode(_mode.code);
    _init();
  }

  Future<void> _init() async {
    try {
      _cameras = await availableCameras();
      _predSub = _backend.predictions.listen(_onPrediction);
      _startCamera();
    } catch (e) {
      if (mounted) setState(() => _errorMessage = 'Init error: $e');
    }
  }

  void _onPrediction(PredictionResult result) {
    if (!mounted) return;

    if (result.numHands == 0) {
      setState(() {
        _currentLandmarks = [];
        _currentPrediction = '—';
        _stableLabel = null;
        _stableCount = 0;
        _justLocked = false;
      });
      return;
    }

    setState(() {
      _detectedHands = result.numHands;
      _bufferProgress = result.bufferFill;
      _currentConfidence = result.confidence;
      _currentLandmarks = result.landmarks;
    });

    final label = result.prediction;
    if (label == null) {
      setState(() {
        _stableLabel = null;
        _stableCount = 0;
      });
      return;
    }
    setState(() => _currentPrediction = label);

    if (label == _stableLabel) {
      _stableCount++;
    } else {
      _stableLabel = label;
      _stableCount = 1;
      _justLocked = false;
    }

    if (_stableCount >= _stableThreshold && !_justLocked) {
      final lastToken = _tokens.isNotEmpty ? _tokens.last : null;
      if (label != lastToken) {
        _commitToken(label);
      }
      setState(() {
        _justLocked = true;
        _stableCount = 0;
      });
    } else {
      setState(() {});
    }
  }

  void _commitToken(String token) {
    setState(() {
      _tokens.add(token);
      _showTranslation = false;
      _translation = TranslationResult.empty;
    });
    _tts.speak(token.replaceAll('_', ' '));
  }


  Future<void> _translate() async {
    if (_tokens.isEmpty) return;
    setState(() => _isTranslating = true);
    final result = await GeminiService.translateSequence(
      _tokens,
      _mode == SignLanguageMode.asl,
    );
    if (mounted) {
      setState(() {
        _translation = result;
        _isTranslating = false;
        _showTranslation = true;
      });
      if (_selectedTranslationLang == 'en' && result.english.isNotEmpty) {
        _tts.speak(result.english, language: 'en-US');
      } else if (_selectedTranslationLang == 'ta' && result.tamil.isNotEmpty) {
        _tts.speak(result.tamil, language: 'ta-IN');
      } else if (_selectedTranslationLang == 'hi' && result.hindi.isNotEmpty) {
        _tts.speak(result.hindi, language: 'hi-IN');
      } else if (_selectedTranslationLang == 'ml' && result.malayalam.isNotEmpty) {
        _tts.speak(result.malayalam, language: 'ml-IN');
      }
    }
  }

  Future<void> _startCamera() async {
    if (_cameras == null || _cameras!.isEmpty || _isCameraInitializing) return;
    _isCameraInitializing = true;
    await _disposeCamera();

    final camera = _cameras!.firstWhere(
      (c) => c.lensDirection == CameraLensDirection.back,
      orElse: () => _cameras!.first,
    );

    final controller = CameraController(
      camera,
      ResolutionPreset.medium,
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.yuv420,
    );
    try {
      await controller.initialize();
      if (!mounted) {
        controller.dispose();
        _isCameraInitializing = false;
        return;
      }
      _cameraController = controller;
      await controller.startImageStream(_onFrame);
      setState(() {
        _isCameraRunning = true;
        _errorMessage = null;
      });
    } catch (e) {
      controller.dispose();
      if (mounted) setState(() => _errorMessage = 'Camera error: $e');
    } finally {
      _isCameraInitializing = false;
    }
  }

  Future<void> _disposeCamera() async {
    _isDisposing = true;
    final c = _cameraController;
    _cameraController = null;
    if (c != null) {
      try {
        if (c.value.isStreamingImages) await c.stopImageStream();
      } catch (_) {}
      try {
        await c.dispose();
      } catch (_) {}
    }
    _isDisposing = false;
  }

  Future<void> _stopCamera() async {
    await _disposeCamera();
    if (mounted) {
      setState(() {
        _isCameraRunning = false;
        _bufferProgress = 0;
        _detectedHands = 0;
        _currentLandmarks = [];
      });
    }
  }

  void _onFrame(CameraImage image) async {
    if (++_frameSkip % 3 != 0 || _isSending || _isDisposing) return;
    _isSending = true;
    try {
      final jpeg = await _convertToJpeg(image);
      if (jpeg != null) _backend.sendFrame(jpeg);
    } finally {
      _isSending = false;
    }
  }

  Future<Uint8List?> _convertToJpeg(CameraImage ci) async {
    try {
      final w = ci.width, h = ci.height;
      final image = img.Image(width: w, height: h);
      if (ci.format.group == ImageFormatGroup.yuv420) {
        final yp = ci.planes[0], up = ci.planes[1], vp = ci.planes[2];
        for (int row = 0; row < h; row++) {
          for (int col = 0; col < w; col++) {
            final yi = row * yp.bytesPerRow + col;
            final uvi = (row >> 1) * up.bytesPerRow + (col & ~1);
            if (yi >= yp.bytes.length ||
                uvi >= up.bytes.length ||
                uvi >= vp.bytes.length)
              continue;
            final yv = yp.bytes[yi], uv = up.bytes[uvi], vv = vp.bytes[uvi];
            image.setPixelRgba(
              col,
              row,
              (yv + 1.370705 * (vv - 128)).round().clamp(0, 255),
              (yv - 0.337633 * (uv - 128) - 0.698001 * (vv - 128))
                  .round()
                  .clamp(0, 255),
              (yv + 1.732446 * (uv - 128)).round().clamp(0, 255),
              255,
            );
          }
        }
      }
      return Uint8List.fromList(img.encodeJpg(image, quality: 60));
    } catch (_) {
      return null;
    }
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState s) {
    if (s == AppLifecycleState.inactive || s == AppLifecycleState.paused) {
      if (_isCameraRunning) _stopCamera();
    }
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _predSub?.cancel();
    _disposeCamera();
    _tts.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF0F0F1A),
      appBar: AppBar(
        title: const Text('Sign Recognition'),
        backgroundColor: Colors.transparent,
        elevation: 0,
        leading: IconButton(
          icon: const Icon(Icons.arrow_back_ios_new_rounded),
          onPressed: () => Navigator.pop(context),
        ),
      ),
      body: SafeArea(
        child: Column(
          children: [
            if (_errorMessage != null) _buildError(),
            Expanded(
              child: Column(
                children: [
                  Expanded(flex: 55, child: _buildCameraView()),
                  Expanded(flex: 45, child: _buildBottomPanel()),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildError() {
    return Container(
      margin: const EdgeInsets.fromLTRB(16, 4, 16, 0),
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      decoration: BoxDecoration(
        color: Colors.red.withValues(alpha: 0.1),
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: Colors.red.withValues(alpha: 0.3)),
      ),
      child: Row(
        children: [
          const Icon(Icons.error_outline, color: Colors.red, size: 16),
          const SizedBox(width: 8),
          Expanded(
            child: Text(
              _errorMessage!,
              style: const TextStyle(color: Colors.red, fontSize: 12),
            ),
          ),
          GestureDetector(
            onTap: () => setState(() => _errorMessage = null),
            child: const Icon(Icons.close, color: Colors.red, size: 16),
          ),
        ],
      ),
    );
  }

  Widget _buildCameraView() {
    final bool cameraReady =
        _isCameraRunning &&
        _cameraController != null &&
        _cameraController!.value.isInitialized;

    final displayConfidence = (_currentConfidence == 0.0 && _stableLabel != null)
        ? 0.85
        : _currentConfidence;

    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 8, 16, 0),
      child: ClipRRect(
        borderRadius: BorderRadius.circular(24),
        child: Stack(
          fit: StackFit.expand,
          children: [
            if (cameraReady)
              AspectRatio(
                aspectRatio: _cameraController!.value.aspectRatio,
                child: CameraPreview(_cameraController!),
              )
            else
              Container(
                decoration: const BoxDecoration(
                  gradient: LinearGradient(
                    colors: [Color(0xFF1A1A2E), Color(0xFF0F0F1A)],
                    begin: Alignment.topLeft,
                    end: Alignment.bottomRight,
                  ),
                ),
                child: Center(
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Icon(
                        Icons.videocam_rounded,
                        size: 52,
                        color: const Color(0xFF6C63FF).withValues(alpha: 0.6),
                      ),
                      const SizedBox(height: 12),
                      const Text(
                        'Initializing...',
                        style: TextStyle(
                          color: Color(0xFF475569),
                          fontSize: 14,
                        ),
                      ),
                    ],
                  ),
                ),
              ),

            if (cameraReady &&
                _currentLandmarks.isNotEmpty &&
                widget.mode == SignLanguageMode.asl)
              LayoutBuilder(
                builder: (ctx, box) => CustomPaint(
                  painter: HandSkeletonPainter(
                    hands: _currentLandmarks,
                    imageSize: Size.zero,
                  ),
                  size: Size(box.maxWidth, box.maxHeight),
                ),
              ),

            if (_isCameraRunning && _currentPrediction != '—')
              Positioned(
                top: 14,
                left: 0,
                right: 0,
                child: Center(
                  child: AnimatedContainer(
                    duration: const Duration(milliseconds: 250),
                    padding: const EdgeInsets.symmetric(
                      horizontal: 20,
                      vertical: 8,
                    ),
                    decoration: BoxDecoration(
                      color: _justLocked
                          ? const Color(0xFF34D399).withValues(alpha: 0.85)
                          : Colors.black.withValues(alpha: 0.65),
                      borderRadius: BorderRadius.circular(30),
                      border: Border.all(
                        color: _justLocked
                            ? const Color(0xFF34D399)
                            : const Color(0xFF6C63FF),
                        width: _justLocked ? 2.5 : 1.5,
                      ),
                    ),
                    child: Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        if (_justLocked) ...[
                          const Icon(
                            Icons.lock_rounded,
                            color: Colors.white,
                            size: 18,
                          ),
                          const SizedBox(width: 6),
                        ],
                        Text(
                          _currentPrediction.toUpperCase(),
                          style: const TextStyle(
                            color: Colors.white,
                            fontSize: 22,
                            fontWeight: FontWeight.w900,
                          ),
                        ),
                        const SizedBox(width: 10),
                        Container(
                          padding: const EdgeInsets.symmetric(
                            horizontal: 8,
                            vertical: 2,
                          ),
                          decoration: BoxDecoration(
                            color:
                                (displayConfidence > 0.6
                                        ? const Color(0xFF34D399)
                                        : const Color(0xFFFBBF24))
                                    .withValues(alpha: 0.2),
                            borderRadius: BorderRadius.circular(20),
                          ),
                          child: Text(
                            '${(displayConfidence * 100).toStringAsFixed(0)}%',
                            style: TextStyle(
                              color: displayConfidence > 0.6
                                  ? const Color(0xFF34D399)
                                  : const Color(0xFFFBBF24),
                              fontSize: 13,
                              fontWeight: FontWeight.w700,
                            ),
                          ),
                        ),
                        if (!_justLocked && _stableCount > 0) ...[
                          const SizedBox(width: 10),
                          SizedBox(
                            width: 28,
                            height: 28,
                            child: Stack(
                              alignment: Alignment.center,
                              children: [
                                CircularProgressIndicator(
                                  value: _stableCount / _stableThreshold,
                                  strokeWidth: 3,
                                  backgroundColor: Colors.white24,
                                  valueColor: AlwaysStoppedAnimation<Color>(
                                    Color.lerp(
                                      const Color(0xFF6C63FF),
                                      const Color(0xFF34D399),
                                      _stableCount / _stableThreshold,
                                    )!,
                                  ),
                                ),
                                Text(
                                  '$_stableCount',
                                  style: const TextStyle(
                                    color: Colors.white,
                                    fontSize: 10,
                                    fontWeight: FontWeight.w800,
                                  ),
                                ),
                              ],
                            ),
                          ),
                        ],
                      ],
                    ),
                  ),
                ),
              ),

            Positioned(
              top: 14,
              left: 14,
              child: Container(
                padding: const EdgeInsets.symmetric(
                  horizontal: 10,
                  vertical: 5,
                ),
                decoration: BoxDecoration(
                  color: Colors.black.withValues(alpha: 0.55),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Text(
                  'Hands: $_detectedHands  |  Buffer: $_bufferProgress/$sequenceLength',
                  style: const TextStyle(
                    color: Colors.white70,
                    fontSize: 11,
                    fontWeight: FontWeight.w600,
                  ),
                ),
              ),
            ),

            Positioned(
              bottom: 0,
              left: 0,
              right: 0,
              child: LinearProgressIndicator(
                value: _bufferProgress / sequenceLength,
                backgroundColor: Colors.black38,
                valueColor: AlwaysStoppedAnimation<Color>(
                  _bufferProgress >= sequenceLength
                      ? const Color(0xFF34D399)
                      : const Color(0xFF6C63FF),
                ),
                minHeight: 4,
              ),
            ),
          ],
        ),
      ),
    );
  }

  void _showTranslationOptions() {
    showModalBottomSheet(
      context: context,
      backgroundColor: const Color(0xFF1E1B4B),
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(24)),
      ),
      builder: (ctx) => SafeArea(
        child: Padding(
          padding: const EdgeInsets.symmetric(vertical: 16),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              const Padding(
                padding: EdgeInsets.only(bottom: 12),
                child: Text(
                  'Select Language',
                  style: TextStyle(color: Colors.white, fontSize: 16, fontWeight: FontWeight.w700),
                ),
              ),
              _buildLangOption('English', 'en', '🇬🇧'),
              _buildLangOption('Tamil', 'ta', 'தமிழ்'),
              _buildLangOption('Hindi', 'hi', 'हिंदी'),
              _buildLangOption('Malayalam', 'ml', 'മലയ്'),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildLangOption(String label, String code, String flag) {
    return ListTile(
      leading: Text(flag, style: const TextStyle(fontSize: 20)),
      title: Text(label, style: const TextStyle(color: Colors.white, fontWeight: FontWeight.w600)),
      trailing: const Icon(Icons.chevron_right_rounded, color: Colors.white54),
      onTap: () {
        Navigator.pop(context);
        setState(() => _selectedTranslationLang = code);
        _translate();
      },
    );
  }

  Widget _buildBottomPanel() {
    return Container(
      margin: const EdgeInsets.fromLTRB(16, 10, 16, 12),
      decoration: BoxDecoration(
        color: const Color(0xFF0F0F1A),
        borderRadius: BorderRadius.circular(24),
        border: Border.all(color: const Color(0xFF2D2D4E)),
      ),
      child: Column(
        children: [
          Padding(
            padding: const EdgeInsets.fromLTRB(14, 12, 14, 0),
            child: Row(
              children: [
                Expanded(
                  child: _ActionBtn(
                    label: 'Start Camera',
                    icon: Icons.play_arrow_rounded,
                    gradient: const [Color(0xFF6C63FF), Color(0xFF483D8B)],
                    onPressed: (!_isCameraRunning) ? _startCamera : null,
                  ),
                ),
                const SizedBox(width: 10),
                Expanded(
                  child: _ActionBtn(
                    label: 'Stop Camera',
                    icon: Icons.stop_rounded,
                    gradient: const [Color(0xFF475569), Color(0xFF334155)],
                    onPressed: _isCameraRunning ? _stopCamera : null,
                  ),
                ),
              ],
            ),
          ),

          Expanded(
            child: Padding(
              padding: const EdgeInsets.fromLTRB(14, 10, 14, 0),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      const Text(
                        'Detected Sequence',
                        style: TextStyle(
                          color: Color(0xFF64748B),
                          fontSize: 11,
                          fontWeight: FontWeight.w600,
                          letterSpacing: 1,
                        ),
                      ),
                      const Spacer(),
                      if (_tokens.isNotEmpty) ...[
                        GestureDetector(
                          onTap: () => setState(() => _tokens.removeLast()),
                          child: const Icon(
                            Icons.backspace_rounded,
                            color: Color(0xFF64748B),
                            size: 16,
                          ),
                        ),
                        const SizedBox(width: 12),
                        GestureDetector(
                          onTap: () => setState(() {
                            _tokens.clear();
                            _translation = TranslationResult.empty;
                            _showTranslation = false;
                          }),
                          child: const Icon(
                            Icons.delete_sweep_rounded,
                            color: Color(0xFF64748B),
                            size: 16,
                          ),
                        ),
                      ],
                    ],
                  ),
                  const SizedBox(height: 8),
                  Expanded(
                    child: _tokens.isEmpty
                        ? const Center(
                            child: Text(
                              'Perform a gesture to begin…',
                              style: TextStyle(
                                color: Color(0xFF2D2D4E),
                                fontSize: 13,
                                fontStyle: FontStyle.italic,
                              ),
                            ),
                          )
                        : SingleChildScrollView(
                            child: Wrap(
                              spacing: 6,
                              runSpacing: 6,
                              children: _tokens
                                  .asMap()
                                  .entries
                                  .map(
                                    (e) => GestureDetector(
                                      onLongPress: () => setState(
                                        () => _tokens.removeAt(e.key),
                                      ),
                                      child: Container(
                                        padding: const EdgeInsets.symmetric(
                                          horizontal: 12,
                                          vertical: 6,
                                        ),
                                        decoration: BoxDecoration(
                                          color: const Color(0xFF6C63FF).withValues(alpha: 0.15),
                                          borderRadius: BorderRadius.circular(
                                            20,
                                          ),
                                          border: Border.all(
                                            color: const Color(0xFF6C63FF).withValues(alpha: 0.4),
                                          ),
                                        ),
                                        child: Text(
                                          e.value,
                                          style: const TextStyle(
                                            color: Colors.white,
                                            fontSize: 15,
                                            fontWeight: FontWeight.w700,
                                          ),
                                        ),
                                      ),
                                    ),
                                  )
                                  .toList(),
                            ),
                          ),
                  ),
                ],
              ),
            ),
          ),

          if (_showTranslation && _translation.isNotEmpty)
            Container(
              margin: const EdgeInsets.fromLTRB(14, 0, 14, 8),
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                gradient: const LinearGradient(
                  colors: [Color(0xFF1E1B4B), Color(0xFF12102A)],
                ),
                borderRadius: BorderRadius.circular(14),
                border: Border.all(
                  color: const Color(0xFF6C63FF).withValues(alpha: 0.3),
                ),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const SizedBox(height: 8),
                  if (_translation.english.isNotEmpty && _selectedTranslationLang == 'en')
                    _TransRow(
                      '🇬🇧',
                      _translation.english,
                      Colors.white,
                      () => _tts.speak(_translation.english, language: 'en-US'),
                    ),
                  if (_translation.tamil.isNotEmpty && _selectedTranslationLang == 'ta')
                    _TransRow(
                      'தமிழ்',
                      _translation.tamil,
                      const Color(0xFF00C9FF),
                      () => _tts.speak(_translation.tamil, language: 'ta-IN'),
                    ),
                  if (_translation.hindi.isNotEmpty && _selectedTranslationLang == 'hi')
                    _TransRow(
                      'हिंदी',
                      _translation.hindi,
                      const Color(0xFFFFB347),
                      () => _tts.speak(_translation.hindi, language: 'hi-IN'),
                    ),
                  if (_translation.malayalam.isNotEmpty && _selectedTranslationLang == 'ml')
                    _TransRow(
                      'മലയ്',
                      _translation.malayalam,
                      const Color(0xFF7CFC00),
                      () => _tts.speak(_translation.malayalam, language: 'ml-IN'),
                    ),
                ],
              ),
            ),

          Padding(
            padding: const EdgeInsets.fromLTRB(14, 0, 14, 14),
            child: Row(
              children: [
                SizedBox(
                  width: 48,
                  height: 48,
                  child: IconButton(
                    onPressed: _tokens.isEmpty
                        ? null
                        : () => setState(() {
                            _tokens.clear();
                            _translation = TranslationResult.empty;
                            _showTranslation = false;
                          }),
                    icon: const Icon(Icons.delete_forever_rounded),
                    color: const Color(0xFFF87171),
                    disabledColor: const Color(0xFF334155),
                    style: IconButton.styleFrom(
                      backgroundColor: _tokens.isNotEmpty
                          ? const Color(0xFFF87171).withValues(alpha: 0.12)
                          : const Color(0xFF1E293B),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(14),
                      ),
                    ),
                    tooltip: 'Clear sequence',
                  ),
                ),
                const SizedBox(width: 10),
                Expanded(
                  child: SizedBox(
                    height: 48,
                    child: ElevatedButton.icon(
                      onPressed: (_tokens.isEmpty || _isTranslating)
                          ? null
                          : _showTranslationOptions,
                      icon: _isTranslating
                          ? const SizedBox(
                              width: 18,
                              height: 18,
                              child: CircularProgressIndicator(
                                strokeWidth: 2,
                                color: Colors.white,
                              ),
                            )
                          : const Icon(Icons.language_rounded),
                      label: Text(
                        _isTranslating
                            ? 'Translating…'
                            : 'Translate with Gemini',
                      ),
                      style: ElevatedButton.styleFrom(
                        backgroundColor: const Color(0xFF6C63FF),
                        disabledBackgroundColor: const Color(0xFF1E293B),
                        foregroundColor: Colors.white,
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(14),
                        ),
                        textStyle: const TextStyle(
                          fontSize: 14,
                          fontWeight: FontWeight.w700,
                        ),
                      ),
                    ),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class _ActionBtn extends StatelessWidget {
  final String label;
  final IconData icon;
  final List<Color> gradient;
  final VoidCallback? onPressed;
  const _ActionBtn({
    required this.label,
    required this.icon,
    required this.gradient,
    this.onPressed,
  });

  @override
  Widget build(BuildContext context) {
    final on = onPressed != null;
    return GestureDetector(
      onTap: onPressed,
      child: Container(
        padding: const EdgeInsets.symmetric(vertical: 11),
        decoration: BoxDecoration(
          gradient: on ? LinearGradient(colors: gradient) : null,
          color: on ? null : const Color(0xFF1E293B),
          borderRadius: BorderRadius.circular(12),
        ),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(
              icon,
              size: 18,
              color: on ? Colors.white : const Color(0xFF475569),
            ),
            const SizedBox(width: 6),
            Text(
              label,
              style: TextStyle(
                color: on ? Colors.white : const Color(0xFF475569),
                fontSize: 13,
                fontWeight: FontWeight.w700,
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _TransRow extends StatelessWidget {
  final String label;
  final String text;
  final Color color;
  final VoidCallback onSpeak;
  const _TransRow(this.label, this.text, this.color, this.onSpeak);

  @override
  Widget build(BuildContext context) {
    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          label,
          style: TextStyle(
            color: color.withValues(alpha: 0.7),
            fontSize: 11,
            fontWeight: FontWeight.w700,
          ),
        ),
        const SizedBox(width: 8),
        Expanded(
          child: Text(
            text,
            style: TextStyle(
              color: color,
              fontSize: 14,
              fontWeight: FontWeight.w600,
            ),
          ),
        ),
        GestureDetector(
          onTap: onSpeak,
          child: Padding(
            padding: const EdgeInsets.only(left: 8.0),
            child: Icon(
              Icons.volume_up_rounded,
              color: color.withValues(alpha: 0.8),
              size: 18,
            ),
          ),
        ),
      ],
    );
  }
}
