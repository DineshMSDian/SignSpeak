import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:image/image.dart' as img;
import '../config.dart';
import '../services/backend_service.dart';
import '../services/tts_service.dart';
import '../widgets/server_config_dialog.dart';

// ══════════════════════════════════════════════════════════════════
// MediaPipe hand skeleton connections
// ══════════════════════════════════════════════════════════════════
const List<List<int>> kHandConnections = [
  [0,1],[1,2],[2,3],[3,4],
  [0,5],[5,6],[6,7],[7,8],
  [0,9],[9,10],[10,11],[11,12],
  [0,13],[13,14],[14,15],[15,16],
  [0,17],[17,18],[18,19],[19,20],
  [5,9],[9,13],[13,17],
];
const Set<int> kFingertips = {4, 8, 12, 16, 20};

// ══════════════════════════════════════════════════════════════════
// Hand Skeleton Painter — Correct coordinate mapping with BoxFit.cover
// ══════════════════════════════════════════════════════════════════
class HandSkeletonPainter extends CustomPainter {
  final List<List<Map<String, double>>> hands;
  final Size previewSize;
  final bool mirrorX;

  const HandSkeletonPainter({
    required this.hands,
    required this.previewSize,
    this.mirrorX = true,
  });

  @override
  void paint(Canvas canvas, Size size) {
    if (hands.isEmpty) return;

    // BoxFit.cover: scale to fill, crop excess
    final double camAspect = previewSize.width / previewSize.height;
    final double widgetAspect = size.width / size.height;

    double scaleX, scaleY, offsetX = 0, offsetY = 0;
    if (camAspect > widgetAspect) {
      scaleY = size.height;
      scaleX = size.height * camAspect;
      offsetX = (size.width - scaleX) / 2;
    } else {
      scaleX = size.width;
      scaleY = size.width / camAspect;
      offsetY = (size.height - scaleY) / 2;
    }

    Offset toCanvas(Map<String, double> lm) {
      final nx = mirrorX ? (1.0 - lm['x']!) : lm['x']!;
      return Offset(offsetX + nx * scaleX, offsetY + lm['y']! * scaleY);
    }

    final bonePaint = Paint()
      ..color = Colors.white.withOpacity(0.9)
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
      ..color = Colors.white.withOpacity(0.7)
      ..strokeWidth = 1.8
      ..style = PaintingStyle.stroke;

    for (final hand in hands) {
      if (hand.length < 21) continue;
      for (final c in kHandConnections) {
        canvas.drawLine(toCanvas(hand[c[0]]), toCanvas(hand[c[1]]), bonePaint);
      }
      for (int i = 0; i < hand.length; i++) {
        final o = toCanvas(hand[i]);
        if (kFingertips.contains(i)) {
          canvas.drawCircle(o, 9.0, tipPaint);
          canvas.drawCircle(o, 9.0, tipRing);
        } else {
          canvas.drawCircle(o, 5.5, jointPaint);
        }
      }
    }
  }

  @override
  bool shouldRepaint(HandSkeletonPainter old) => true;
}

// ══════════════════════════════════════════════════════════════════
// Gemini Service — English + Tamil + Hindi
// ══════════════════════════════════════════════════════════════════
class GeminiService {
  static const String _apiKey = 'YOUR_GEMINI_API_KEY'; // ← replace
  static const String _endpoint =
      'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=$_apiKey';

  static Future<Map<String, String>> translate(
      List<String> tokens, bool isASL) async {
    if (tokens.isEmpty) return {'english': '', 'tamil': '', 'hindi': ''};
    final input = isASL ? tokens.join('') : tokens.join(' ');
    final prompt = isASL
        ? 'These are ASL fingerspelled letters: "$input". Combine into the most likely word or sentence. '
            'Reply ONLY with JSON: {"english":"...","tamil":"...","hindi":"..."} — no explanation.'
        : 'These are ISL gesture words: "$input". Form a natural English sentence, translate to Tamil and Hindi. '
            'Reply ONLY with JSON: {"english":"...","tamil":"...","hindi":"..."} — no explanation.';

    try {
      final res = await http.post(Uri.parse(_endpoint),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({
          'contents': [{'parts': [{'text': prompt}]}],
          'generationConfig': {'temperature': 0.2, 'maxOutputTokens': 200},
        }),
      ).timeout(const Duration(seconds: 12));
      if (res.statusCode == 200) {
        final data = jsonDecode(res.body);
        final text = data['candidates']?[0]?['content']?['parts']?[0]?['text'] as String? ?? '';
        final clean = text.replaceAll(RegExp(r'```json|```'), '').trim();
        final parsed = jsonDecode(clean) as Map<String, dynamic>;
        return {
          'english': parsed['english'] as String? ?? input,
          'tamil': parsed['tamil'] as String? ?? '',
          'hindi': parsed['hindi'] as String? ?? '',
        };
      }
    } catch (e) { debugPrint('[Gemini] $e'); }
    return {'english': input, 'tamil': '', 'hindi': ''};
  }
}

// ══════════════════════════════════════════════════════════════════
// HomeScreen
// ══════════════════════════════════════════════════════════════════
class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});
  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> with WidgetsBindingObserver {
  final BackendService _backend = BackendService();
  final TtsService _tts = TtsService();

  CameraController? _cameraController;
  List<CameraDescription>? _cameras;
  bool _isCameraRunning = false;
  bool _isCameraInitializing = false;
  bool _isDisposing = false;
  Size _previewSize = const Size(640, 480);

  SignLanguageMode _mode = SignLanguageMode.asl;
  String _currentPrediction = '—';
  double _currentConfidence = 0.0;
  int _bufferProgress = 0;
  int _detectedHands = 0;
  int _modelClasses = 0;
  bool _isConnected = false;
  bool _isConnecting = false;
  String? _errorMessage;
  List<List<Map<String, double>>> _currentLandmarks = [];

  final List<String> _tokens = [];
  Map<String, String> _translation = {};
  bool _isTranslating = false;
  bool _showTranslation = false;

  // Hold-to-confirm
  String? _holdCandidate;
  DateTime? _holdStart;
  DateTime? _lastCommit;
  static const _holdDur = Duration(milliseconds: 1200);
  static const _cooldown = Duration(milliseconds: 900);

  int _frameSkip = 0;
  bool _isSending = false;

  StreamSubscription<PredictionResult>? _predSub;
  StreamSubscription<bool>? _connSub;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _init();
  }

  Future<void> _init() async {
    try {
      _cameras = await availableCameras();
      _predSub = _backend.predictions.listen(_onPrediction);
      _connSub = _backend.connectionState.listen((c) {
        if (mounted) setState(() => _isConnected = c);
        if (!c && _isCameraRunning) {
          _stopCamera();
          setState(() => _errorMessage = 'Lost connection');
        }
      });
    } catch (e) {
      setState(() => _errorMessage = 'Init error: $e');
    }
  }

  void _onPrediction(PredictionResult result) {
    if (!mounted) return;
    setState(() {
      _detectedHands = result.numHands;
      _bufferProgress = result.bufferFill;
      _currentConfidence = result.confidence;
      _currentLandmarks = result.landmarks;
    });
    final label = result.prediction;
    if (label == null) return;
    setState(() => _currentPrediction = label);

    final now = DateTime.now();
    if (_lastCommit != null && now.difference(_lastCommit!) < _cooldown) {
      _holdCandidate = null;
      return;
    }
    if (label == _holdCandidate) {
      if (_holdStart != null && now.difference(_holdStart!) >= _holdDur) {
        _commitToken(label);
        _holdCandidate = null;
        _holdStart = null;
        _lastCommit = now;
      }
    } else {
      _holdCandidate = label;
      _holdStart = now;
    }
  }

  void _commitToken(String token) {
    setState(() { _tokens.add(token); _showTranslation = false; _translation = {}; });
    _tts.speak(token.replaceAll('_', ' '));
  }

  Future<void> _translate() async {
    if (_tokens.isEmpty) return;
    setState(() => _isTranslating = true);
    final result = await GeminiService.translate(_tokens, _mode == SignLanguageMode.asl);
    if (mounted) setState(() { _translation = result; _isTranslating = false; _showTranslation = true; });
    if ((result['english'] ?? '').isNotEmpty) _tts.speak(result['english']!);
  }

  Future<void> _connectToServer() async {
    if (_isConnecting) return;
    setState(() { _isConnecting = true; _errorMessage = null; });
    final health = await _backend.healthCheck();
    if (health != null) {
      _modelClasses = health['classes'] as int? ?? 0;
      final connected = await _backend.connect();
      if (mounted) setState(() { _isConnected = connected; _isConnecting = false; });
    } else {
      if (mounted) setState(() { _isConnecting = false; _errorMessage = 'Cannot reach server'; });
    }
  }

  Future<void> _switchMode(SignLanguageMode mode) async {
    if (mode == _mode) return;
    final result = await _backend.setMode(mode.code);
    if (result != null && result['success'] == true) {
      setState(() {
        _mode = mode;
        _modelClasses = result['classes'] as int? ?? 0;
        _currentPrediction = '—';
        _currentConfidence = 0.0;
        _tokens.clear();
        _translation = {};
        _showTranslation = false;
      });
    }
  }

  Future<void> _startCamera() async {
    if (_cameras == null || _cameras!.isEmpty || _isCameraInitializing || !_isConnected) return;
    _isCameraInitializing = true;
    await _disposeCamera();

    // Front camera matches training data (collected on laptop front cam)
    final camera = _cameras!.firstWhere(
      (c) => c.lensDirection == CameraLensDirection.front,
      orElse: () => _cameras!.first,
    );

    final controller = CameraController(
      camera, ResolutionPreset.medium,
      enableAudio: false, imageFormatGroup: ImageFormatGroup.yuv420,
    );
    try {
      await controller.initialize();
      if (!mounted) { controller.dispose(); _isCameraInitializing = false; return; }
      _cameraController = controller;
      final ps = controller.value.previewSize;
      // previewSize is in landscape; for portrait we swap w/h
      if (ps != null) _previewSize = Size(ps.height, ps.width);
      await controller.startImageStream(_onFrame);
      setState(() { _isCameraRunning = true; _errorMessage = null; });
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
      try { if (c.value.isStreamingImages) await c.stopImageStream(); } catch (_) {}
      try { await c.dispose(); } catch (_) {}
    }
    _isDisposing = false;
  }

  Future<void> _stopCamera() async {
    await _disposeCamera();
    if (mounted) setState(() {
      _isCameraRunning = false;
      _bufferProgress = 0;
      _detectedHands = 0;
      _currentLandmarks = [];
    });
  }

  void _onFrame(CameraImage image) async {
    if (++_frameSkip % 3 != 0 || _isSending || _isDisposing || !_isConnected) return;
    _isSending = true;
    try {
      final jpeg = await _convertToJpeg(image);
      if (jpeg != null) _backend.sendFrame(jpeg);
    } finally { _isSending = false; }
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
            if (yi >= yp.bytes.length || uvi >= up.bytes.length || uvi >= vp.bytes.length) continue;
            final yv = yp.bytes[yi], uv = up.bytes[uvi], vv = vp.bytes[uvi];
            image.setPixelRgba(col, row,
              (yv + 1.370705 * (vv - 128)).round().clamp(0, 255),
              (yv - 0.337633 * (uv - 128) - 0.698001 * (vv - 128)).round().clamp(0, 255),
              (yv + 1.732446 * (uv - 128)).round().clamp(0, 255),
              255);
          }
        }
      }
      return Uint8List.fromList(img.encodeJpg(image, quality: 60));
    } catch (_) { return null; }
  }

  void _showServerConfig() {
    showDialog(
      context: context,
      builder: (ctx) => ServerConfigDialog(
        currentIp: _backend.serverIp,
        currentPort: _backend.serverPort,
        onSave: (ip, port) { _backend.setServer(ip, port: port); _connectToServer(); },
      ),
    );
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
    _connSub?.cancel();
    _disposeCamera();
    _backend.dispose();
    _tts.dispose();
    super.dispose();
  }

  // ════════════════════════════════════════════════════════════
  // BUILD
  // ════════════════════════════════════════════════════════════
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF0F0F1A),
      body: SafeArea(
        child: Column(
          children: [
            _buildHeader(),
            _buildConnectionBar(),
            _buildModeToggle(),
            if (_errorMessage != null) _buildError(),
            Expanded(
              child: Column(
                children: [
                  // Camera: 55% of remaining space
                  Expanded(flex: 55, child: _buildCameraView()),
                  // Bottom panel: 45%
                  Expanded(flex: 45, child: _buildBottomPanel()),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildHeader() {
    return Padding(
      padding: const EdgeInsets.fromLTRB(20, 12, 12, 0),
      child: Row(
        children: [
          ShaderMask(
            shaderCallback: (b) => const LinearGradient(
              colors: [Color(0xFF6C63FF), Color(0xFF00C9FF)],
            ).createShader(b),
            child: const Text('🤟 SignSpeak',
                style: TextStyle(fontSize: 26, fontWeight: FontWeight.w900, color: Colors.white)),
          ),
          const Spacer(),
          IconButton(
            icon: Icon(Icons.settings_rounded,
                color: _isConnected ? const Color(0xFF34D399) : const Color(0xFF475569)),
            onPressed: _showServerConfig,
          ),
        ],
      ),
    );
  }

  Widget _buildConnectionBar() {
    final color = _isConnected ? const Color(0xFF34D399)
        : _isConnecting ? const Color(0xFFFBBF24) : const Color(0xFFF87171);
    final label = _isConnected ? 'Connected to ${_backend.serverIp}'
        : _isConnecting ? 'Connecting…' : 'Tap to configure server';
    return GestureDetector(
      onTap: _isConnected || _isConnecting ? null : _showServerConfig,
      child: Container(
        margin: const EdgeInsets.fromLTRB(16, 8, 16, 4),
        padding: const EdgeInsets.symmetric(vertical: 9, horizontal: 14),
        decoration: BoxDecoration(
          color: color.withOpacity(0.08),
          borderRadius: BorderRadius.circular(12),
          border: Border.all(color: color.withOpacity(0.3)),
        ),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            if (_isConnecting)
              SizedBox(width: 13, height: 13, child: CircularProgressIndicator(strokeWidth: 2, color: color))
            else
              Icon(_isConnected ? Icons.cloud_done_rounded : Icons.cloud_off_rounded, size: 15, color: color),
            const SizedBox(width: 8),
            Text(label, style: TextStyle(fontSize: 12, fontWeight: FontWeight.w600, color: color)),
            if (_isConnected) ...[
              const Spacer(),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
                decoration: BoxDecoration(color: color.withOpacity(0.15), borderRadius: BorderRadius.circular(20)),
                child: Text('$_modelClasses classes',
                    style: TextStyle(fontSize: 10, color: color, fontWeight: FontWeight.w700)),
              ),
            ],
          ],
        ),
      ),
    );
  }

  Widget _buildModeToggle() {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 4),
      child: Container(
        decoration: BoxDecoration(
          color: const Color(0xFF1A1A2E),
          borderRadius: BorderRadius.circular(14),
          border: Border.all(color: const Color(0xFF2D2D4E)),
        ),
        child: Row(
          children: SignLanguageMode.values.map((m) {
            final sel = m == _mode;
            return Expanded(
              child: GestureDetector(
                onTap: (_isCameraRunning || !_isConnected) ? null : () => _switchMode(m),
                child: AnimatedContainer(
                  duration: const Duration(milliseconds: 200),
                  padding: const EdgeInsets.symmetric(vertical: 10),
                  decoration: BoxDecoration(
                    gradient: sel ? LinearGradient(colors: m == SignLanguageMode.asl
                        ? [const Color(0xFF6C63FF), const Color(0xFF483D8B)]
                        : [const Color(0xFF00C9FF), const Color(0xFF0066CC)]) : null,
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: Text(m.badge, textAlign: TextAlign.center,
                      style: TextStyle(fontSize: 13,
                          fontWeight: sel ? FontWeight.w700 : FontWeight.w500,
                          color: sel ? Colors.white : const Color(0xFF64748B))),
                ),
              ),
            );
          }).toList(),
        ),
      ),
    );
  }

  Widget _buildError() {
    return Container(
      margin: const EdgeInsets.fromLTRB(16, 4, 16, 0),
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      decoration: BoxDecoration(
        color: Colors.red.withOpacity(0.1), borderRadius: BorderRadius.circular(10),
        border: Border.all(color: Colors.red.withOpacity(0.3)),
      ),
      child: Row(children: [
        const Icon(Icons.error_outline, color: Colors.red, size: 16),
        const SizedBox(width: 8),
        Expanded(child: Text(_errorMessage!, style: const TextStyle(color: Colors.red, fontSize: 12))),
        GestureDetector(onTap: () => setState(() => _errorMessage = null),
            child: const Icon(Icons.close, color: Colors.red, size: 16)),
      ]),
    );
  }

  // ── Camera view ────────────────────────────────────────────────
  Widget _buildCameraView() {
    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 8, 16, 0),
      child: ClipRRect(
        borderRadius: BorderRadius.circular(24),
        child: Stack(
          fit: StackFit.expand,
          children: [
            // Camera or placeholder
            if (_isCameraRunning && _cameraController != null && _cameraController!.value.isInitialized)
              CameraPreview(_cameraController!)
            else
              Container(
                decoration: const BoxDecoration(
                  gradient: LinearGradient(colors: [Color(0xFF1A1A2E), Color(0xFF0F0F1A)],
                      begin: Alignment.topLeft, end: Alignment.bottomRight),
                ),
                child: Center(child: Column(mainAxisSize: MainAxisSize.min, children: [
                  Icon(_isConnected ? Icons.videocam_rounded : Icons.wifi_off_rounded,
                      size: 52, color: const Color(0xFF6C63FF).withOpacity(0.6)),
                  const SizedBox(height: 12),
                  Text(_isConnected ? 'Tap Start Camera' : 'Connect to server first',
                      style: const TextStyle(color: Color(0xFF475569), fontSize: 14)),
                ])),
              ),

            // ✅ Skeleton with proper BoxFit.cover-aware coordinate mapping
            if (_isCameraRunning && _currentLandmarks.isNotEmpty)
              LayoutBuilder(builder: (ctx, box) => CustomPaint(
                painter: HandSkeletonPainter(
                  hands: _currentLandmarks,
                  previewSize: _previewSize,
                  mirrorX: true,
                ),
                size: Size(box.maxWidth, box.maxHeight),
              )),

            // Live prediction badge
            if (_isCameraRunning && _currentPrediction != '—')
              Positioned(
                top: 14, left: 0, right: 0,
                child: Center(child: Container(
                  padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 8),
                  decoration: BoxDecoration(
                    color: Colors.black.withOpacity(0.65),
                    borderRadius: BorderRadius.circular(30),
                    border: Border.all(color: const Color(0xFF6C63FF), width: 1.5),
                  ),
                  child: Row(mainAxisSize: MainAxisSize.min, children: [
                    Text(_currentPrediction.toUpperCase(),
                        style: const TextStyle(color: Colors.white, fontSize: 22, fontWeight: FontWeight.w900)),
                    const SizedBox(width: 10),
                    Container(
                      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
                      decoration: BoxDecoration(
                        color: (_currentConfidence > 0.6
                            ? const Color(0xFF34D399) : const Color(0xFFFBBF24)).withOpacity(0.2),
                        borderRadius: BorderRadius.circular(20),
                      ),
                      child: Text('${(_currentConfidence * 100).toStringAsFixed(0)}%',
                          style: TextStyle(
                              color: _currentConfidence > 0.6
                                  ? const Color(0xFF34D399) : const Color(0xFFFBBF24),
                              fontSize: 13, fontWeight: FontWeight.w700)),
                    ),
                  ]),
                )),
              ),

            // Hold ring
            if (_holdCandidate != null)
              Positioned(bottom: 14, right: 14,
                  child: _HoldRing(letter: _holdCandidate!, holdStart: _holdStart, holdDuration: _holdDur)),

            // HUD
            Positioned(top: 14, left: 14,
              child: Container(
                padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
                decoration: BoxDecoration(color: Colors.black.withOpacity(0.55), borderRadius: BorderRadius.circular(8)),
                child: Text('Hands: $_detectedHands  |  Buffer: $_bufferProgress/$sequenceLength',
                    style: const TextStyle(color: Colors.white70, fontSize: 11, fontWeight: FontWeight.w600)),
              ),
            ),

            // Buffer bar
            Positioned(bottom: 0, left: 0, right: 0,
              child: LinearProgressIndicator(
                value: _bufferProgress / sequenceLength,
                backgroundColor: Colors.black38,
                valueColor: AlwaysStoppedAnimation<Color>(
                    _bufferProgress >= sequenceLength ? const Color(0xFF34D399) : const Color(0xFF6C63FF)),
                minHeight: 4,
              ),
            ),
          ],
        ),
      ),
    );
  }

  // ── Bottom panel ──────────────────────────────────────────────
  Widget _buildBottomPanel() {
    return Container(
      margin: const EdgeInsets.fromLTRB(16, 10, 16, 12),
      decoration: BoxDecoration(
        color: const Color(0xFF1A1A2E),
        borderRadius: BorderRadius.circular(24),
        border: Border.all(color: const Color(0xFF2D2D4E)),
      ),
      child: Column(children: [
        // Camera buttons
        Padding(
          padding: const EdgeInsets.fromLTRB(14, 12, 14, 0),
          child: Row(children: [
            Expanded(child: _ActionBtn(
              label: 'Start Camera', icon: Icons.play_arrow_rounded,
              gradient: const [Color(0xFF6C63FF), Color(0xFF483D8B)],
              onPressed: (!_isCameraRunning && _isConnected) ? _startCamera : null,
            )),
            const SizedBox(width: 10),
            Expanded(child: _ActionBtn(
              label: 'Stop Camera', icon: Icons.stop_rounded,
              gradient: const [Color(0xFF475569), Color(0xFF334155)],
              onPressed: _isCameraRunning ? _stopCamera : null,
            )),
          ]),
        ),

        // Token chips
        Expanded(child: Padding(
          padding: const EdgeInsets.fromLTRB(14, 10, 14, 0),
          child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            Row(children: [
              const Text('Detected Sequence',
                  style: TextStyle(color: Color(0xFF64748B), fontSize: 11,
                      fontWeight: FontWeight.w600, letterSpacing: 1)),
              const Spacer(),
              if (_tokens.isNotEmpty) ...[
                GestureDetector(
                  onTap: () => setState(() => _tokens.removeLast()),
                  child: const Icon(Icons.backspace_rounded, color: Color(0xFF64748B), size: 16),
                ),
                const SizedBox(width: 12),
                GestureDetector(
                  onTap: () => setState(() { _tokens.clear(); _translation = {}; _showTranslation = false; }),
                  child: const Icon(Icons.delete_sweep_rounded, color: Color(0xFF64748B), size: 16),
                ),
              ],
            ]),
            const SizedBox(height: 8),
            Expanded(child: _tokens.isEmpty
                ? const Center(child: Text('Hold a sign for 1.2s to add it…',
                    style: TextStyle(color: Color(0xFF2D2D4E), fontSize: 13, fontStyle: FontStyle.italic)))
                : SingleChildScrollView(child: Wrap(spacing: 6, runSpacing: 6,
                    children: _tokens.asMap().entries.map((e) => GestureDetector(
                      onLongPress: () => setState(() => _tokens.removeAt(e.key)),
                      child: Container(
                        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                        decoration: BoxDecoration(
                          color: const Color(0xFF6C63FF).withOpacity(0.15),
                          borderRadius: BorderRadius.circular(20),
                          border: Border.all(color: const Color(0xFF6C63FF).withOpacity(0.4)),
                        ),
                        child: Text(e.value,
                            style: const TextStyle(color: Colors.white, fontSize: 15, fontWeight: FontWeight.w700)),
                      ),
                    )).toList(),
                  ))),
          ]),
        )),

        // Translation result card
        if (_showTranslation && _translation.isNotEmpty)
          Container(
            margin: const EdgeInsets.fromLTRB(14, 0, 14, 8),
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              gradient: const LinearGradient(colors: [Color(0xFF1E1B4B), Color(0xFF12102A)]),
              borderRadius: BorderRadius.circular(14),
              border: Border.all(color: const Color(0xFF6C63FF).withOpacity(0.3)),
            ),
            child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
              Row(children: [
                const Text('✨ Translation', style: TextStyle(color: Color(0xFF6C63FF),
                    fontSize: 11, fontWeight: FontWeight.w700, letterSpacing: 1)),
                const Spacer(),
                GestureDetector(
                  onTap: () { if ((_translation['english'] ?? '').isNotEmpty) _tts.speak(_translation['english']!); },
                  child: const Icon(Icons.volume_up_rounded, color: Color(0xFF34D399), size: 16),
                ),
              ]),
              const SizedBox(height: 8),
              if ((_translation['english'] ?? '').isNotEmpty)
                _TransRow('🇬🇧', _translation['english']!, Colors.white),
              if ((_translation['tamil'] ?? '').isNotEmpty) ...[
                const SizedBox(height: 4),
                _TransRow('தமிழ்', _translation['tamil']!, const Color(0xFF00C9FF)),
              ],
              if ((_translation['hindi'] ?? '').isNotEmpty) ...[
                const SizedBox(height: 4),
                _TransRow('हिंदी', _translation['hindi']!, const Color(0xFFFFB347)),
              ],
            ]),
          ),

        // Translate button
        Padding(
          padding: const EdgeInsets.fromLTRB(14, 0, 14, 14),
          child: SizedBox(
            width: double.infinity, height: 48,
            child: ElevatedButton.icon(
              onPressed: (_tokens.isEmpty || _isTranslating) ? null : _translate,
              icon: _isTranslating
                  ? const SizedBox(width: 18, height: 18,
                      child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white))
                  : const Icon(Icons.language_rounded),
              label: Text(_isTranslating ? 'Translating…' : 'Translate with Gemini'),
              style: ElevatedButton.styleFrom(
                backgroundColor: const Color(0xFF6C63FF),
                disabledBackgroundColor: const Color(0xFF1E293B),
                foregroundColor: Colors.white,
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
                textStyle: const TextStyle(fontSize: 14, fontWeight: FontWeight.w700),
              ),
            ),
          ),
        ),
      ]),
    );
  }
}

// ══════════════════════════════════════════════════════════════════
// Hold Ring — properly animated with its own timer
// ══════════════════════════════════════════════════════════════════
class _HoldRing extends StatefulWidget {
  final String letter;
  final DateTime? holdStart;
  final Duration holdDuration;
  const _HoldRing({required this.letter, required this.holdStart, required this.holdDuration});

  @override
  State<_HoldRing> createState() => _HoldRingState();
}

class _HoldRingState extends State<_HoldRing> {
  late final Timer _t;
  @override
  void initState() {
    super.initState();
    _t = Timer.periodic(const Duration(milliseconds: 40), (_) { if (mounted) setState(() {}); });
  }
  @override
  void dispose() { _t.cancel(); super.dispose(); }

  @override
  Widget build(BuildContext context) {
    final progress = widget.holdStart == null ? 0.0
        : (DateTime.now().difference(widget.holdStart!).inMilliseconds /
            widget.holdDuration.inMilliseconds).clamp(0.0, 1.0);
    return SizedBox(width: 56, height: 56,
      child: Stack(alignment: Alignment.center, children: [
        CircularProgressIndicator(
          value: progress, strokeWidth: 4,
          backgroundColor: Colors.white24,
          valueColor: AlwaysStoppedAnimation<Color>(
              Color.lerp(const Color(0xFF6C63FF), const Color(0xFFFFD700), progress)!),
        ),
        Text(widget.letter.toUpperCase(),
            style: const TextStyle(color: Colors.white, fontSize: 20, fontWeight: FontWeight.w900)),
      ]),
    );
  }
}

// ══════════════════════════════════════════════════════════════════
// Helper widgets
// ══════════════════════════════════════════════════════════════════
class _ActionBtn extends StatelessWidget {
  final String label;
  final IconData icon;
  final List<Color> gradient;
  final VoidCallback? onPressed;
  const _ActionBtn({required this.label, required this.icon, required this.gradient, this.onPressed});

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
        child: Row(mainAxisAlignment: MainAxisAlignment.center, children: [
          Icon(icon, size: 18, color: on ? Colors.white : const Color(0xFF475569)),
          const SizedBox(width: 6),
          Text(label, style: TextStyle(
              color: on ? Colors.white : const Color(0xFF475569),
              fontSize: 13, fontWeight: FontWeight.w700)),
        ]),
      ),
    );
  }
}

class _TransRow extends StatelessWidget {
  final String label;
  final String text;
  final Color color;
  const _TransRow(this.label, this.text, this.color);

  @override
  Widget build(BuildContext context) {
    return Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
      Text(label, style: TextStyle(color: color.withOpacity(0.7), fontSize: 11, fontWeight: FontWeight.w700)),
      const SizedBox(width: 8),
      Expanded(child: Text(text, style: TextStyle(color: color, fontSize: 14, fontWeight: FontWeight.w600))),
    ]);
  }
}
