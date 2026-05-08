import 'dart:async';
import 'package:flutter/material.dart';
import '../config.dart';
import '../services/backend_service.dart';
import '../widgets/server_config_dialog.dart';
import 'recognition_screen.dart';

class ModeSelectionScreen extends StatefulWidget {
  const ModeSelectionScreen({super.key});

  @override
  State<ModeSelectionScreen> createState() => _ModeSelectionScreenState();
}

class _ModeSelectionScreenState extends State<ModeSelectionScreen> {
  final BackendService _backend = BackendService();
  bool _isConnected = false;
  bool _isConnecting = false;
  String? _errorMessage;
  int _modelClasses = 0;
  StreamSubscription<bool>? _connSub;

  @override
  void initState() {
    super.initState();
    _connSub = _backend.connectionState.listen((c) {
      if (mounted) setState(() => _isConnected = c);
      if (!c) {
        setState(() => _errorMessage = 'Lost connection');
      }
    });
    Future.delayed(const Duration(milliseconds: 100), _connectToServer);
  }

  Future<void> _connectToServer() async {
    if (_isConnecting) return;
    setState(() {
      _isConnecting = true;
      _errorMessage = null;
    });
    final health = await _backend.healthCheck();
    if (health != null) {
      _modelClasses = health['classes'] as int? ?? 0;
      final connected = await _backend.connect();
      if (mounted)
        setState(() {
          _isConnected = connected;
          _isConnecting = false;
        });
    } else {
      if (mounted)
        setState(() {
          _isConnecting = false;
          _errorMessage = 'Cannot reach server';
        });
    }
  }

  void _showServerConfig() {
    showDialog(
      context: context,
      builder: (ctx) => ServerConfigDialog(
        currentIp: _backend.serverIp,
        currentPort: _backend.serverPort,
        onSave: (ip, port) {
          _backend.setServer(ip, port: port);
          _connectToServer();
        },
      ),
    );
  }

  void _navigateToRecognition(SignLanguageMode mode) {
    if (!_isConnected) return;
    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (_) => RecognitionScreen(mode: mode, backend: _backend),
      ),
    );
  }

  @override
  void dispose() {
    _connSub?.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF0F0F1A),
      body: SafeArea(
        child: Column(
          children: [
            _buildHeader(),
            _buildConnectionBar(),
            if (_errorMessage != null)
              Container(
                margin: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                padding: const EdgeInsets.symmetric(
                  vertical: 8,
                  horizontal: 12,
                ),
                decoration: BoxDecoration(
                  color: const Color(0xFFF87171).withValues(alpha: 0.1),
                  borderRadius: BorderRadius.circular(8),
                  border: Border.all(
                    color: const Color(0xFFF87171).withValues(alpha: 0.3),
                  ),
                ),
                child: Row(
                  children: [
                    const Icon(
                      Icons.error_outline_rounded,
                      color: Color(0xFFF87171),
                      size: 16,
                    ),
                    const SizedBox(width: 8),
                    Expanded(
                      child: Text(
                        _errorMessage!,
                        style: const TextStyle(
                          color: Color(0xFFF87171),
                          fontSize: 13,
                        ),
                      ),
                    ),
                    GestureDetector(
                      onTap: () => setState(() => _errorMessage = null),
                      child: const Icon(
                        Icons.close_rounded,
                        color: Color(0xFFF87171),
                        size: 16,
                      ),
                    ),
                  ],
                ),
              ),
            Expanded(
              child: Center(
                child: Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 24),
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      // Hero icon
                      Container(
                        width: 80,
                        height: 80,
                        decoration: BoxDecoration(
                          color: const Color(0xFF1A1A38),
                          borderRadius: BorderRadius.circular(20),
                          border: Border.all(
                            color: const Color(0xFF3C3489),
                            width: 0.5,
                          ),
                        ),
                        child: const Center(
                          child: Icon(
                            Icons.back_hand_outlined,
                            color: Color(0xFF7F77DD),
                            size: 36,
                          ),
                        ),
                      ),
                      const SizedBox(height: 20),
                      const Text(
                        'Sign Recognition',
                        style: TextStyle(
                          fontSize: 20,
                          fontWeight: FontWeight.w500,
                          color: Colors.white,
                        ),
                      ),
                      const SizedBox(height: 8),
                      const SizedBox(
                        width: 220,
                        child: Text(
                          'Real-time gesture detection with multilingual translation',
                          textAlign: TextAlign.center,
                          style: TextStyle(
                            fontSize: 12,
                            color: Color(0xFF64748B),
                          ),
                        ),
                      ),
                      const SizedBox(height: 24),
                      // Stat cards row
                      Row(
                        children: [
                          _buildStatCard('20', 'Gestures'),
                          const SizedBox(width: 10),
                          _buildStatCard('10', 'Sentences'),
                          const SizedBox(width: 10),
                          _buildStatCard('3+', 'Languages'),
                        ],
                      ),
                    ],
                  ),
                ),
              ),
            ),
            // Start button
            Padding(
              padding: const EdgeInsets.fromLTRB(16, 0, 16, 16),
              child: GestureDetector(
                onTap: () => _navigateToRecognition(SignLanguageMode.isl),
                child: Opacity(
                  opacity: _isConnected ? 1.0 : 0.4,
                  child: Container(
                    width: double.infinity,
                    padding: const EdgeInsets.symmetric(vertical: 16),
                    decoration: BoxDecoration(
                      color: const Color(0xFF534AB7),
                      borderRadius: BorderRadius.circular(14),
                    ),
                    child: const Row(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        Icon(
                          Icons.play_arrow_rounded,
                          color: Colors.white,
                          size: 20,
                        ),
                        SizedBox(width: 8),
                        Text(
                          'Start recognition',
                          style: TextStyle(
                            fontSize: 14,
                            fontWeight: FontWeight.w500,
                            color: Colors.white,
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildStatCard(String value, String label) {
    return Expanded(
      child: Container(
        padding: const EdgeInsets.all(10),
        decoration: BoxDecoration(
          color: const Color(0xFF13132A),
          borderRadius: BorderRadius.circular(10),
          border: Border.all(
            color: const Color(0xFF2E2E50),
            width: 0.5,
          ),
        ),
        child: Column(
          children: [
            Text(
              value,
              style: const TextStyle(
                fontSize: 16,
                fontWeight: FontWeight.w500,
                color: Color(0xFFA78BFA),
              ),
            ),
            const SizedBox(height: 2),
            Text(
              label,
              style: const TextStyle(
                fontSize: 10,
                color: Color(0xFF64748B),
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
            child: const Text(
              '🤟 SignSpeak',
              style: TextStyle(
                fontSize: 26,
                fontWeight: FontWeight.w900,
                color: Colors.white,
              ),
            ),
          ),
          const Spacer(),
          IconButton(
            icon: Icon(
              Icons.settings_rounded,
              color: _isConnected
                  ? const Color(0xFF34D399)
                  : const Color(0xFF475569),
            ),
            onPressed: _showServerConfig,
          ),
        ],
      ),
    );
  }

  Widget _buildConnectionBar() {
    final color = _isConnected
        ? const Color(0xFF34D399)
        : _isConnecting
        ? const Color(0xFFFBBF24)
        : const Color(0xFFF87171);
    final label = _isConnected
        ? 'Connected to ${_backend.serverIp}'
        : _isConnecting
        ? 'Connecting…'
        : 'Tap to configure server';

    return GestureDetector(
      onTap: _isConnected || _isConnecting ? null : _showServerConfig,
      child: Container(
        margin: const EdgeInsets.fromLTRB(16, 8, 16, 4),
        padding: const EdgeInsets.symmetric(vertical: 9, horizontal: 14),
        decoration: BoxDecoration(
          color: color.withValues(alpha: 0.08),
          borderRadius: BorderRadius.circular(12),
          border: Border.all(color: color.withValues(alpha: 0.3)),
        ),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            if (_isConnecting)
              SizedBox(
                width: 13,
                height: 13,
                child: CircularProgressIndicator(strokeWidth: 2, color: color),
              )
            else
              Icon(
                _isConnected
                    ? Icons.cloud_done_rounded
                    : Icons.cloud_off_rounded,
                size: 15,
                color: color,
              ),
            const SizedBox(width: 8),
            Text(
              label,
              style: TextStyle(
                fontSize: 12,
                fontWeight: FontWeight.w600,
                color: color,
              ),
            ),
            if (_isConnected) ...[
              const Spacer(),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
                decoration: BoxDecoration(
                  color: color.withValues(alpha: 0.15),
                  borderRadius: BorderRadius.circular(20),
                ),
                child: Text(
                  '$_modelClasses classes',
                  style: TextStyle(
                    fontSize: 10,
                    color: color,
                    fontWeight: FontWeight.w700,
                  ),
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }
}
