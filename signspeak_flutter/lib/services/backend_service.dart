import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:web_socket_channel/web_socket_channel.dart';

class PredictionResult {
  final String? prediction;
  final double confidence;
  final String? rawLabel;
  final int numHands;
  final int bufferFill;
  final bool bufferReady;
  final double processingMs;
  final String mode;
  final List<List<Map<String, double>>> landmarks;

  PredictionResult({
    this.prediction,
    this.confidence = 0.0,
    this.rawLabel,
    this.numHands = 0,
    this.bufferFill = 0,
    this.bufferReady = false,
    this.processingMs = 0.0,
    this.mode = 'ASL',
    this.landmarks = const [],
  });

  factory PredictionResult.fromJson(Map<String, dynamic> json) {
    List<List<Map<String, double>>> parsedLandmarks = [];
    if (json['landmarks'] != null) {
      for (var hand in json['landmarks']) {
        List<Map<String, double>> points = [];
        for (var pt in hand) {
          points.add({
            'x': (pt['x'] as num).toDouble(),
            'y': (pt['y'] as num).toDouble(),
          });
        }
        parsedLandmarks.add(points);
      }
    }
    return PredictionResult(
      prediction: json['prediction'] as String?,
      confidence: (json['confidence'] as num?)?.toDouble() ?? 0.0,
      rawLabel: json['raw_label'] as String?,
      numHands: json['num_hands'] as int? ?? 0,
      bufferFill: json['buffer_fill'] as int? ?? 0,
      bufferReady: json['buffer_ready'] as bool? ?? false,
      processingMs: (json['processing_ms'] as num?)?.toDouble() ?? 0.0,
      mode: json['mode'] as String? ?? 'ASL',
      landmarks: parsedLandmarks,
    );
  }
}

class BackendService {
  WebSocketChannel? _channel;
  String _serverIp = '192.168.1.100';
  int _serverPort = 8000;
  bool _isConnected = false;

  final _predictionController = StreamController<PredictionResult>.broadcast();
  Stream<PredictionResult> get predictions => _predictionController.stream;

  final _connectionController = StreamController<bool>.broadcast();
  Stream<bool> get connectionState => _connectionController.stream;

  bool get isConnected => _isConnected;
  String get serverIp => _serverIp;
  int get serverPort => _serverPort;
  String get wsUrl => 'ws://$_serverIp:$_serverPort/ws';

  void setServer(String ip, {int port = 8000}) {
    _serverIp = ip;
    _serverPort = port;
  }

  Future<bool> connect() async {
    try {
      disconnect();

      debugPrint('[Backend] Connecting to $wsUrl');
      _channel = WebSocketChannel.connect(Uri.parse(wsUrl));
      await _channel!.ready;

      _isConnected = true;
      _connectionController.add(true);
      debugPrint('[Backend] Connected!');

      _channel!.stream.listen(
        (data) {
          try {
            final json = jsonDecode(data as String) as Map<String, dynamic>;
            _predictionController.add(PredictionResult.fromJson(json));
          } catch (e) {
            debugPrint('[Backend] Parse error: $e');
          }
        },
        onError: (error) {
          debugPrint('[Backend] Stream error: $error');
          _handleDisconnect();
        },
        onDone: () {
          debugPrint('[Backend] Connection closed');
          _handleDisconnect();
        },
      );

      return true;
    } catch (e) {
      debugPrint('[Backend] Connection failed: $e');
      _isConnected = false;
      _connectionController.add(false);
      return false;
    }
  }

  void sendFrame(Uint8List jpegBytes) {
    if (!_isConnected || _channel == null) return;
    try {
      _channel!.sink.add(jpegBytes);
    } catch (e) {
      debugPrint('[Backend] Send error: $e');
      _handleDisconnect();
    }
  }

  Future<Map<String, dynamic>?> setMode(String mode) async {
    try {
      final client = HttpClient();
      client.connectionTimeout = const Duration(seconds: 3);
      final request = await client.postUrl(
        Uri.parse('http://$_serverIp:$_serverPort/set-mode/$mode'),
      );
      final response = await request.close();
      final body = await response.transform(utf8.decoder).join();
      client.close();
      return jsonDecode(body) as Map<String, dynamic>;
    } catch (e) {
      debugPrint('[Backend] Set mode error: $e');
      return null;
    }
  }

  Future<Map<String, dynamic>?> healthCheck() async {
    try {
      final client = HttpClient();
      client.connectionTimeout = const Duration(seconds: 3);
      final request = await client.getUrl(
        Uri.parse('http://$_serverIp:$_serverPort/health'),
      );
      final response = await request.close();
      final body = await response.transform(utf8.decoder).join();
      client.close();
      return jsonDecode(body) as Map<String, dynamic>;
    } catch (e) {
      debugPrint('[Backend] Health check failed: $e');
      return null;
    }
  }

  void _handleDisconnect() {
    _isConnected = false;
    _connectionController.add(false);
  }

  void disconnect() {
    try {
      _channel?.sink.close();
    } catch (_) {}
    _channel = null;
    _isConnected = false;
    _connectionController.add(false);
  }

  void dispose() {
    disconnect();
    _predictionController.close();
    _connectionController.close();
  }
}
