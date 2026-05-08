import 'package:flutter/material.dart';
import '../config.dart';

class ServerConfigDialog extends StatefulWidget {
  final String currentIp;
  final int currentPort;
  final Function(String ip, int port) onSave;

  const ServerConfigDialog({
    super.key,
    required this.currentIp,
    required this.currentPort,
    required this.onSave,
  });

  @override
  State<ServerConfigDialog> createState() => _ServerConfigDialogState();
}

class _ServerConfigDialogState extends State<ServerConfigDialog> {
  late TextEditingController _ipController;
  late TextEditingController _portController;

  @override
  void initState() {
    super.initState();
    _ipController = TextEditingController(text: widget.currentIp);
    _portController = TextEditingController(
      text: widget.currentPort.toString(),
    );
  }

  @override
  void dispose() {
    _ipController.dispose();
    _portController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return AlertDialog(
      backgroundColor: const Color(0xFF1E1B4B),
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
      title: const Row(
        children: [
          Icon(Icons.wifi, color: Color(0xFF6366F1)),
          SizedBox(width: 10),
          Text(
            'Server Settings',
            style: TextStyle(
              color: Color(0xFFE0E7FF),
              fontWeight: FontWeight.w700,
            ),
          ),
        ],
      ),
      content: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          const Text(
            'Enter the IP address shown in the Python server terminal.',
            style: TextStyle(color: Color(0xFF94A3B8), fontSize: 13),
          ),
          const SizedBox(height: 16),
          TextField(
            controller: _ipController,
            style: const TextStyle(
              color: Colors.white,
              fontSize: 18,
              fontWeight: FontWeight.w600,
            ),
            keyboardType: TextInputType.number,
            decoration: InputDecoration(
              labelText: 'Server IP',
              labelStyle: const TextStyle(color: Color(0xFF818CF8)),
              hintText: defaultServerIp,
              hintStyle: TextStyle(color: Colors.white.withValues(alpha: 0.2)),
              filled: true,
              fillColor: const Color(0xFF312E81),
              border: OutlineInputBorder(
                borderRadius: BorderRadius.circular(12),
                borderSide: const BorderSide(color: Color(0xFF6366F1)),
              ),
              focusedBorder: OutlineInputBorder(
                borderRadius: BorderRadius.circular(12),
                borderSide: const BorderSide(
                  color: Color(0xFF6366F1),
                  width: 2,
                ),
              ),
            ),
          ),
          const SizedBox(height: 12),
          TextField(
            controller: _portController,
            style: const TextStyle(
              color: Colors.white,
              fontSize: 18,
              fontWeight: FontWeight.w600,
            ),
            keyboardType: TextInputType.number,
            decoration: InputDecoration(
              labelText: 'Port',
              labelStyle: const TextStyle(color: Color(0xFF818CF8)),
              hintText: '8000',
              hintStyle: TextStyle(color: Colors.white.withValues(alpha: 0.2)),
              filled: true,
              fillColor: const Color(0xFF312E81),
              border: OutlineInputBorder(
                borderRadius: BorderRadius.circular(12),
                borderSide: const BorderSide(color: Color(0xFF6366F1)),
              ),
              focusedBorder: OutlineInputBorder(
                borderRadius: BorderRadius.circular(12),
                borderSide: const BorderSide(
                  color: Color(0xFF6366F1),
                  width: 2,
                ),
              ),
            ),
          ),
        ],
      ),
      actions: [
        TextButton(
          onPressed: () => Navigator.pop(context),
          child: const Text(
            'Cancel',
            style: TextStyle(color: Color(0xFF94A3B8)),
          ),
        ),
        ElevatedButton(
          onPressed: () {
            final ip = _ipController.text.trim();
            final port =
                int.tryParse(_portController.text.trim()) ?? defaultServerPort;
            if (ip.isNotEmpty) {
              widget.onSave(ip, port);
              Navigator.pop(context);
            }
          },
          style: ElevatedButton.styleFrom(
            backgroundColor: const Color(0xFF6366F1),
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(12),
            ),
          ),
          child: const Text('Connect'),
        ),
      ],
    );
  }
}
