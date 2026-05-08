import 'package:flutter/material.dart';

/// Matches the CSS .prediction-box styling from the Streamlit app.
class PredictionCard extends StatelessWidget {
  final String prediction;
  final double confidence;
  final bool isLive;
  final String mode;

  const PredictionCard({
    super.key,
    required this.prediction,
    required this.confidence,
    this.isLive = false,
    this.mode = 'ASL',
  });

  @override
  Widget build(BuildContext context) {
    final confPercent = (confidence * 100).toStringAsFixed(0);
    final confColor = confidence >= 0.7
        ? const Color(0xFF34D399)
        : const Color(0xFFF87171);

    return Container(
      margin: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      padding: const EdgeInsets.symmetric(vertical: 20, horizontal: 24),
      decoration: BoxDecoration(
        gradient: const LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [Color(0xFF1E1B4B), Color(0xFF312E81)],
        ),
        border: Border.all(color: const Color(0xFF6366F1), width: 2),
        borderRadius: BorderRadius.circular(20),
        boxShadow: [
          BoxShadow(
            color: const Color(0xFF6366F1).withValues(alpha: 0.15),
            blurRadius: 32,
            offset: const Offset(0, 8),
          ),
        ],
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Text(
            isLive ? '🔴 LIVE — $mode Prediction' : 'Last Prediction',
            style: const TextStyle(
              fontSize: 11,
              color: Color(0xFF818CF8),
              fontWeight: FontWeight.w600,
              letterSpacing: 3,
            ),
          ),
          const SizedBox(height: 8),

          AnimatedSwitcher(
            duration: const Duration(milliseconds: 200),
            child: Text(
              prediction.toUpperCase(),
              key: ValueKey(prediction),
              style: const TextStyle(
                fontSize: 40,
                fontWeight: FontWeight.w800,
                color: Color(0xFFE0E7FF),
                letterSpacing: 2,
              ),
              textAlign: TextAlign.center,
            ),
          ),
          const SizedBox(height: 6),

          Text(
            'Confidence: $confPercent%',
            style: TextStyle(
              fontSize: 14,
              color: confColor,
              fontWeight: FontWeight.w500,
            ),
          ),
        ],
      ),
    );
  }
}
