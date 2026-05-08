import 'package:flutter/material.dart';

class SentenceBar extends StatelessWidget {
  final List<String> words;
  final VoidCallback onSpeak;
  final VoidCallback onUndo;
  final VoidCallback onClear;

  const SentenceBar({
    super.key,
    required this.words,
    required this.onSpeak,
    required this.onUndo,
    required this.onClear,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        gradient: const LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [Color(0xFF0F172A), Color(0xFF1E293B)],
        ),
        border: Border.all(color: const Color(0xFF334155)),
        borderRadius: BorderRadius.circular(20),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.2),
            blurRadius: 16,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisSize: MainAxisSize.min,
        children: [
          const Text(
            '📝 YOUR SENTENCE',
            style: TextStyle(
              color: Color(0xFF64748B),
              fontSize: 11,
              fontWeight: FontWeight.w600,
              letterSpacing: 2,
            ),
          ),
          const SizedBox(height: 10),

          if (words.isEmpty)
            const Text(
              'Words will appear here as you sign...',
              style: TextStyle(
                color: Color(0xFF475569),
                fontStyle: FontStyle.italic,
                fontSize: 14,
              ),
            )
          else ...[
            Wrap(
              spacing: 6,
              runSpacing: 6,
              children: words
                  .map(
                    (w) => Container(
                      padding: const EdgeInsets.symmetric(
                        horizontal: 12,
                        vertical: 6,
                      ),
                      decoration: BoxDecoration(
                        color: const Color(0xFF312E81),
                        border: Border.all(color: const Color(0xFF4338CA)),
                        borderRadius: BorderRadius.circular(10),
                      ),
                      child: Text(
                        w,
                        style: const TextStyle(
                          color: Color(0xFFC7D2FE),
                          fontSize: 14,
                          fontWeight: FontWeight.w500,
                        ),
                      ),
                    ),
                  )
                  .toList(),
            ),
            const SizedBox(height: 10),

            Container(
              padding: const EdgeInsets.only(top: 8),
              decoration: const BoxDecoration(
                border: Border(top: BorderSide(color: Color(0xFF334155))),
              ),
              child: Text(
                '"${words.join(' ')}"',
                style: const TextStyle(
                  color: Color(0xFFF1F5F9),
                  fontSize: 15,
                  fontWeight: FontWeight.w500,
                  height: 1.5,
                ),
              ),
            ),
          ],

          const SizedBox(height: 14),

          Row(
            children: [
              Expanded(
                child: _ActionButton(
                  icon: Icons.volume_up_rounded,
                  label: 'Speak',
                  color: const Color(0xFF6366F1),
                  onPressed: words.isEmpty ? null : onSpeak,
                ),
              ),
              const SizedBox(width: 8),
              Expanded(
                child: _ActionButton(
                  icon: Icons.undo_rounded,
                  label: 'Undo',
                  color: const Color(0xFF475569),
                  onPressed: words.isEmpty ? null : onUndo,
                ),
              ),
              const SizedBox(width: 8),
              Expanded(
                child: _ActionButton(
                  icon: Icons.delete_outline_rounded,
                  label: 'Clear',
                  color: const Color(0xFF475569),
                  onPressed: words.isEmpty ? null : onClear,
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }
}

class _ActionButton extends StatelessWidget {
  final IconData icon;
  final String label;
  final Color color;
  final VoidCallback? onPressed;

  const _ActionButton({
    required this.icon,
    required this.label,
    required this.color,
    this.onPressed,
  });

  @override
  Widget build(BuildContext context) {
    final isDisabled = onPressed == null;
    return Material(
      color: Colors.transparent,
      child: InkWell(
        onTap: onPressed,
        borderRadius: BorderRadius.circular(12),
        child: Container(
          padding: const EdgeInsets.symmetric(vertical: 10),
          decoration: BoxDecoration(
            color: isDisabled
                ? const Color(0xFF1E293B)
                : color.withValues(alpha: 0.15),
            borderRadius: BorderRadius.circular(12),
            border: Border.all(
              color: isDisabled
                  ? const Color(0xFF334155)
                  : color.withValues(alpha: 0.3),
            ),
          ),
          child: Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(
                icon,
                size: 16,
                color: isDisabled ? const Color(0xFF475569) : color,
              ),
              const SizedBox(width: 6),
              Text(
                label,
                style: TextStyle(
                  fontSize: 12,
                  fontWeight: FontWeight.w600,
                  color: isDisabled ? const Color(0xFF475569) : color,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
