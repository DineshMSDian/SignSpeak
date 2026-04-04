import 'dart:convert';
import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;

class TranslationResult {
  final String english;
  final String tamil;
  final String hindi;
  final String malayalam;

  const TranslationResult({
    required this.english,
    required this.tamil,
    required this.hindi,
    required this.malayalam,
  });

  static const empty = TranslationResult(english: '', tamil: '', hindi: '', malayalam: '');

  bool get isEmpty => english.isEmpty && tamil.isEmpty && hindi.isEmpty && malayalam.isEmpty;
  bool get isNotEmpty => !isEmpty;
}

class GeminiService {
  static const String _apiKeyPart1 = 'gsk_LwFFenvhMAMw';
  static const String _apiKeyPart2 = 'cCU65kSfWGdyb3FY';
  static const String _apiKeyPart3 = 'XuYBcg1TlTXQ6CazDyBF6iVi';
  static String get _apiKey => _apiKeyPart1 + _apiKeyPart2 + _apiKeyPart3;
  static const String _endpoint = 'https://api.groq.com/openai/v1/chat/completions';


  static Future<TranslationResult> translateSequence(
    List<String> sequence,
    bool isASL,
  ) async {
    if (sequence.isEmpty) return TranslationResult.empty;

    final input = sequence.join(isASL ? '' : ' ').toLowerCase().trim();

    final prompt =
        'The following is a sequence of ${isASL ? 'ASL alphabet letters' : 'ISL gesture words'} '
        'detected from a sign language recognition system: $input. '
        'Convert this into a natural, grammatically correct English sentence. '
        'Fix grammatical errors, remove any duplicate or redundant words, and make it sound natural. '
        'Also provide direct translations in Tamil, Hindi, and Malayalam. Do NOT add any extra conversational text or notes. '
        'Respond ONLY as JSON: {"english": "...", "tamil": "...", "hindi": "...", "malayalam": "..."}';

    try {
      final res = await http
          .post(
            Uri.parse(_endpoint),
            headers: {
              'Authorization': 'Bearer $_apiKey',
              'Content-Type': 'application/json'
            },
            body: jsonEncode({
              'model': 'llama-3.1-8b-instant',
              'messages': [
                {'role': 'user', 'content': prompt}
              ],
              'temperature': 0.2,
              'response_format': {'type': 'json_object'}
            }),
          )
          .timeout(const Duration(seconds: 12));

      if (res.statusCode == 200) {
        final data = jsonDecode(res.body);
        final text = data['choices']?[0]?['message']?['content'] as String? ?? '';
        final clean = text.replaceAll(RegExp(r'```json|```'), '').trim();
        final parsed = jsonDecode(clean) as Map<String, dynamic>;

        return TranslationResult(
          english: parsed['english'] as String? ?? input,
          tamil: parsed['tamil'] as String? ?? '',
          hindi: parsed['hindi'] as String? ?? '',
          malayalam: parsed['malayalam'] as String? ?? '',
        );
      } else {
        debugPrint('[GroqService] Error ${res.statusCode}: ${res.body}');
      }
    } catch (e) {
      debugPrint('[GroqService] Exception: $e');
    }

    return TranslationResult(english: input, tamil: '', hindi: '', malayalam: '');
  }
}
