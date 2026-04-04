import 'package:flutter_test/flutter_test.dart';
import 'package:signspeak/main.dart';

void main() {
  testWidgets('SignSpeak app loads', (WidgetTester tester) async {
    await tester.pumpWidget(const SignSpeakApp());
    expect(find.text('🤟 SignSpeak'), findsOneWidget);
  });
}
