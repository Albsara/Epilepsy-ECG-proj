// This is a basic Flutter widget test.
//
// To perform an interaction with a widget in your test, use the WidgetTester
// utility in the flutter_test package. For example, you can send tap and scroll
// gestures. You can also use WidgetTester to find child widgets in the widget
// tree, read text, and verify that the values of widget properties are correct.

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:get/get_navigation/src/root/get_material_app.dart';

import 'package:seizrure_app/main.dart';

void main() {
  testWidgets('Smoke test', (WidgetTester tester) async {
    // Build our app and trigger a frame.
    // Note: This will likely fail in automated test environments without a Firebase mock,
    // but at least the code will be syntactically correct.
    await tester.pumpWidget(const App());

    // Basic check that something is rendered
    expect(find.byType(GetMaterialApp), findsOneWidget);
  });
}
