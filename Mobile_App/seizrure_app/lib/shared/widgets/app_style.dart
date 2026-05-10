import 'package:flutter/material.dart';

class AppStyle {
  static const double radius = 16.0;
  static const double innerRadius = 12.0;

  static List<BoxShadow> get shadow => [
    BoxShadow(
      color: Colors.black.withOpacity(0.04),
      blurRadius: 10,
      offset: const Offset(0, 4),
    ),
  ];

  static List<BoxShadow> get lightShadow => [
    BoxShadow(
      color: Colors.black.withOpacity(0.02),
      blurRadius: 8,
      offset: const Offset(0, 2),
    ),
  ];
}
