import 'package:flutter/material.dart';
import 'primary_button.dart';

class GradientButton extends StatelessWidget {
  const GradientButton({
    super.key,
    required this.text,
    required this.height,
    required this.radius,
    required this.colors,
    required this.onTap,
  });

  final String text;
  final double height;
  final double radius;
  final List<Color> colors;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    return PrimaryButton(text: text, onTap: onTap, height: height);
  }
}
