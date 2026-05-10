import 'package:flutter/material.dart';
import '../utils/app_colors.dart';
import 'app_style.dart';

class SectionCard extends StatelessWidget {
  final String title;
  final Widget child;
  final double? radius;

  const SectionCard({
    super.key,
    required this.title,
    required this.child,
    this.radius,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(radius ?? AppStyle.radius),
        boxShadow: AppStyle.shadow,
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            title,
            style: const TextStyle(
              fontSize: 13,
              fontWeight: FontWeight.w800,
              color: AppColors.textMedium,
            ),
          ),
          const SizedBox(height: 12),
          child,
        ],
      ),
    );
  }
}
