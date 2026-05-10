import 'package:flutter/material.dart';
import 'app_style.dart';

class MiniIconButton extends StatelessWidget {
  final IconData icon;
  final VoidCallback onTap;

  const MiniIconButton({super.key, required this.icon, required this.onTap});

  @override
  Widget build(BuildContext context) {
    return InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(AppStyle.innerRadius),
      child: Container(
        width: 36,
        height: 36,
        decoration: BoxDecoration(
          color: const Color(0xFFF1F5F9),
          borderRadius: BorderRadius.circular(AppStyle.innerRadius),
          border: Border.all(color: const Color(0xFFE7EEF6)),
        ),
        child: Icon(icon, size: 18, color: const Color(0xFF7C8A99)),
      ),
    );
  }
}
