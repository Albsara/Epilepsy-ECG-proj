import 'package:flutter/material.dart';
import 'app_style.dart';
import 'primary_button.dart';

class ActionRowButton extends StatelessWidget {
  final List<Color> iconBg;
  final IconData icon;
  final List<Color> buttonBg;
  final String text;
  final VoidCallback onTap;

  const ActionRowButton({
    super.key,
    required this.iconBg,
    required this.icon,
    required this.buttonBg,
    required this.text,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        Container(
          width: 32,
          height: 32,
          decoration: BoxDecoration(
            shape: BoxShape.circle,
            gradient: LinearGradient(colors: iconBg),
            boxShadow: AppStyle.lightShadow,
          ),
          child: Icon(icon, size: 18, color: Colors.white),
        ),
        const SizedBox(width: 12),
        Expanded(
          child: PrimaryButton(text: text, onTap: onTap, height: 44),
        ),
      ],
    );
  }
}
