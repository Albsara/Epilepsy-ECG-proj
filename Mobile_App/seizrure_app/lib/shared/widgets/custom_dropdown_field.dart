import 'package:flutter/material.dart';
import 'package:get/get.dart';
import '../utils/app_colors.dart';
import 'app_style.dart';

class CustomDropdownField extends StatelessWidget {
  final String label;
  final String value;
  final List<String> items;
  final List<String> itemLabels;
  final IconData icon;
  final Function(String?) onChanged;

  const CustomDropdownField({
    super.key,
    required this.label,
    required this.value,
    required this.items,
    required this.itemLabels,
    required this.icon,
    required this.onChanged,
  });

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          label,
          style: const TextStyle(
            fontSize: 13,
            fontWeight: FontWeight.w700,
            color: AppColors.textMedium,
          ),
        ),
        const SizedBox(height: 8),
        Container(
          height: 56,
          decoration: BoxDecoration(
            color: Colors.white,
            borderRadius: BorderRadius.circular(AppStyle.radius),
            boxShadow: AppStyle.shadow,
          ),
          child: DropdownButtonFormField<String>(
            value: items.contains(value) ? value : items.first,
            icon: const Icon(
              Icons.keyboard_arrow_down_rounded,
              color: AppColors.textMedium,
            ),
            style: const TextStyle(
              fontWeight: FontWeight.w600,
              color: AppColors.textDark,
              fontSize: 15,
            ),
            decoration: InputDecoration(
              prefixIcon: Icon(
                icon,
                color: AppColors.primaryGradientStart,
                size: 20,
              ),
              border: InputBorder.none,
              contentPadding: const EdgeInsets.fromLTRB(0, 4, 12, 4),
            ),
            items: List.generate(items.length, (index) {
              return DropdownMenuItem<String>(
                value: items[index],
                child: Text(itemLabels[index].tr),
              );
            }),
            onChanged: onChanged,
          ),
        ),
      ],
    );
  }
}
