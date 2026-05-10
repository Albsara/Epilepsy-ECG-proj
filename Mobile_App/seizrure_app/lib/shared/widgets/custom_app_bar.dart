import 'package:flutter/material.dart';
import 'package:get/get.dart';
import '../utils/app_colors.dart';
import 'package:seizrure_app/shared/controllers/app_controller.dart';
import 'app_style.dart';

class CustomAppBar extends StatelessWidget {
  final String title;
  final bool isKey;
  final bool showBackButton;
  final bool showLangToggle;
  final bool showRefresh;
  final VoidCallback? onRefresh;
  final List<Widget>? actions;
  final Color? titleColor;

  const CustomAppBar({
    super.key,
    required this.title,
    this.isKey = true,
    this.showBackButton = false,
    this.showLangToggle = true,
    this.showRefresh = false,
    this.onRefresh,
    this.actions,
    this.titleColor,
  });

  @override
  Widget build(BuildContext context) {
    final controller = Get.find<AppController>();
    return Row(
      children: [
        if (showBackButton)
          AppBarButton(
            icon: Icons.arrow_back_ios_new_rounded,
            onTap: () => Get.back(),
          ),
        if (showBackButton) const SizedBox(width: 12),
        Text(
          isKey ? title.tr : title,
          style: TextStyle(
            fontSize: 18,
            fontWeight: FontWeight.w800,
            color: titleColor ?? AppColors.textDark,
            letterSpacing: -0.5,
          ),
        ),
        const Spacer(),
        if (actions != null) ...actions!,
        if (showRefresh) ...[
          const SizedBox(width: 8),
          AppBarButton(
            icon: Icons.refresh_rounded,
            onTap: onRefresh ?? () {},
          ),
        ],
        if (showLangToggle) ...[
          const SizedBox(width: 8),
          AppBarButton(
            icon: Icons.language_rounded,
            onTap: controller.toggleLang,
          ),
        ],
      ],
    );
  }
}

class AppBarButton extends StatelessWidget {
  final IconData icon;
  final VoidCallback onTap;
  final bool badge;
  final Color? iconColor;

  const AppBarButton({
    super.key,
    required this.icon,
    required this.onTap,
    this.badge = false,
    this.iconColor,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Stack(
        children: [
          Container(
            padding: const EdgeInsets.all(10),
            decoration: BoxDecoration(
              color: Colors.white,
              shape: BoxShape.circle,
              boxShadow: AppStyle.lightShadow,
            ),
            child: Icon(icon, size: 20, color: iconColor ?? AppColors.textDark),
          ),
          if (badge)
            Positioned(
              right: 8,
              top: 8,
              child: Container(
                width: 8,
                height: 8,
                decoration: BoxDecoration(
                  color: AppColors.danger,
                  shape: BoxShape.circle,
                  border: Border.all(color: Colors.white, width: 1.5),
                ),
              ),
            ),
        ],
      ),
    );
  }
}
