import 'package:flutter/material.dart';
import 'package:get/get.dart';
import '../utils/app_colors.dart';
import 'common_widgets.dart';

class AppDialogs {
  static Future<void> success({String? title, required String message}) async {
    await Get.dialog(
      _BaseDialog(
        title: title ?? 'success'.tr,
        message: message,
        icon: Icons.check_circle,
        iconColor: AppColors.accent,
        actions: [
          PrimaryButton(
            text: 'ok'.tr,
            onPressed: () => Get.back(),
            fullWidth: true,
          ),
        ],
      ),
    );
  }

  static Future<void> error({String? title, required String message}) async {
    await Get.dialog(
      _BaseDialog(
        title: title ?? 'error'.tr,
        message: message,
        icon: Icons.error_outline,
        iconColor: AppColors.alert,
        actions: [
          PrimaryButton(
            text: 'ok'.tr,
            onPressed: () => Get.back(),
            fullWidth: true,
          ),
        ],
      ),
    );
  }

  static Future<void> warning({String? title, required String message}) async {
    await Get.dialog(
      _BaseDialog(
        title: title ?? 'warning'.tr,
        message: message,
        icon: Icons.warning_amber_rounded,
        iconColor: Colors.orange,
        actions: [
          PrimaryButton(
            text: 'ok'.tr,
            onPressed: () => Get.back(),
            fullWidth: true,
          ),
        ],
      ),
    );
  }

  static Future<bool> confirm({
    String? title,
    required String message,
    String? confirmText,
    String? cancelText,
    bool isDanger = false,
  }) async {
    final result = await Get.dialog<bool>(
      _BaseDialog(
        title: title ?? 'confirm_title'.tr,
        message: message,
        icon: isDanger ? Icons.report_problem : Icons.help_outline,
        iconColor: isDanger ? AppColors.alert : AppColors.accent,
        actions: [
          Row(
            children: [
              Expanded(
                child: OutlineActionButton(
                  text: cancelText ?? 'cancel'.tr,
                  onPressed: () => Get.back(result: false),
                ),
              ),
              const SizedBox(width: 10),
              Expanded(
                child: PrimaryButton(
                  text: confirmText ?? 'yes'.tr,
                  onPressed: () => Get.back(result: true),
                ),
              ),
            ],
          ),
        ],
      ),
    );
    return result ?? false;
  }
}

class _BaseDialog extends StatelessWidget {
  final String title;
  final String message;
  final IconData icon;
  final Color iconColor;
  final List<Widget> actions;

  const _BaseDialog({
    required this.title,
    required this.message,
    required this.icon,
    required this.iconColor,
    required this.actions,
  });

  @override
  Widget build(BuildContext context) {
    return Center(
      child: ConstrainedBox(
        constraints: const BoxConstraints(maxWidth: 400),
        child: Material(
          color: Colors.transparent,
          child: Container(
            margin: const EdgeInsets.all(24),
            padding: const EdgeInsets.all(24),
            decoration: BoxDecoration(
              color: Colors.white,
              borderRadius: BorderRadius.circular(20),
              boxShadow: [
                BoxShadow(
                  color: Colors.black.withOpacity(0.1),
                  blurRadius: 20,
                  offset: const Offset(0, 10),
                ),
              ],
            ),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              crossAxisAlignment: CrossAxisAlignment.center,
              children: [
                Container(
                  padding: const EdgeInsets.all(16),
                  decoration: BoxDecoration(
                    color: iconColor.withOpacity(0.1),
                    shape: BoxShape.circle,
                  ),
                  child: Icon(icon, color: iconColor, size: 32),
                ),
                const SizedBox(height: 16),
                Text(
                  title,
                  style: const TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.w900,
                    color: AppColors.text,
                  ),
                  textAlign: TextAlign.center,
                ),
                const SizedBox(height: 8),
                Text(
                  message,
                  style: const TextStyle(
                    fontSize: 14,
                    fontWeight: FontWeight.w700,
                    color: AppColors.muted,
                  ),
                  textAlign: TextAlign.center,
                ),
                const SizedBox(height: 24),
                ...actions,
              ],
            ),
          ),
        ),
      ),
    );
  }
}
