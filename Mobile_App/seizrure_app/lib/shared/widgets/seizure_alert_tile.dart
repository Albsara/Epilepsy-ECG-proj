import 'package:flutter/material.dart';
import 'package:get/get.dart';
import 'package:seizrure_app/shared/controllers/app_controller.dart';
import '../utils/app_colors.dart';

class SeizureAlertTile extends StatelessWidget {
  final Map<String, String> item;
  final bool isCaregiver;
  final VoidCallback? onHandled;

  const SeizureAlertTile({
    super.key,
    required this.item,
    required this.isCaregiver,
    this.onHandled,
  });

  @override
  Widget build(BuildContext context) {
    final bool isHandled = item['is_handled'] == 'true';

    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(16),
        boxShadow: const [
          BoxShadow(
            color: AppColors.shadowColor,
            blurRadius: 10,
            offset: Offset(0, 4),
          ),
        ],
        border: isHandled
            ? Border.all(color: const Color(0xFFDCFCE7), width: 1)
            : Border.all(color: Colors.transparent, width: 1),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.center,
        children: [
          // Left Icon Indicator
          Container(
            width: 44,
            height: 44,
            decoration: BoxDecoration(
              color: isHandled
                  ? const Color(0xFFDCFCE7)
                  : const Color(0xFFFEE2E2),
              shape: BoxShape.circle,
            ),
            child: Icon(
              isHandled
                  ? Icons.check_circle_rounded
                  : Icons.warning_amber_rounded,
              color: isHandled ? const Color(0xFF166534) : AppColors.danger,
              size: 22,
            ),
          ),
          const SizedBox(width: 14),

          // Data Section
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                if (isCaregiver && item['patientId'] != null)
                  Padding(
                    padding: const EdgeInsets.only(bottom: 2),
                    child: Obx(() {
                      final controller = Get.find<AppController>();
                      final name = controller.patientNames[item['patientId']];

                      // Only show if we have a name, to avoid showing empty space or placeholder word
                      if (name == null || name.isEmpty) {
                        return const SizedBox.shrink();
                      }

                      return Text(
                        name,
                        style: const TextStyle(
                          fontWeight: FontWeight.w900,
                          fontSize: 16,
                          color: AppColors.primaryGradientEnd,
                          letterSpacing: -0.3,
                        ),
                      );
                    }),
                  ),
                Text(
                  item['time'] ?? '',
                  style: const TextStyle(
                    fontWeight: FontWeight.w700,
                    fontSize: 13,
                    color: AppColors.textDark,
                  ),
                ),
                const SizedBox(height: 6),
                SingleChildScrollView(
                  scrollDirection: Axis.horizontal,
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Row(
                        children: [
                          _buildMetricChip(
                            Icons.favorite_rounded,
                            'hr_value'.trParams({'value': item['hr'] ?? '-'}),
                            isHandled
                                ? const Color(0xFF166534).withOpacity(0.7)
                                : AppColors.danger.withOpacity(0.7),
                          ),
                          const SizedBox(width: 8),
                          _buildMetricChip(
                            Icons.speed_rounded,
                            'hrv_value'.trParams({'value': item['hrv'] ?? '-'}),
                            AppColors.textMedium,
                          ),
                          const SizedBox(width: 8),
                          _buildMetricChip(
                            Icons.medication_outlined,
                            '${'medication'.tr}: ${item['medication']?.tr ?? '-'}',
                            const Color(0xFF64748B),
                          ),
                        ],
                      ),
                      const SizedBox(height: 4),
                      Row(
                        children: [
                          _buildMetricChip(
                            Icons.warning_amber_rounded,
                            '${'symptoms'.tr}: ${item['symptoms']?.tr ?? '-'}',
                            const Color(0xFF64748B),
                          ),
                          const SizedBox(width: 8),
                          _buildMetricChip(
                            Icons.bedtime_outlined,
                            '${'sleep'.tr}: ${item['sleep']?.tr ?? '-'}',
                            const Color(0xFF64748B),
                          ),
                          const SizedBox(width: 8),
                          _buildMetricChip(
                            Icons.psychology_outlined,
                            '${'stress'.tr}: ${item['stress']?.tr ?? '-'}',
                            const Color(0xFF64748B),
                          ),
                        ],
                      ),
                    ],
                  ),
                ),
              ],
            ),
          ),

          const SizedBox(width: 8),

          // Status Badge / Handle Button
          if (isCaregiver)
            GestureDetector(
              onTap: isHandled ? null : onHandled,
              child: AnimatedContainer(
                duration: const Duration(milliseconds: 300),
                padding: const EdgeInsets.symmetric(
                  horizontal: 12,
                  vertical: 8,
                ),
                decoration: BoxDecoration(
                  color: isHandled
                      ? const Color(0xFFDCFCE7)
                      : const Color(0xFF991B1B),
                  borderRadius: BorderRadius.circular(20),
                  boxShadow: isHandled
                      ? null
                      : [
                          BoxShadow(
                            color: const Color(0xFF991B1B).withOpacity(0.3),
                            blurRadius: 8,
                            offset: const Offset(0, 3),
                          ),
                        ],
                ),
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Icon(
                      isHandled ? Icons.check_rounded : Icons.touch_app_rounded,
                      size: 14,
                      color: isHandled ? const Color(0xFF166534) : Colors.white,
                    ),
                    const SizedBox(width: 4),
                    Text(
                      isHandled ? 'Handled'.tr : 'handle'.tr,
                      style: TextStyle(
                        fontSize: 11,
                        fontWeight: FontWeight.w800,
                        color: isHandled
                            ? const Color(0xFF166534)
                            : Colors.white,
                      ),
                    ),
                  ],
                ),
              ),
            )
          else if (isHandled)
            // Simple display for patient (Only show handled)
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
              decoration: BoxDecoration(
                color: const Color(0xFFDCFCE7),
                borderRadius: BorderRadius.circular(20),
              ),
              child: Text(
                'Handled'.tr,
                style: const TextStyle(
                  fontSize: 10,
                  fontWeight: FontWeight.w800,
                  color: Color(0xFF166534),
                ),
              ),
            ),
        ],
      ),
    );
  }

  Widget _buildMetricChip(IconData icon, String text, Color color) {
    return Row(
      children: [
        Icon(icon, size: 10, color: color),
        const SizedBox(width: 4),
        Text(
          text,
          style: TextStyle(
            fontSize: 11,
            fontWeight: FontWeight.w600,
            color: color,
          ),
        ),
      ],
    );
  }
}
