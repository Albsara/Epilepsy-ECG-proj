import 'package:flutter/material.dart';
import 'package:get/get.dart';
import 'package:intl/intl.dart';
import '../controllers/data_controller.dart';
import '../models/models.dart';
import '../utils/app_colors.dart';
import '../widgets/admin_shell.dart';
import '../widgets/common_widgets.dart';

class AlertsHistoryPage extends StatelessWidget {
  const AlertsHistoryPage({super.key});

  @override
  Widget build(BuildContext context) {
    final data = Get.find<DataController>();

    return AdminShell(
      title: 'alerts_history'.tr,
      child: Obx(() {
        if (data.globalAlerts.isEmpty) {
          return Center(
            child: Padding(
              padding: const EdgeInsets.all(48.0),
              child: Text(
                'no_alerts'.tr ?? 'No alerts found.',
                style: const TextStyle(
                  color: AppColors.muted,
                  fontWeight: FontWeight.w700,
                  fontSize: 16,
                ),
              ),
            ),
          );
        }

        return SectionContainer(
          title: 'all_alerts'.tr ?? 'All Alerts',
          subtitle: 'alerts_desc'.tr ?? 'View history of all patient alerts globally.',
          child: ListView.separated(
            shrinkWrap: true,
            physics: const NeverScrollableScrollPhysics(),
            itemCount: data.globalAlerts.length,
            separatorBuilder: (_, __) =>
                const Divider(color: AppColors.border, height: 18),
            itemBuilder: (context, i) {
              final alert = data.globalAlerts[i];
              final patient = data.byId(alert.patientId);
              final pName = patient?.name ?? 'Unknown Patient';

              return LayoutBuilder(
                builder: (context, constraints) {
                  final isMobile = constraints.maxWidth < 600;
                  final timeStr = DateFormat('yyyy/MM/dd HH:mm').format(alert.time);

                  Widget actionWidget() {
                    if (alert.isHandled) {
                      return Container(
                        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                        decoration: BoxDecoration(
                          color: const Color(0xFF4CAF50).withOpacity(0.1),
                          borderRadius: BorderRadius.circular(12),
                          border: Border.all(color: const Color(0xFF4CAF50).withOpacity(0.3)),
                        ),
                        child: Text(
                          'handled'.tr ?? 'Handled',
                          style: const TextStyle(
                            color: Color(0xFF4CAF50),
                            fontWeight: FontWeight.w900,
                            fontSize: 12,
                          ),
                        ),
                      );
                    } else {
                      return OutlineActionButton(
                        text: 'mark_handled'.tr ?? 'Mark Handled',
                        icon: Icons.check,
                        textColor: AppColors.accent,
                        borderColor: AppColors.accent,
                        onPressed: () => data.handleAlert(alert),
                      );
                    }
                  }

                  return Container(
                    padding: const EdgeInsets.all(12),
                    decoration: BoxDecoration(
                      color: Colors.white,
                      borderRadius: BorderRadius.circular(12),
                      border: Border.all(color: AppColors.border),
                    ),
                    child: Row(
                      children: [
                        Container(
                          width: 10,
                          height: 44,
                          decoration: BoxDecoration(
                            color: alert.isHandled
                                ? const Color(0xFF4CAF50).withOpacity(0.6)
                                : AppColors.alert.withOpacity(0.85),
                            borderRadius: BorderRadius.circular(8),
                          ),
                        ),
                        const SizedBox(width: 12),
                        Expanded(
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Text(
                                pName,
                                style: const TextStyle(
                                  fontWeight: FontWeight.w900,
                                  color: AppColors.text,
                                  fontSize: 16,
                                ),
                              ),
                              const SizedBox(height: 4),
                              if (!isMobile)
                                Text(
                                  '${'time'.tr ?? 'Time'}: $timeStr',
                                  style: const TextStyle(
                                    fontSize: 12,
                                    fontWeight: FontWeight.w700,
                                    color: AppColors.muted,
                                  ),
                                ),
                            ],
                          ),
                        ),
                        const SizedBox(width: 12),
                        Column(
                          crossAxisAlignment: CrossAxisAlignment.end,
                          children: [
                            Text(
                              '${'hr'.tr ?? 'HR'}: ${alert.heartRate} bpm',
                              style: TextStyle(
                                fontSize: 13,
                                fontWeight: FontWeight.w900,
                                color: alert.heartRate > 120 ? AppColors.alert : AppColors.text,
                              ),
                            ),
                            Text(
                              '${'hrv'.tr ?? 'HRV'}: ${alert.hrv} ms',
                              style: const TextStyle(
                                fontSize: 12,
                                fontWeight: FontWeight.w700,
                                color: AppColors.muted,
                              ),
                            ),
                          ],
                        ),
                        const SizedBox(width: 24),
                        if (!isMobile) actionWidget(),
                        if (isMobile)
                          Padding(
                            padding: const EdgeInsets.only(left: 12),
                            child: alert.isHandled
                                ? const Icon(Icons.check_circle, color: Color(0xFF4CAF50))
                                : IconButton(
                                    icon: const Icon(Icons.warning, color: AppColors.alert),
                                    onPressed: () => data.handleAlert(alert),
                                  ),
                          ),
                      ],
                    ),
                  );
                },
              );
            },
          ),
        );
      }),
    );
  }
}
