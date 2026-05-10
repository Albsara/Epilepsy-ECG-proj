import 'package:flutter/material.dart';
import 'package:get/get.dart';
import 'package:seizrure_app/shared/controllers/app_controller.dart';
import '../shared/utils/app_colors.dart';
import '../shared/widgets/common_widgets.dart';
import '../shared/widgets/seizure_alert_tile.dart';

class PatientDetailsView extends GetView<AppController> {
  const PatientDetailsView({super.key});

  @override
  Widget build(BuildContext context) {
    final w = MediaQuery.of(context).size.width;

    return Scaffold(
      backgroundColor: AppColors.background,
      body: SafeArea(
        child: RefreshIndicator(
          onRefresh: controller.refreshData,
          color: AppColors.primaryGradientEnd,
          child: SingleChildScrollView(
            physics: const AlwaysScrollableScrollPhysics(),
            padding: EdgeInsets.symmetric(horizontal: w * 0.05),
            child: Column(
              children: [
                const SizedBox(height: 10),
                Obx(
                  () => CustomAppBar(
                    title:
                        controller.patientNames[controller
                            .currentViewedPatientId
                            .value] ??
                        controller.historyPatientName.value,
                    showBackButton: true,
                  ),
                ),
                const SizedBox(height: 16),

                // Live Vitals Card for the specific patient
                Obx(
                  () => Container(
                    width: double.infinity,
                    padding: const EdgeInsets.all(20),
                    decoration: BoxDecoration(
                      gradient: const LinearGradient(
                        colors: [
                          AppColors.primaryGradientStart,
                          AppColors.primaryGradientEnd,
                        ],
                        begin: Alignment.topLeft,
                        end: Alignment.bottomRight,
                      ),
                      borderRadius: BorderRadius.circular(24),
                      boxShadow: [
                        BoxShadow(
                          color: AppColors.primaryGradientEnd.withOpacity(0.2),
                          blurRadius: 15,
                          offset: const Offset(0, 8),
                        ),
                      ],
                    ),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Row(
                          children: [
                            Container(
                              padding: const EdgeInsets.all(6),
                              decoration: BoxDecoration(
                                color: Colors.white.withOpacity(0.2),
                                shape: BoxShape.circle,
                              ),
                              child: const Icon(
                                Icons.sensors_rounded,
                                color: Colors.white,
                                size: 16,
                              ),
                            ),
                            const SizedBox(width: 10),
                            const Text(
                              'Live Status',
                              style: TextStyle(
                                color: Colors.white,
                                fontSize: 16,
                                fontWeight: FontWeight.w800,
                              ),
                            ),
                            const Spacer(),
                            Container(
                              padding: const EdgeInsets.symmetric(
                                horizontal: 8,
                                vertical: 4,
                              ),
                              decoration: BoxDecoration(
                                color: Colors.white.withOpacity(0.25),
                                borderRadius: BorderRadius.circular(20),
                              ),
                              child: const Text(
                                'LIVE',
                                style: TextStyle(
                                  color: Colors.white,
                                  fontSize: 9,
                                  fontWeight: FontWeight.w900,
                                ),
                              ),
                            ),
                          ],
                        ),
                        const SizedBox(height: 20),
                        Row(
                          children: [
                            Expanded(
                              child: _buildLiveMetric(
                                Icons.favorite_rounded,
                                controller.bpm.value.toString(),
                                'BPM',
                                Colors.white,
                              ),
                            ),
                            const SizedBox(width: 20),
                            Expanded(
                              child: _buildLiveMetric(
                                Icons.speed_rounded,
                                "${controller.hrv.value} ms",
                                'HRV',
                                Colors.white,
                              ),
                            ),
                          ],
                        ),
                        const SizedBox(height: 20),
                        // Context Row 1
                        Row(
                          children: [
                            Expanded(
                              child: _buildLiveMetric(
                                Icons.medication_rounded,
                                controller.medication.value.isEmpty
                                    ? '-'
                                    : controller.medication.value
                                          .toLowerCase()
                                          .tr,
                                'medication'.tr,
                                Colors.white,
                              ),
                            ),
                            const SizedBox(width: 20),
                            Expanded(
                              child: _buildLiveMetric(
                                Icons.psychology_rounded,
                                controller.stress.value.isEmpty
                                    ? '-'
                                    : controller.stress.value.toLowerCase().tr,
                                'stress'.tr,
                                Colors.white,
                              ),
                            ),
                          ],
                        ),
                        const SizedBox(height: 20),
                        // Context Row 2
                        Row(
                          children: [
                            Expanded(
                              child: _buildLiveMetric(
                                Icons.nights_stay_rounded,
                                controller.sleep.value.isEmpty
                                    ? '-'
                                    : controller.sleep.value.toLowerCase().tr,
                                'sleep'.tr,
                                Colors.white,
                              ),
                            ),
                            const SizedBox(width: 20),
                            Expanded(
                              child: _buildLiveMetric(
                                Icons.assignment_ind_rounded,
                                controller.symptoms.value.isEmpty
                                    ? '-'
                                    : controller.symptoms.value
                                          .toLowerCase()
                                          .tr,
                                'symptoms'.tr,
                                Colors.white,
                              ),
                            ),
                          ],
                        ),
                      ],
                    ),
                  ),
                ),

                const SizedBox(height: 24),
                Align(
                  alignment: Alignment.centerLeft,
                  child: Text(
                    'seizure_history'.tr,
                    style: const TextStyle(
                      fontSize: 18,
                      fontWeight: FontWeight.w800,
                      color: AppColors.textDark,
                    ),
                  ),
                ),
                const SizedBox(height: 8),
                Obx(() {
                  if (controller.historyList.isEmpty) {
                    return Center(
                      child: Padding(
                        padding: const EdgeInsets.symmetric(vertical: 40),
                        child: Text(
                          'no_alerts'.tr,
                          style: const TextStyle(
                            color: AppColors.textGrey,
                            fontSize: 14,
                          ),
                        ),
                      ),
                    );
                  }

                  return ListView.builder(
                    padding: const EdgeInsets.symmetric(vertical: 8),
                    clipBehavior: Clip.none,
                    shrinkWrap: true,
                    physics: const NeverScrollableScrollPhysics(),
                    itemCount: controller.historyList.length,
                    itemBuilder: (ctx, i) {
                      final item = controller.historyList[i];
                      return SeizureAlertTile(
                        item: item,
                        isCaregiver: true,
                        onHandled: () {
                          controller.toggleAlertHandled(
                            item['patientId']!,
                            item['id']!,
                            item['is_handled'] == 'true',
                          );
                        },
                      );
                    },
                  );
                }),
                const SizedBox(height: 30),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildLiveMetric(
    IconData icon,
    String value,
    String unit,
    Color color,
  ) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            Icon(icon, size: 14, color: color),
            const SizedBox(width: 8),
            Text(
              unit,
              style: TextStyle(
                color: Colors.white.withOpacity(0.5),
                fontSize: 11,
                fontWeight: FontWeight.w700,
              ),
            ),
          ],
        ),
        const SizedBox(height: 4),
        Text(
          value,
          style: const TextStyle(
            color: Colors.white,
            fontSize: 18,
            fontWeight: FontWeight.w900,
          ),
        ),
      ],
    );
  }
}
