import 'package:flutter/material.dart';
import 'package:flutter/foundation.dart';
import 'package:get/get.dart';
import 'package:seizrure_app/shared/controllers/app_controller.dart';
import '../shared/utils/app_colors.dart';
import '../shared/widgets/common_widgets.dart';
import '../shared/routes/app_routes.dart';
import '../shared/widgets/seizure_alert_tile.dart';

class CaregiverHomeView extends GetView<AppController> {
  const CaregiverHomeView({super.key});

  @override
  Widget build(BuildContext context) {
    final w = MediaQuery.of(context).size.width;

    return Scaffold(
      backgroundColor: AppColors.background,
      body: SafeArea(
        child: Padding(
          padding: EdgeInsets.symmetric(horizontal: w * 0.065),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const SizedBox(height: 10),
              CustomAppBar(
                title: 'caregiver_home'.tr,
                showLangToggle: true,
                showRefresh: kIsWeb,
                onRefresh: controller.refreshData,
                showBackButton: false,
                actions: [
                  AppBarButton(
                    icon: Icons.person_outline_rounded,
                    onTap: () => Get.toNamed(Routes.caregiverProfile),
                  ),
                ],
              ),
              const SizedBox(height: 20),

              Expanded(
                child: RefreshIndicator(
                  onRefresh: controller.refreshData,
                  color: AppColors.primaryGradientEnd,
                  child: SingleChildScrollView(
                    physics: const AlwaysScrollableScrollPhysics(),
                    clipBehavior: Clip.none,
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        // Assigned Patients Section
                        Obx(
                          () => controller.assignedPatients.isNotEmpty
                              ? Column(
                                  crossAxisAlignment: CrossAxisAlignment.start,
                                  children: [
                                    Text(
                                      'patient_status'.tr,
                                      style: const TextStyle(
                                        fontSize: 18,
                                        fontWeight: FontWeight.w800,
                                        color: AppColors.textDark,
                                      ),
                                    ),
                                    const SizedBox(height: 12),
                                    ...controller.assignedPatients
                                        .map(
                                          (pData) => Padding(
                                            padding: const EdgeInsets.only(
                                              bottom: 16,
                                            ),
                                            child: _buildPatientStatusCard(
                                              context,
                                              pData,
                                            ),
                                          ),
                                        )
                                        .toList(),
                                  ],
                                )
                              : const SizedBox.shrink(),
                        ),

                        const SizedBox(height: 8),

                        // Latest Alerts Header
                        Row(
                          mainAxisAlignment: MainAxisAlignment.spaceBetween,
                          children: [
                            Text(
                              'latest_alerts'.tr,
                              style: const TextStyle(
                                fontSize: 18,
                                fontWeight: FontWeight.w800,
                                color: AppColors.textDark,
                              ),
                            ),
                            TextButton(
                              onPressed: () {
                                // Get current caregiver and their patient IDs
                                // Actually, we can just fetch all assigned alerts for all patients in current list
                                final pIds = controller.assignedPatients
                                    .map((e) => e.patient.authUid)
                                    .toList();
                                controller.viewAllAlerts(pIds);
                              },
                              child: Text(
                                'view_all'.tr,
                                style: const TextStyle(
                                  color: AppColors.primaryGradientEnd,
                                  fontWeight: FontWeight.w700,
                                ),
                              ),
                            ),
                          ],
                        ),

                        const SizedBox(height: 8),

                        // Alerts List (Latest 3)
                        Obx(() {
                          final alerts = controller.historyList
                              .take(3)
                              .toList();
                          if (alerts.isEmpty) {
                            return Center(
                              child: Padding(
                                padding: const EdgeInsets.symmetric(
                                  vertical: 40,
                                ),
                                child: Text(
                                  'no_alerts'.tr,
                                  style: const TextStyle(
                                    color: AppColors.textGrey,
                                    fontSize: 16,
                                  ),
                                ),
                              ),
                            );
                          }

                          return Column(
                            children: alerts
                                .map(
                                  (item) => SeizureAlertTile(
                                    item: item,
                                    isCaregiver: true,
                                    onHandled: () {
                                      controller.toggleAlertHandled(
                                        item['patientId']!,
                                        item['id']!,
                                        item['is_handled'] == 'true',
                                      );
                                    },
                                  ),
                                )
                                .toList(),
                          );
                        }),
                        const SizedBox(height: 20),
                      ],
                    ),
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildPatientStatusCard(
    BuildContext context,
    AssignedPatientData pData,
  ) {
    return GestureDetector(
      onTap: () => controller.loadPatientHistory(pData.patient.authUid),
      child: Obx(
        () => Container(
          width: double.infinity,
          padding: const EdgeInsets.all(20),
          decoration: BoxDecoration(
            gradient: LinearGradient(
              colors: pData.isDanger.value
                  ? [AppColors.dangerGradientStart, AppColors.dangerGradientEnd]
                  : [
                      AppColors.primaryGradientStart,
                      AppColors.primaryGradientEnd,
                    ],
              begin: Alignment.topLeft,
              end: Alignment.bottomRight,
            ),
            borderRadius: BorderRadius.circular(24),
            boxShadow: [
              BoxShadow(
                color:
                    (pData.isDanger.value
                            ? AppColors.danger
                            : AppColors.primaryGradientEnd)
                        .withOpacity(0.25),
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
                      color: Colors.white.withOpacity(0.15),
                      shape: BoxShape.circle,
                    ),
                    child: const Icon(
                      Icons.person_rounded,
                      color: Colors.white,
                      size: 18,
                    ),
                  ),
                  const SizedBox(width: 10),
                  Expanded(
                    child: Text(
                      pData.patient.name,
                      style: const TextStyle(
                        color: Colors.white,
                        fontSize: 16,
                        fontWeight: FontWeight.w800,
                      ),
                    ),
                  ),
                  Container(
                    padding: const EdgeInsets.symmetric(
                      horizontal: 10,
                      vertical: 4,
                    ),
                    decoration: BoxDecoration(
                      color: Colors.white.withOpacity(0.2),
                      borderRadius: BorderRadius.circular(20),
                    ),
                    child: Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Container(
                          width: 6,
                          height: 6,
                          decoration: const BoxDecoration(
                            color: Color(0xFF4ADE80),
                            shape: BoxShape.circle,
                          ),
                        ),
                        const SizedBox(width: 6),
                        const Text(
                          'Live',
                          style: TextStyle(
                            color: Colors.white,
                            fontSize: 10,
                            fontWeight: FontWeight.w700,
                          ),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 20),
              Row(
                children: [
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Obx(
                          () => Text(
                            pData.hr.value.toString(),
                            style: const TextStyle(
                              color: Colors.white,
                              fontSize: 24,
                              fontWeight: FontWeight.w900,
                            ),
                          ),
                        ),
                        Text(
                          'heart_rate'.tr,
                          style: TextStyle(
                            color: Colors.white.withOpacity(0.7),
                            fontSize: 11,
                            fontWeight: FontWeight.w600,
                          ),
                        ),
                      ],
                    ),
                  ),
                  Container(
                    width: 1,
                    height: 30,
                    color: Colors.white.withOpacity(0.15),
                  ),
                  const SizedBox(width: 20),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Obx(
                          () => Text(
                            pData.hrv.value.toString(),
                            style: const TextStyle(
                              color: Colors.white,
                              fontSize: 24,
                              fontWeight: FontWeight.w900,
                            ),
                          ),
                        ),
                        Text(
                          'heart_rate_variability'.tr,
                          style: TextStyle(
                            color: Colors.white.withOpacity(0.7),
                            fontSize: 11,
                            fontWeight: FontWeight.w600,
                          ),
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(width: 10),
                  const Icon(Icons.chevron_right_rounded, color: Colors.white),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }
}
