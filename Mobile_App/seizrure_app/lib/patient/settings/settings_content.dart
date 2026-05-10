import 'package:flutter/material.dart';
import 'package:get/get.dart';
import 'package:seizrure_app/shared/controllers/app_controller.dart';
import 'package:seizrure_app/shared/controllers/auth_controller.dart';
import '../../shared/widgets/common_widgets.dart';
import '../../shared/utils/app_colors.dart';
import '../../shared/routes/app_routes.dart';

class SettingsContent extends GetView<AppController> {
  const SettingsContent({super.key});

  @override
  Widget build(BuildContext context) {
    // keep indicator/nav in sync - Logic handled by MainLayout
    WidgetsBinding.instance.addPostFrameCallback((_) {
      //  if (controller.navIndex.value != 2) controller.navIndex.value = 2;
    });

    final mq = MediaQuery.of(context);
    final h = mq.size.height;
    final radius = 18.0;

    return Column(
      children: [
        SizedBox(height: h * 0.015),
        CustomAppBar(
          title: 'settings',
          actions: [
            AppBarButton(
              icon: Icons.notifications_none_rounded,
              onTap: () {},
              badge: true,
            ),
          ],
        ),
        SizedBox(height: h * 0.02),

        // Profile card
        GestureDetector(
          onTap: () => Get.toNamed(Routes.profile),
          child: Container(
            width: double.infinity,
            padding: const EdgeInsets.all(14),
            decoration: BoxDecoration(
              color: Colors.white,
              borderRadius: BorderRadius.circular(radius),
              boxShadow: const [
                BoxShadow(
                  color: AppColors.shadowColor,
                  blurRadius: 18,
                  offset: Offset(0, 10),
                ),
              ],
            ),
            child: Row(
              children: [
                Container(
                  width: 44,
                  height: 44,
                  decoration: BoxDecoration(
                    shape: BoxShape.circle,
                    color: const Color(0xFFEFF3F7),
                    border: Border.all(color: const Color(0xFFE6EDF4)),
                  ),
                  child: const Icon(
                    Icons.person,
                    color: Color(0xFF93A0AE),
                    size: 24,
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: Obx(
                    () => Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          controller.userName.value,
                          style: const TextStyle(
                            fontSize: 14,
                            fontWeight: FontWeight.w800,
                            color: Color(0xFF2A2E35),
                          ),
                        ),
                        const SizedBox(height: 4),
                        Text(
                          controller.userEmail.value,
                          style: const TextStyle(
                            fontSize: 11,
                            fontWeight: FontWeight.w600,
                            color: Color(0xFF9AA7B4),
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
                const Icon(Icons.chevron_right, color: Color(0xFFB0BCC9)),
              ],
            ),
          ),
        ),

        SizedBox(height: h * 0.02),

        // Emergency Contact section (Only visible if caregivers assigned)
        Obx(
          () => controller.assignedCaregivers.isNotEmpty
              ? Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Padding(
                      padding: const EdgeInsets.only(left: 4, bottom: 12),
                      child: Text(
                        'emergency_contact'.tr,
                        style: const TextStyle(
                          fontSize: 16,
                          fontWeight: FontWeight.w800,
                          color: Color(0xFF2A2E35),
                          letterSpacing: -0.5,
                        ),
                      ),
                    ),
                    ...controller.assignedCaregivers
                        .map(
                          (cg) => Container(
                            margin: const EdgeInsets.only(bottom: 12),
                            padding: const EdgeInsets.all(16),
                            decoration: BoxDecoration(
                              color: Colors.white,
                              borderRadius: BorderRadius.circular(radius),
                              boxShadow: const [
                                BoxShadow(
                                  color: AppColors.shadowColor,
                                  blurRadius: 15,
                                  offset: Offset(0, 5),
                                ),
                              ],
                            ),
                            child: Row(
                              children: [
                                Container(
                                  padding: const EdgeInsets.all(8),
                                  decoration: BoxDecoration(
                                    color: const Color(0xFFF1F5F9),
                                    shape: BoxShape.circle,
                                  ),
                                  child: const Icon(
                                    Icons.person_outline,
                                    size: 20,
                                    color: Color(0xFF64748B),
                                  ),
                                ),
                                const SizedBox(width: 12),
                                Expanded(
                                  child: Column(
                                    crossAxisAlignment:
                                        CrossAxisAlignment.start,
                                    children: [
                                      Text(
                                        cg.name,
                                        style: const TextStyle(
                                          fontSize: 13,
                                          fontWeight: FontWeight.w800,
                                          color: Color(0xFF2A2E35),
                                        ),
                                      ),
                                      const SizedBox(height: 4),
                                      Text(
                                        cg.phone,
                                        style: const TextStyle(
                                          fontSize: 11,
                                          fontWeight: FontWeight.w600,
                                          color: Color(0xFF9AA7B4),
                                        ),
                                      ),
                                    ],
                                  ),
                                ),
                                const SizedBox(width: 10),
                                MiniIconButton(
                                  icon: Icons.message_outlined,
                                  onTap: () => controller.sendSms(cg.phone),
                                ),
                                const SizedBox(width: 8),
                                MiniIconButton(
                                  icon: Icons.call_outlined,
                                  onTap: () =>
                                      controller.makePhoneCall(cg.phone),
                                ),
                              ],
                            ),
                          ),
                        )
                        .toList(),
                  ],
                )
              : const SizedBox.shrink(),
        ),

        SizedBox(height: h * 0.02),

        // Logout Button
        SecondaryButton(
          text: 'logout'.tr,
          color: AppColors.danger,
          icon: Icons.logout_rounded,
          onTap: () => Get.find<AuthController>().logout(),
        ),
      ],
    );
  }
}
