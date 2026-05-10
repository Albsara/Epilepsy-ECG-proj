import 'package:flutter/material.dart';
import 'package:get/get.dart';
import 'package:seizrure_app/shared/controllers/app_controller.dart';
import 'package:seizrure_app/shared/controllers/auth_controller.dart';
import '../shared/utils/app_colors.dart';
import '../shared/widgets/common_widgets.dart';
import '../shared/routes/app_routes.dart';

class CaregiverProfileView extends GetView<AppController> {
  const CaregiverProfileView({super.key});

  @override
  Widget build(BuildContext context) {
    final w = MediaQuery.of(context).size.width;

    final nameController = TextEditingController(
      text: controller.userName.value,
    );
    final phoneController = TextEditingController(
      text: controller.userPhone.value,
    );

    return Scaffold(
      backgroundColor: AppColors.background,
      body: SafeArea(
        child: SingleChildScrollView(
          padding: EdgeInsets.symmetric(horizontal: w * 0.065),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const SizedBox(height: 10),
              CustomAppBar(
                title: 'manage_account'.tr,
                showBackButton: true,
                showLangToggle: true,
              ),
              const SizedBox(height: 30),

              // Profile Header/Icon
              Center(
                child: Column(
                  children: [
                    Container(
                      width: 100,
                      height: 100,
                      decoration: BoxDecoration(
                        color: Colors.white,
                        shape: BoxShape.circle,
                        boxShadow: [
                          BoxShadow(
                            color: Colors.black.withOpacity(0.05),
                            blurRadius: 20,
                            offset: const Offset(0, 10),
                          ),
                        ],
                      ),
                      child: const Icon(
                        Icons.person_outline_rounded,
                        size: 50,
                        color: AppColors.primaryGradientEnd,
                      ),
                    ),
                    const SizedBox(height: 16),
                    Obx(
                      () => Text(
                        controller.userName.value,
                        style: const TextStyle(
                          fontSize: 22,
                          fontWeight: FontWeight.w800,
                          color: AppColors.textDark,
                        ),
                      ),
                    ),
                    Text(
                      'caregiver'.tr,
                      style: const TextStyle(
                        fontSize: 14,
                        color: AppColors.textGrey,
                        fontWeight: FontWeight.w500,
                      ),
                    ),
                  ],
                ),
              ),

              const SizedBox(height: 40),

              // Form Fields
              CustomTextField(
                label: 'name'.tr,
                controller: nameController,
                icon: Icons.person_outline_rounded,
              ),
              const SizedBox(height: 20),
              CustomTextField(
                label: 'phone_number'.tr,
                controller: phoneController,
                icon: Icons.phone_android_rounded,
                isPhone: true,
              ),

              const SizedBox(height: 40),

              // Save Button
              PrimaryButton(
                text: 'save_changes'.tr,
                onTap: () async {
                  await controller.saveProfile(
                    name: nameController.text,
                    phone: phoneController.text,
                  );
                },
              ),

              const SizedBox(height: 30),
              // Manage Patients Section
              Text(
                'manage_patients'.tr,
                style: const TextStyle(
                  fontSize: 18,
                  fontWeight: FontWeight.w800,
                  color: AppColors.textDark,
                ),
              ),
              const SizedBox(height: 12),
              Obx(
                () => Column(
                  children: [
                    ...controller.assignedPatients.map(
                      (pData) => Container(
                        margin: const EdgeInsets.only(bottom: 12),
                        padding: const EdgeInsets.symmetric(
                          horizontal: 16,
                          vertical: 12,
                        ),
                        decoration: BoxDecoration(
                          color: Colors.white,
                          borderRadius: BorderRadius.circular(16),
                          boxShadow: [
                            BoxShadow(
                              color: Colors.black.withOpacity(0.03),
                              blurRadius: 10,
                              offset: const Offset(0, 4),
                            ),
                          ],
                        ),
                        child: Row(
                          children: [
                            const CircleAvatar(
                              backgroundColor: AppColors.background,
                              child: Icon(
                                Icons.person,
                                color: AppColors.textGrey,
                                size: 20,
                              ),
                            ),
                            const SizedBox(width: 12),
                            Expanded(
                              child: Text(
                                pData.patient.name,
                                style: const TextStyle(
                                  fontWeight: FontWeight.w700,
                                ),
                              ),
                            ),
                            IconButton(
                              icon: const Icon(
                                Icons.link_off_rounded,
                                color: AppColors.danger,
                              ),
                              onPressed: () {
                                Get.defaultDialog(
                                  title: 'confirm_unassign'.tr,
                                  middleText: 'unassign_warning'.tr,
                                  textConfirm: 'yes'.tr,
                                  textCancel: 'no'.tr,
                                  confirmTextColor: Colors.white,
                                  onConfirm: () {
                                    Get.back();
                                    controller.unlinkPatient(
                                      pData.patient.authUid,
                                    );
                                  },
                                );
                              },
                            ),
                          ],
                        ),
                      ),
                    ),
                  ],
                ),
              ),

              const SizedBox(height: 16),

              // Logout Button
              SecondaryButton(
                text: 'logout'.tr,
                color: AppColors.danger,
                icon: Icons.logout_rounded,
                onTap: () => Get.find<AuthController>().logout(),
              ),
              const SizedBox(height: 40),
            ],
          ),
        ),
      ),
    );
  }

  void _showAddPatientDialog(BuildContext context) {
    final idController = TextEditingController();
    Get.defaultDialog(
      title: 'add_patient'.tr,
      content: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 16),
        child: Column(
          children: [
            Text(
              'enter_patient_id_hint'.tr,
              textAlign: TextAlign.center,
              style: const TextStyle(fontSize: 12, color: AppColors.textGrey),
            ),
            const SizedBox(height: 16),
            TextField(
              controller: idController,
              decoration: InputDecoration(
                hintText: 'Patient ID',
                prefixIcon: const Icon(Icons.qr_code_rounded),
                border: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(12),
                ),
              ),
            ),
          ],
        ),
      ),
      textConfirm: 'add'.tr,
      textCancel: 'cancel'.tr,
      confirmTextColor: Colors.white,
      onConfirm: () {
        if (idController.text.isNotEmpty) {
          Get.back();
          controller.linkPatient(idController.text.trim());
        }
      },
    );
  }
}
