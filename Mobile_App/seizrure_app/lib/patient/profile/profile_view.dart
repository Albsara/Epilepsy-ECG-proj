import 'package:flutter/material.dart';
import 'package:get/get.dart';
import 'package:seizrure_app/shared/controllers/app_controller.dart';
import '../../shared/widgets/common_widgets.dart';

class ProfileView extends GetView<AppController> {
  const ProfileView({super.key});

  @override
  Widget build(BuildContext context) {
    final mq = MediaQuery.of(context);
    final w = mq.size.width;
    final h = mq.size.height;

    final horizontalPadding = w * 0.065;

    // Form controllers (initialized with current data)
    final nameCtrl = TextEditingController(text: controller.userName.value);
    final phoneCtrl = TextEditingController(text: controller.userPhone.value);
    final birthDateCtrl = TextEditingController(
      text: controller.userBirthDate.value,
    );

    return Scaffold(
      backgroundColor: const Color(0xFFF2F5F8),
      body: SafeArea(
        child: Padding(
          padding: EdgeInsets.symmetric(horizontal: horizontalPadding),
          child: Column(
            children: [
              SizedBox(height: h * 0.015),
              const CustomAppBar(title: 'manage_profile', showBackButton: true),
              SizedBox(height: h * 0.03),

              Expanded(
                child: SingleChildScrollView(
                  child: Column(
                    children: [
                      // Avatar
                      Center(
                        child: Stack(
                          children: [
                            Container(
                              width: 100,
                              height: 100,
                              decoration: BoxDecoration(
                                shape: BoxShape.circle,
                                color: const Color(0xFFEFF3F7),
                                border: Border.all(
                                  color: const Color(0xFFE6EDF4),
                                  width: 3,
                                ),
                              ),
                              child: const Icon(
                                Icons.person,
                                color: Color(0xFF93A0AE),
                                size: 50,
                              ),
                            ),
                            Positioned(
                              bottom: 0,
                              right: 0,
                              child: Container(
                                padding: const EdgeInsets.all(6),
                                decoration: const BoxDecoration(
                                  color: Color(0xFF5BA7D9),
                                  shape: BoxShape.circle,
                                ),
                                child: const Icon(
                                  Icons.camera_alt,
                                  color: Colors.white,
                                  size: 16,
                                ),
                              ),
                            ),
                          ],
                        ),
                      ),
                      SizedBox(height: h * 0.04),

                      CustomTextField(
                        label: 'name'.tr,
                        controller: nameCtrl,
                        icon: Icons.person_outline,
                      ),
                      const SizedBox(height: 16),
                      CustomTextField(
                        label: 'phone_number'.tr,
                        controller: phoneCtrl,
                        icon: Icons.phone_outlined,
                      ),
                      const SizedBox(height: 16),
                      Row(
                        children: [
                          Expanded(
                            child: CustomTextField(
                              label: 'birth_date'.tr,
                              controller: birthDateCtrl,
                              icon: Icons.calendar_today,
                            ),
                          ),
                          const SizedBox(width: 16),
                          Expanded(
                            child: Obx(() => CustomDropdownField(
                              label: 'gender'.tr,
                              value: controller.userGender.value,
                              items: const ['Male', 'Female'],
                              itemLabels: const ['male', 'female'],
                              icon: Icons.people_outline,
                              onChanged: (val) {
                                if (val != null) {
                                  controller.userGender.value = val;
                                }
                              },
                            )),
                          ),
                        ],
                      ),

                      SizedBox(height: h * 0.05),

                      GradientButton(
                        text: 'save_changes'.tr,
                        height: 50,
                        radius: 14,
                        colors: const [Color(0xFF76CDE6), Color(0xFF5BA7D9)],
                        onTap: () async {
                          await controller.saveProfile(
                            name: nameCtrl.text,
                            phone: phoneCtrl.text,
                            birthDate: birthDateCtrl.text,
                            gender: controller.userGender.value,
                          );
                          Get.back();
                        },
                      ),
                    ],
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
