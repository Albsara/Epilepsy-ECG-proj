import 'package:flutter/material.dart';
import 'package:get/get.dart';
import '../utils/app_colors.dart';
import '../widgets/common_widgets.dart';

class ForgotPasswordView extends StatelessWidget {
  const ForgotPasswordView({super.key});

  @override
  Widget build(BuildContext context) {
    final mq = MediaQuery.of(context);
    final h = mq.size.height;
    final w = mq.size.width;

    return Scaffold(
      backgroundColor: AppColors.background,
      body: SafeArea(
        child: SingleChildScrollView(
          child: Padding(
            padding: EdgeInsets.symmetric(horizontal: w * 0.08),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const SizedBox(height: 10),
                const CustomAppBar(
                  title: 'reset_password',
                  showBackButton: true,
                ),
                SizedBox(height: h * 0.06),
                // Icon
                Container(
                  width: 64,
                  height: 64,
                  decoration: BoxDecoration(
                    color: AppColors.primaryGradientEnd.withOpacity(0.1),
                    borderRadius: BorderRadius.circular(20),
                  ),
                  child: const Icon(
                    Icons.lock_reset_rounded,
                    color: AppColors.primaryGradientEnd,
                    size: 32,
                  ),
                ),
                const SizedBox(height: 32),
                Text(
                  'forgot_password_q'.tr,
                  style: const TextStyle(
                    fontSize: 28,
                    fontWeight: FontWeight.w900,
                    color: AppColors.textDark,
                    letterSpacing: -0.5,
                  ),
                ),
                const SizedBox(height: 12),
                Text(
                  'forgot_password_info'.tr,
                  style: const TextStyle(
                    fontSize: 15,
                    fontWeight: FontWeight.w500,
                    color: AppColors.textGrey,
                    height: 1.5,
                  ),
                ),
                SizedBox(height: h * 0.06),

                // Email Field
                CustomTextField(
                  label: 'email_address'.tr,
                  hint: 'name@example.com',
                  icon: Icons.email_outlined,
                ),

                SizedBox(height: h * 0.06),

                // Reset Button
                PrimaryButton(
                  text: 'send_reset_link'.tr,
                  onTap: () {
                    Get.defaultDialog(
                      title: 'success'.tr,
                      middleText: 'reset_link_sent'.tr,
                      textConfirm: 'OK',
                      confirmTextColor: Colors.white,
                      onConfirm: () {
                        Get.back(); // close dialog
                        Get.back(); // go back to login
                      },
                    );
                  },
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
