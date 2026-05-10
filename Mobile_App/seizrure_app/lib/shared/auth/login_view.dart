import 'package:flutter/material.dart';
import 'package:get/get.dart';
import 'package:seizrure_app/shared/controllers/auth_controller.dart';
import '../utils/app_colors.dart';
import '../routes/app_routes.dart';
import '../widgets/common_widgets.dart';

class LoginView extends StatefulWidget {
  const LoginView({super.key});

  @override
  State<LoginView> createState() => _LoginViewState();
}

class _LoginViewState extends State<LoginView> {
  final AuthController authController = Get.find<AuthController>();
  final emailController = TextEditingController();
  final passwordController = TextEditingController();

  @override
  Widget build(BuildContext context) {
    final mq = MediaQuery.of(context);
    final h = mq.size.height;
    final w = mq.size.width;

    return Scaffold(
      backgroundColor: AppColors.background,
      body: SafeArea(
        child: SingleChildScrollView(
          padding: EdgeInsets.symmetric(horizontal: w * 0.08),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const SizedBox(height: 10),
              const CustomAppBar(title: '', showLangToggle: true),
              SizedBox(height: h * 0.08),
              // Logo or App Icon
              Center(
                child: Container(
                  width: 80,
                  height: 80,
                  decoration: BoxDecoration(
                    gradient: const LinearGradient(
                      colors: [
                        AppColors.primaryGradientStart,
                        AppColors.primaryGradientEnd,
                      ],
                    ),
                    borderRadius: BorderRadius.circular(24),
                    boxShadow: [
                      BoxShadow(
                        color: AppColors.primaryGradientEnd.withOpacity(0.3),
                        blurRadius: 20,
                        offset: const Offset(0, 10),
                      ),
                    ],
                  ),
                  child: const Icon(
                    Icons.bar_chart_rounded,
                    color: Colors.white,
                    size: 40,
                  ),
                ),
              ),
              SizedBox(height: h * 0.05),
              Text(
                'welcome_back'.tr,
                style: const TextStyle(
                  fontSize: 28,
                  fontWeight: FontWeight.w900,
                  color: AppColors.textDark,
                  letterSpacing: -0.5,
                ),
              ),
              const SizedBox(height: 8),
              Text(
                'sign_in_to_continue'.tr,
                style: const TextStyle(
                  fontSize: 15,
                  fontWeight: FontWeight.w500,
                  color: AppColors.textGrey,
                ),
              ),
              SizedBox(height: h * 0.06),

              // Email Field
              CustomTextField(
                label: 'email_address'.tr,
                icon: Icons.email_outlined,
                controller: emailController,
              ),
              const SizedBox(height: 20),

              // Password Field
              CustomTextField(
                label: 'password'.tr,
                icon: Icons.lock_outline_rounded,
                isPassword: true,
                controller: passwordController,
              ),

              // Forgot Password
              Align(
                alignment: Alignment.centerRight,
                child: TextButton(
                  onPressed: () => Get.toNamed(Routes.forgotPassword),
                  child: Text(
                    'forgot_password_q'.tr,
                    style: const TextStyle(
                      color: AppColors.primaryGradientEnd,
                      fontWeight: FontWeight.w700,
                      fontSize: 13,
                    ),
                  ),
                ),
              ),

              SizedBox(height: h * 0.04),

              // Login Button
              PrimaryButton(
                text: 'sign_in'.tr,
                onTap: () => authController.login(
                  emailController.text.trim(),
                  passwordController.text,
                ),
              ),
              const SizedBox(height: 40),
            ],
          ),
        ),
      ),
    );
  }
}
