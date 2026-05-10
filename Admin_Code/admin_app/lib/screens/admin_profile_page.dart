import 'package:admin_app/utils/app_colors.dart';
import 'package:flutter/material.dart';
import 'package:get/get.dart';
import '../controllers/auth_controller.dart';
import '../widgets/admin_shell.dart';
import '../widgets/common_widgets.dart';

class AdminProfilePage extends StatefulWidget {
  const AdminProfilePage({super.key});

  @override
  State<AdminProfilePage> createState() => _AdminProfilePageState();
}

class _AdminProfilePageState extends State<AdminProfilePage> {
  final auth = Get.find<AuthController>();
  final nameC = TextEditingController();
  final emailC = TextEditingController();
  final phoneC = TextEditingController();

  @override
  void initState() {
    super.initState();
    nameC.text = auth.adminName.value;
    emailC.text = auth.adminEmail.value;
    phoneC.text = auth.adminPhone.value;
  }

  @override
  Widget build(BuildContext context) {
    return AdminShell(
      title: 'personal_info'.tr,
      child: SectionContainer(
        title: 'edit_personal_info'.tr,
        subtitle: 'update_profile_desc'.tr,
        trailing: OutlineActionButton(
          text: 'back'.tr,
          icon: Icons.arrow_back,
          onPressed: () => Get.back(),
        ),
        child: LayoutBuilder(
          builder: (context, constraints) {
            final wide = constraints.maxWidth >= 900;

            final fields = [
              TextField(
                controller: nameC,
                decoration: InputDecoration(labelText: 'name'.tr),
              ),
              TextField(
                controller: emailC,
                keyboardType: TextInputType.emailAddress,
                decoration: InputDecoration(labelText: 'email'.tr),
                enabled: false, // email comes from FirebaseAuth session
              ),
              TextField(
                controller: phoneC,
                keyboardType: TextInputType.phone,
                decoration: InputDecoration(labelText: 'phone'.tr),
              ),
            ];

            if (wide) {
              return Column(
                children: [
                  Row(
                    children: [
                      Expanded(child: fields[0]),
                      const SizedBox(width: 12),
                      Expanded(child: fields[1]),
                    ],
                  ),
                  const SizedBox(height: 12),
                  Row(
                    children: [
                      Expanded(child: fields[2]),
                      const SizedBox(width: 12),
                      const Expanded(child: SizedBox()),
                    ],
                  ),
                  const SizedBox(height: 14),
                  Row(
                    children: [
                      Expanded(
                        child: OutlineActionButton(
                          text: 'cancel'.tr,
                          onPressed: () => Get.back(),
                        ),
                      ),
                      const SizedBox(width: 10),
                      Expanded(
                        child: PrimaryButton(
                          text: 'save'.tr,
                          icon: Icons.save,
                          onPressed: () => auth.updateProfile(
                            name: nameC.text,
                            email: emailC.text,
                            phone: phoneC.text,
                          ),
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 12),
                  const Divider(color: AppColors.border),
                  const SizedBox(height: 12),
                  Row(
                    children: [
                      Expanded(
                        child: OutlineActionButton(
                          text: 'change_password'.tr,
                          icon: Icons.lock_outline,
                          onPressed: _showChangePasswordDialog,
                        ),
                      ),
                      const SizedBox(width: 12),
                      const Expanded(child: SizedBox()),
                    ],
                  ),
                ],
              );
            }

            return Column(
              children: [
                for (int i = 0; i < fields.length; i++) ...[
                  fields[i],
                  if (i != fields.length - 1) const SizedBox(height: 12),
                ],
                const SizedBox(height: 14),
                PrimaryButton(
                  text: 'save'.tr,
                  icon: Icons.save,
                  fullWidth: true,
                  onPressed: () => auth.updateProfile(
                    name: nameC.text,
                    email: emailC.text,
                    phone: phoneC.text,
                  ),
                ),
                const SizedBox(height: 12),
                const Divider(color: AppColors.border),
                const SizedBox(height: 12),
                OutlineActionButton(
                  text: 'change_password'.tr,
                  icon: Icons.lock_outline,
                  fullWidth: true,
                  onPressed: _showChangePasswordDialog,
                ),
              ],
            );
          },
        ),
      ),
    );
  }

  void _showChangePasswordDialog() {
    final passC = TextEditingController();
    final confirmPassC = TextEditingController();

    Get.dialog(
      Center(
        child: ConstrainedBox(
          constraints: const BoxConstraints(maxWidth: 400),
          child: Material(
            color: Colors.transparent,
            child: Container(
              padding: const EdgeInsets.all(24),
              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(16),
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
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    'change_password'.tr,
                    style: const TextStyle(
                      fontSize: 18,
                      fontWeight: FontWeight.w900,
                      color: AppColors.text,
                    ),
                  ),
                  const SizedBox(height: 8),
                  Text(
                    'change_password_desc'.tr,
                    style: const TextStyle(
                      fontSize: 13,
                      fontWeight: FontWeight.w700,
                      color: AppColors.muted,
                    ),
                  ),
                  const SizedBox(height: 20),
                  TextField(
                    controller: passC,
                    obscureText: true,
                    decoration: InputDecoration(
                      labelText: 'new_password'.tr,
                      prefixIcon: const Icon(Icons.lock_outline),
                    ),
                  ),
                  const SizedBox(height: 12),
                  TextField(
                    controller: confirmPassC,
                    obscureText: true,
                    decoration: InputDecoration(
                      labelText: 'confirm_password'.tr,
                      prefixIcon: const Icon(Icons.lock_reset),
                    ),
                  ),
                  const SizedBox(height: 24),
                  Row(
                    children: [
                      Expanded(
                        child: OutlineActionButton(
                          text: 'cancel'.tr,
                          onPressed: () => Get.back(),
                        ),
                      ),
                      const SizedBox(width: 12),
                      Expanded(
                        child: PrimaryButton(
                          text: 'update'.tr,
                          onPressed: () {
                            if (passC.text != confirmPassC.text) {
                              Get.snackbar(
                                'error'.tr,
                                'passwords_do_not_match'.tr,
                                backgroundColor: Colors.red.withOpacity(0.1),
                                colorText: Colors.red,
                              );
                              return;
                            }
                            Get.back();
                            auth.changePassword(passC.text);
                          },
                        ),
                      ),
                    ],
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}
