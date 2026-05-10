import 'package:flutter/material.dart';
import 'package:get/get.dart';
import '../controllers/data_controller.dart';
import '../routes.dart';
import '../utils/app_colors.dart';
import '../widgets/admin_shell.dart';
import '../widgets/common_widgets.dart';
import '../widgets/app_dialogs.dart';

class PatientsPage extends StatelessWidget {
  const PatientsPage({super.key});

  @override
  Widget build(BuildContext context) {
    final data = Get.find<DataController>();
    final isMobile = MediaQuery.of(context).size.width < 600;

    return AdminShell(
      title: 'patients'.tr,
      headerActions: [
        PrimaryButton(
          text: isMobile ? 'add'.tr : 'add_patient'.tr,
          icon: Icons.person_add_alt_1,
          onPressed: () => Get.toNamed(Routes.addPatient),
        ),
      ],
      child: Obx(() {
        if (data.patients.isEmpty) {
          return SectionContainer(
            title: 'no_patients'.tr,
            subtitle: 'no_patients_desc'.tr,
            child: PrimaryButton(
              text: 'add_patient'.tr,
              icon: Icons.person_add_alt_1,
              onPressed: () => Get.toNamed(Routes.addPatient),
            ),
          );
        }

        return SectionContainer(
          title: 'all_patients'.tr,
          subtitle: 'tap_patient_desc'.tr,
          child: ListView.separated(
            shrinkWrap: true,
            physics: const NeverScrollableScrollPhysics(),
            itemCount: data.patients.length,
            separatorBuilder: (_, __) =>
                const Divider(color: AppColors.border, height: 18),
            itemBuilder: (context, i) {
              final p = data.patients[i];
              return LayoutBuilder(
                builder: (context, constraints) {
                  final isMobile = constraints.maxWidth < 600;

                  void openDetail() => Get.toNamed(
                    Routes.patientDetail,
                    arguments: {'patientId': p.id},
                  );

                  Widget actionsDesktop() {
                    return Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        OutlineActionButton(
                          text: 'edit'.tr,
                          icon: Icons.edit_outlined,
                          onPressed: () => Get.toNamed(
                            Routes.editPatient,
                            arguments: {'patientId': p.id},
                          ),
                        ),
                        const SizedBox(width: 10),
                        OutlineActionButton(
                          text: 'delete'.tr,
                          icon: Icons.delete_outline,
                          borderColor: AppColors.alert.withOpacity(0.35),
                          textColor: AppColors.alert,
                          onPressed: () => _confirmDeletePatient(p.id),
                        ),
                      ],
                    );
                  }

                  Widget actionsMobile() {
                    return PopupMenuButton<String>(
                      color: Colors.white,
                      tooltip: 'actions'.tr,
                      onSelected: (v) {
                        if (v == 'edit') {
                          Get.toNamed(
                            Routes.editPatient,
                            arguments: {'patientId': p.id},
                          );
                        } else if (v == 'delete') {
                          _confirmDeletePatient(p.id);
                        }
                      },
                      itemBuilder: (context) => [
                        PopupMenuItem(
                          value: 'edit',
                          child: Row(
                            children: [
                              Icon(Icons.edit_outlined),
                              SizedBox(width: 10),
                              Text('edit'.tr),
                            ],
                          ),
                        ),
                        PopupMenuItem(
                          value: 'delete',
                          child: Row(
                            children: [
                              Icon(
                                Icons.delete_outline,
                                color: AppColors.alert,
                              ),
                              SizedBox(width: 10),
                              Text('delete'.tr),
                            ],
                          ),
                        ),
                      ],
                      child: Container(
                        padding: const EdgeInsets.all(8),
                        decoration: BoxDecoration(
                          color: Colors.white,
                          borderRadius: BorderRadius.circular(10),
                          border: Border.all(color: AppColors.border),
                        ),
                        child: const Icon(
                          Icons.more_vert,
                          color: AppColors.muted,
                        ),
                      ),
                    );
                  }

                  return Material(
                    color: Colors.transparent,
                    child: InkWell(
                      onTap: openDetail,
                      borderRadius: BorderRadius.circular(12),
                      child: Container(
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
                                color: AppColors.accent.withOpacity(0.85),
                                borderRadius: BorderRadius.circular(8),
                              ),
                            ),
                            const SizedBox(width: 12),
                            Expanded(
                              child: Column(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                                  Text(
                                    p.name,
                                    style: const TextStyle(
                                      fontWeight: FontWeight.w900,
                                      color: AppColors.text,
                                    ),
                                  ),
                                  const SizedBox(height: 4),
                                  if (!isMobile)
                                    Text(
                                      '${p.email} • ${p.phone}',
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
                            if (!isMobile)
                              Text(
                                '${'caregivers'.tr}: ${p.caregivers.length}',
                                style: const TextStyle(
                                  fontSize: 12,
                                  fontWeight: FontWeight.w900,
                                  color: AppColors.muted,
                                ),
                              ),
                            if (!isMobile) const SizedBox(width: 12),

                            // Actions
                            if (isMobile) ...[
                              actionsMobile(),
                              const SizedBox(width: 8),
                              const Icon(
                                Icons.chevron_right,
                                color: AppColors.border,
                              ),
                            ] else ...[
                              actionsDesktop(),
                            ],
                          ],
                        ),
                      ),
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

  Future<void> _confirmDeletePatient(String patientId) async {
    final confirmed = await AppDialogs.confirm(
      title: 'delete_patient_title'.tr,
      message: 'delete_patient_desc'.tr,
      confirmText: 'delete'.tr,
      isDanger: true,
    );

    if (confirmed) {
      final data = Get.find<DataController>();
      await data.deletePatient(patientId);
    }
  }
}
