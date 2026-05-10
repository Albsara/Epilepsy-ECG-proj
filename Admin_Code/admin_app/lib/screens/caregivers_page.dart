import 'package:flutter/material.dart';
import 'package:get/get.dart';
import '../controllers/data_controller.dart';
import '../models/models.dart';
import '../utils/app_colors.dart';
import '../widgets/admin_shell.dart';
import '../widgets/common_widgets.dart';
import '../widgets/app_dialogs.dart';

class CaregiversPage extends StatelessWidget {
  const CaregiversPage({super.key});

  @override
  Widget build(BuildContext context) {
    final data = Get.find<DataController>();
    final isMobile = MediaQuery.of(context).size.width < 600;

    return AdminShell(
      title: 'caregivers'.tr,
      headerActions: [
        PrimaryButton(
          text: isMobile ? 'add'.tr : ('create_new_caregiver'.tr ?? 'Create Caregiver'),
          icon: Icons.person_add_alt_1,
          onPressed: () => _openCreateCaregiverForm(null),
        ),
      ],
      child: Obx(() {
        if (data.allCaregivers.isEmpty) {
          return SectionContainer(
            title: 'no_caregivers'.tr,
            subtitle: 'no_caregivers_desc'.tr ?? 'Add your first caregiver.',
            child: PrimaryButton(
              text: 'create_new_caregiver'.tr ?? 'Create Caregiver',
              icon: Icons.person_add_alt_1,
              onPressed: () => _openCreateCaregiverForm(null),
            ),
          );
        }

        return SectionContainer(
          title: 'all_caregivers'.tr ?? 'All Caregivers',
          subtitle: 'tap_caregiver_desc'.tr ?? 'Manage caregivers in the system.',
          child: ListView.separated(
            shrinkWrap: true,
            physics: const NeverScrollableScrollPhysics(),
            itemCount: data.allCaregivers.length,
            separatorBuilder: (_, __) =>
                const Divider(color: AppColors.border, height: 18),
            itemBuilder: (context, i) {
              final c = data.allCaregivers[i];
              return LayoutBuilder(
                builder: (context, constraints) {
                  final isMobile = constraints.maxWidth < 600;

                  Widget actionsDesktop() {
                    return Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        OutlineActionButton(
                          text: 'assign_patients'.tr ?? 'Assign Patients',
                          icon: Icons.personal_injury_outlined,
                          onPressed: () => _openAssignPatientDialog(c),
                        ),
                        const SizedBox(width: 10),
                        OutlineActionButton(
                          text: 'edit'.tr,
                          icon: Icons.edit_outlined,
                          onPressed: () => _openCreateCaregiverForm(c),
                        ),
                        const SizedBox(width: 10),
                        OutlineActionButton(
                          text: 'delete'.tr,
                          icon: Icons.delete_outline,
                          borderColor: AppColors.alert.withOpacity(0.35),
                          textColor: AppColors.alert,
                          onPressed: () => _confirmDeleteCaregiver(c.id),
                        ),
                      ],
                    );
                  }

                  Widget actionsMobile() {
                    return PopupMenuButton<String>(
                      color: Colors.white,
                      tooltip: 'actions'.tr,
                      onSelected: (v) {
                        if (v == 'assign') {
                          _openAssignPatientDialog(c);
                        } else if (v == 'edit') {
                          _openCreateCaregiverForm(c);
                        } else if (v == 'delete') {
                          _confirmDeleteCaregiver(c.id);
                        }
                      },
                      itemBuilder: (context) => [
                        PopupMenuItem(
                          value: 'assign',
                          child: Row(
                            children: [
                              Icon(Icons.personal_injury_outlined),
                              SizedBox(width: 10),
                              Text('assign_patients'.tr ?? 'Assign Patients'),
                            ],
                          ),
                        ),
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

                  return Container(
                    padding: const EdgeInsets.all(12),
                    decoration: BoxDecoration(
                      color: Colors.white,
                      borderRadius: BorderRadius.circular(12),
                      border: Border.all(color: AppColors.border),
                    ),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Row(
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
                                    c.name,
                                    style: const TextStyle(
                                      fontWeight: FontWeight.w900,
                                      color: AppColors.text,
                                    ),
                                  ),
                                  const SizedBox(height: 4),
                                  if (!isMobile)
                                    Text(
                                      '${c.email} • ${c.phone}',
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
                                '${'assigned_patients'.tr ?? 'Patients'}: ${c.patientIds.length}',
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
                            ] else ...[
                              actionsDesktop(),
                            ],
                          ],
                        ),
                        if (c.patientIds.isNotEmpty) ...[
                          const SizedBox(height: 12),
                          const Divider(color: AppColors.border, height: 1),
                          const SizedBox(height: 10),
                          Wrap(
                            spacing: 8,
                            runSpacing: 8,
                            children: c.patientIds.map((pid) {
                              final p = data.byId(pid);
                              if (p == null) return const SizedBox.shrink();
                              return Chip(
                                label: Text(p.name, style: const TextStyle(fontSize: 12)),
                                backgroundColor: AppColors.accent.withOpacity(0.05),
                                deleteIconColor: AppColors.alert,
                                onDeleted: () => _confirmUnassignPatient(c.id, pid),
                              );
                            }).toList(),
                          ),
                        ],
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

  void _openCreateCaregiverForm(Caregiver? caregiver) {
    final data = Get.find<DataController>();

    final nameC = TextEditingController(text: caregiver?.name ?? '');
    final emailC = TextEditingController(text: caregiver?.email ?? '');
    final phoneC = TextEditingController(text: caregiver?.phone ?? '');

    final isEdit = caregiver != null;

    Get.dialog(
      Center(
        child: ConstrainedBox(
          constraints: const BoxConstraints(maxWidth: 560),
          child: Material(
            color: Colors.transparent,
            child: Container(
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(14),
                border: Border.all(color: AppColors.border),
              ),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    isEdit ? 'edit_caregiver'.tr : 'add_caregiver'.tr,
                    style: const TextStyle(
                      fontSize: 16,
                      fontWeight: FontWeight.w900,
                      color: AppColors.text,
                    ),
                  ),
                  const SizedBox(height: 8),
                  Text(
                    'caregiver_form_desc'.tr ?? 'Enter details for caregiver.',
                    style: TextStyle(
                      fontSize: 12,
                      fontWeight: FontWeight.w700,
                      color: AppColors.muted,
                    ),
                  ),
                  const SizedBox(height: 14),
                  TextField(
                    controller: nameC,
                    decoration: InputDecoration(labelText: 'name'.tr),
                  ),
                  const SizedBox(height: 12),
                  TextField(
                    controller: emailC,
                    keyboardType: TextInputType.emailAddress,
                    decoration: InputDecoration(labelText: 'email'.tr),
                  ),
                  const SizedBox(height: 12),
                  TextField(
                    controller: phoneC,
                    keyboardType: TextInputType.phone,
                    decoration: InputDecoration(labelText: 'phone'.tr),
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
                          text: isEdit ? 'save'.tr : 'add'.tr,
                          icon: isEdit ? Icons.save : Icons.person_add_alt,
                          onPressed: () async {
                            final name = nameC.text.trim();
                            final email = emailC.text.trim();
                            final phone = phoneC.text.trim();

                            if (name.isEmpty ||
                                email.isEmpty ||
                                phone.isEmpty) {
                              AppDialogs.warning(
                                title: 'missing_fields'.tr,
                                message: 'missing_fields'.tr,
                              );
                              return;
                            }

                            Get.back();
                            try {
                              if (isEdit) {
                                await data.updateCaregiver(
                                  caregiverId: caregiver.id,
                                  name: name,
                                  email: email,
                                  phone: phone,
                                );
                              } else {
                                await data.addCaregiver(
                                  name: name,
                                  email: email,
                                  phone: phone,
                                );
                              }
                            } catch (e) {
                              AppDialogs.error(message: e.toString());
                            }
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
      barrierColor: Colors.black.withOpacity(0.35),
    );
  }

  void _openAssignPatientDialog(Caregiver caregiver) {
    final data = Get.find<DataController>();
    Get.dialog(
      Center(
        child: ConstrainedBox(
          constraints: const BoxConstraints(maxWidth: 560, maxHeight: 600),
          child: Material(
            color: Colors.transparent,
            child: Container(
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(14),
                border: Border.all(color: AppColors.border),
              ),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      Text(
                        'assign_patients'.tr ?? 'Assign Patients',
                        style: const TextStyle(
                          fontSize: 16,
                          fontWeight: FontWeight.w900,
                          color: AppColors.text,
                        ),
                      ),
                      IconButton(
                        icon: const Icon(Icons.close),
                        onPressed: () => Get.back(),
                        padding: EdgeInsets.zero,
                        constraints: const BoxConstraints(),
                      ),
                    ],
                  ),
                  const SizedBox(height: 14),
                  const Divider(color: AppColors.border, height: 1),
                  const SizedBox(height: 10),
                  Expanded(
                    child: Obx(() {
                      final all = data.patients.where((p) => !p.caregiverIds.contains(caregiver.id)).toList();
                      if (all.isEmpty) {
                        return Center(
                          child: Text(
                            'no_patients'.tr ?? 'No patients found.',
                            style: const TextStyle(
                              color: AppColors.muted,
                              fontWeight: FontWeight.w700,
                            ),
                          ),
                        );
                      }
                      return ListView.separated(
                        itemCount: all.length,
                        separatorBuilder: (_, __) => const Divider(color: AppColors.border, height: 18),
                        itemBuilder: (context, i) {
                          final p = all[i];

                          return Row(
                            children: [
                              const Icon(Icons.person, color: AppColors.muted),
                              const SizedBox(width: 10),
                              Expanded(
                                child: Column(
                                  crossAxisAlignment: CrossAxisAlignment.start,
                                  children: [
                                    Text(
                                      p.name,
                                      style: const TextStyle(fontWeight: FontWeight.w900),
                                    ),
                                    Text(
                                      '${p.email} • ${p.phone.isEmpty ? '—' : p.phone}',
                                      style: const TextStyle(
                                        fontSize: 12,
                                        fontWeight: FontWeight.w700,
                                        color: AppColors.muted,
                                      ),
                                    ),
                                  ],
                                ),
                              ),
                              const SizedBox(width: 10),
                              OutlineActionButton(
                                text: 'assign'.tr ?? 'Assign',
                                icon: Icons.check,
                                textColor: AppColors.accent,
                                borderColor: AppColors.accent,
                                onPressed: () => data.assignCaregiver(p.id, caregiver.id),
                              ),
                            ],
                          );
                        },
                      );
                    }),
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
      barrierColor: Colors.black.withOpacity(0.35),
    );
  }

  Future<void> _confirmUnassignPatient(String caregiverId, String patientId) async {
    final confirmed = await AppDialogs.confirm(
      title: 'unassign'.tr ?? 'Unassign Patient',
      message: 'unassign_desc'.tr ?? 'Are you sure you want to unassign this patient?',
      confirmText: 'unassign'.tr ?? 'Remove',
      isDanger: true,
    );

    if (confirmed) {
      Get.find<DataController>().unassignCaregiver(patientId, caregiverId);
    }
  }

  Future<void> _confirmDeleteCaregiver(String caregiverId) async {
    final confirmed = await AppDialogs.confirm(
      title: 'delete_caregiver_title'.tr,
      message: 'delete_caregiver_desc'.tr ?? 'This will remove the caregiver from the system entirely.',
      confirmText: 'delete'.tr,
      isDanger: true,
    );

    if (confirmed) {
      final data = Get.find<DataController>();
      await data.deleteCaregiver(caregiverId);
    }
  }
}
