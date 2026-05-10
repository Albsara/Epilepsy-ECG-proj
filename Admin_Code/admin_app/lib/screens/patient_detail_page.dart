import 'package:flutter/material.dart';
import 'package:get/get.dart';
import '../controllers/data_controller.dart';
import '../models/models.dart';
import '../routes.dart';
import '../utils/app_colors.dart';
import '../widgets/admin_shell.dart';

import '../widgets/common_widgets.dart';
import '../widgets/app_dialogs.dart';

class PatientDetailPage extends StatelessWidget {
  const PatientDetailPage({super.key});

  @override
  Widget build(BuildContext context) {
    final data = Get.find<DataController>();
    final isMobile = MediaQuery.of(context).size.width < 600;
    final args = (Get.arguments as Map?) ?? {};
    final patientId = args['patientId']?.toString() ?? '';

    // IMPORTANT: attach Firestore streams for caregivers + alerts when entering detail
    data.attachDetailStreams(patientId);

    final p = data.byId(patientId);

    if (p == null) {
      return AdminShell(
        title: 'Patient',
        child: SectionContainer(
          title: 'not_found'.tr,
          subtitle: 'patient_not_found'.tr,
          child: OutlineActionButton(
            text: 'back'.tr,
            icon: Icons.arrow_back,
            onPressed: () => Get.back(),
          ),
        ),
      );
    }

    return AdminShell(
      title: 'Patient',
      headerActions: [
        OutlineActionButton(
          text: isMobile ? 'edit'.tr : 'edit_info'.tr,
          icon: Icons.edit_outlined,
          onPressed: () =>
              Get.toNamed(Routes.editPatient, arguments: {'patientId': p.id}),
        ),
      ],
      child: Obx(() {
        final live = p.liveStatus.value;

        return Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            SectionContainer(
              title: 'personal_info'.tr,
              subtitle: 'patient_form_desc'.tr,
              trailing: OutlineActionButton(
                text: 'back'.tr,
                icon: Icons.arrow_back,
                onPressed: () => Get.back(),
              ),
              child: Column(
                children: [
                  KeyValueRow(k: 'name'.tr, v: p.name),
                  KeyValueRow(k: 'email'.tr, v: p.email),
                  KeyValueRow(
                    k: 'birthdate'.tr,
                    v: '${p.birthdate.year}-${p.birthdate.month.toString().padLeft(2, '0')}-${p.birthdate.day.toString().padLeft(2, '0')}',
                  ),
                  KeyValueRow(k: 'phone'.tr, v: p.phone),
                  const Divider(color: AppColors.border, height: 18),
                  Align(
                    alignment: Alignment.centerLeft,
                    child: Text(
                      p.details.isEmpty ? '—' : p.details,
                      style: const TextStyle(
                        fontSize: 12,
                        fontWeight: FontWeight.w800,
                        color: AppColors.text,
                      ),
                    ),
                  ),
                ],
              ),
            ),
            SectionContainer(
              title: 'live_status'.tr,
              subtitle: 'Current situation of the patient',
              child: LayoutBuilder(
                builder: (context, constraints) {
                  final wide = constraints.maxWidth >= 900;
                  final items = [
                    _LivePill(label: 'HR', value: '${live.hr}'),
                    _LivePill(label: 'HRV', value: '${live.hrv}'),
                    _LivePill(label: 'medication'.tr, value: live.medication),
                    _LivePill(label: 'symptoms'.tr, value: live.symptoms),
                    _LivePill(label: 'sleep'.tr, value: live.sleep.label),
                    _LivePill(label: 'stress'.tr, value: live.stress),
                  ];

                  if (wide) {
                    return Wrap(spacing: 10, runSpacing: 10, children: items);
                  }

                  return Column(
                    children: [
                      for (int i = 0; i < items.length; i++) ...[
                        items[i],
                        if (i != items.length - 1) const SizedBox(height: 10),
                      ],
                    ],
                  );
                },
              ),
            ),
            SectionContainer(
              title: 'caregivers'.tr,
              subtitle:
                  'Add, edit, or remove caregivers linked to this patient.',
              trailing: PrimaryButton(
                text: 'assign_caregiver'.tr ?? 'Assign',
                icon: Icons.person_add_alt,
                onPressed: () => _openAssignCaregiverDialog(p.id),
              ),
              child: p.caregivers.isEmpty
                  ? Text(
                      'no_caregivers'.tr,
                      style: TextStyle(
                        color: AppColors.muted,
                        fontWeight: FontWeight.w700,
                      ),
                    )
                  : ListView.separated(
                      shrinkWrap: true,
                      physics: const NeverScrollableScrollPhysics(),
                      itemCount: p.caregivers.length,
                      separatorBuilder: (_, __) =>
                          const Divider(color: AppColors.border, height: 18),
                      itemBuilder: (context, i) {
                        final c = p.caregivers[i];
                        return LayoutBuilder(
                          builder: (context, constraints) {
                            final isMobile = constraints.maxWidth < 600;

                            Widget actionsDesktop() {
                              return Row(
                                mainAxisSize: MainAxisSize.min,
                                children: [
                                  OutlineActionButton(
                                    text: 'edit'.tr,
                                    icon: Icons.edit_outlined,
                                    onPressed: () => _openCreateCaregiverForm(
                                      patientId: p.id,
                                      caregiver: c,
                                    ),
                                  ),
                                  const SizedBox(width: 10),
                                  OutlineActionButton(
                                    text: 'unassign'.tr ?? 'Remove',
                                    icon: Icons.link_off,
                                    borderColor: AppColors.alert.withOpacity(
                                      0.35,
                                    ),
                                    textColor: AppColors.alert,
                                    onPressed: () => _confirmUnassignCaregiver(
                                      patientId: p.id,
                                      caregiverId: c.id,
                                    ),
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
                                    _openCreateCaregiverForm(
                                      patientId: p.id,
                                      caregiver: c,
                                    );
                                  } else if (v == 'unassign') {
                                    _confirmUnassignCaregiver(
                                      patientId: p.id,
                                      caregiverId: c.id,
                                    );
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
                                    value: 'unassign',
                                    child: Row(
                                      children: [
                                        Icon(
                                          Icons.link_off,
                                          color: AppColors.alert,
                                        ),
                                        SizedBox(width: 10),
                                        Text('unassign'.tr ?? 'Remove'),
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
                              child: Row(
                                children: [
                                  const Icon(
                                    Icons.support_agent,
                                    color: AppColors.muted,
                                  ),
                                  const SizedBox(width: 10),
                                  Expanded(
                                    child: Column(
                                      crossAxisAlignment:
                                          CrossAxisAlignment.start,
                                      children: [
                                        Text(
                                          c.name,
                                          style: const TextStyle(
                                            fontWeight: FontWeight.w900,
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
                                        if (isMobile)
                                          Text(
                                            c.phone.isEmpty ? '—' : c.phone,
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

                                  // Actions
                                  if (isMobile)
                                    actionsMobile()
                                  else
                                    actionsDesktop(),
                                ],
                              ),
                            );
                          },
                        );
                      },
                    ),
            ),
            SectionContainer(
              title: 'alerts'.tr,
              subtitle: 'history of seizure alerts for this patient',
              child: p.alerts.isEmpty
                  ? Text(
                      'no_alerts'.tr,
                      style: TextStyle(
                        color: AppColors.muted,
                        fontWeight: FontWeight.w700,
                      ),
                    )
                  : ListView.separated(
                      shrinkWrap: true,
                      physics: const NeverScrollableScrollPhysics(),
                      itemCount: p.alerts.length,
                      separatorBuilder: (_, __) =>
                          const Divider(color: AppColors.border, height: 18),
                      itemBuilder: (context, i) {
                        final a = p.alerts[i];
                        final Color statusColor = a.isHandled ? Colors.green : AppColors.alert;
                        
                        return Container(
                          padding: const EdgeInsets.all(12),
                          decoration: BoxDecoration(
                            color: statusColor.withOpacity(0.06),
                            borderRadius: BorderRadius.circular(12),
                            border: Border.all(
                              color: statusColor.withOpacity(0.22),
                            ),
                          ),
                          child: Row(
                            children: [
                              Container(
                                width: 10,
                                height: 44,
                                decoration: BoxDecoration(
                                  color: statusColor,
                                  borderRadius: BorderRadius.circular(8),
                                ),
                              ),
                              const SizedBox(width: 12),
                              Expanded(
                                child: Column(
                                  crossAxisAlignment: CrossAxisAlignment.start,
                                  children: [
                                    Text(
                                      formatDateTimeAmPm(a.time),
                                      style: const TextStyle(
                                        fontWeight: FontWeight.w900,
                                        color: AppColors.text,
                                      ),
                                    ),
                                    const SizedBox(height: 4),
                                    Text(
                                      a.isHandled ? 'Handled'.tr : 'Requires Review'.tr,
                                      style: TextStyle(
                                        fontSize: 11,
                                        fontWeight: FontWeight.w900,
                                        color: statusColor,
                                      ),
                                    ),
                                    const SizedBox(height: 4),
                                    Text(
                                      'Med: ${a.medication} • Symp: ${a.symptoms} • Sleep: ${a.sleep} • Stress: ${a.stress}',
                                      style: TextStyle(
                                        fontSize: 10,
                                        fontWeight: FontWeight.w700,
                                        color: AppColors.muted,
                                      ),
                                    ),
                                  ],
                                ),
                              ),
                              const SizedBox(width: 12),
                              Column(
                                crossAxisAlignment: CrossAxisAlignment.end,
                                children: [
                                  Text(
                                    'HR: ${a.heartRate}',
                                    style: const TextStyle(
                                      fontWeight: FontWeight.w900,
                                    ),
                                  ),
                                  Text(
                                    'HRV: ${a.hrv}',
                                    style: const TextStyle(
                                      fontWeight: FontWeight.w900,
                                    ),
                                  ),
                                ],
                              ),
                            ],
                          ),
                        );
                      },
                    ),
            ),
          ],
        );
      }),
    );
  }

  void _openAssignCaregiverDialog(String patientId) {
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
                        'assign_caregiver'.tr ?? 'Assign Caregivers',
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
                  PrimaryButton(
                    text: 'create_new_caregiver'.tr ?? 'Create New Caregiver',
                    icon: Icons.add,
                    onPressed: () {
                      Get.back();
                      _openCreateCaregiverForm(patientId: patientId);
                    },
                  ),
                  const SizedBox(height: 14),
                  const Divider(color: AppColors.border, height: 1),
                  const SizedBox(height: 10),
                  Expanded(
                    child: Obx(() {
                      final all = data.allCaregivers.where((c) => !c.patientIds.contains(patientId)).toList();
                      if (all.isEmpty) {
                        return Center(
                          child: Text(
                            'no_caregivers'.tr ?? 'No caregivers found.',
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
                          final c = all[i];

                          return Row(
                            children: [
                              const Icon(Icons.support_agent, color: AppColors.muted),
                              const SizedBox(width: 10),
                              Expanded(
                                child: Column(
                                  crossAxisAlignment: CrossAxisAlignment.start,
                                  children: [
                                    Text(
                                      c.name,
                                      style: const TextStyle(fontWeight: FontWeight.w900),
                                    ),
                                    Text(
                                      '${c.email} • ${c.phone.isEmpty ? '—' : c.phone}',
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
                                onPressed: () => data.assignCaregiver(patientId, c.id),
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

  void _openCreateCaregiverForm({String? patientId, Caregiver? caregiver}) {
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
                                  assignToPatientId: patientId,
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

  Future<void> _confirmUnassignCaregiver({
    required String patientId,
    required String caregiverId,
  }) async {
    final confirmed = await AppDialogs.confirm(
      title: 'unassign'.tr ?? 'Unassign Caregiver',
      message: 'unassign_desc'.tr ?? 'Are you sure you want to unassign this caregiver?',
      confirmText: 'unassign'.tr ?? 'Remove',
      isDanger: true,
    );

    if (confirmed) {
      final data = Get.find<DataController>();
      await data.unassignCaregiver(patientId, caregiverId);
    }
  }
}

class _LivePill extends StatelessWidget {
  final String label;
  final String value;

  const _LivePill({required this.label, required this.value});

  @override
  Widget build(BuildContext context) {
    return Container(
      constraints: const BoxConstraints(minWidth: 180),
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: AppColors.border),
      ),
      child: Row(
        children: [
          Expanded(
            child: Text(
              label,
              style: const TextStyle(
                fontSize: 12,
                fontWeight: FontWeight.w900,
                color: AppColors.muted,
              ),
            ),
          ),
          const SizedBox(width: 10),
          Expanded(
            child: Text(
              value,
              style: const TextStyle(
                fontSize: 12,
                fontWeight: FontWeight.w900,
                color: AppColors.text,
              ),
              textAlign: TextAlign.right,
              overflow: TextOverflow.ellipsis,
            ),
          ),
        ],
      ),
    );
  }
}
