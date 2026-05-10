import 'package:flutter/material.dart';
import 'package:get/get.dart';
import '../controllers/data_controller.dart';
import '../routes.dart';
import '../utils/app_colors.dart';
import '../widgets/admin_shell.dart';
import '../widgets/common_widgets.dart';
import '../widgets/app_dialogs.dart';

class AddPatientPage extends StatelessWidget {
  const AddPatientPage({super.key});

  @override
  Widget build(BuildContext context) {
    return const _PatientFormPage(title: 'Add patient', isEdit: false);
  }
}

class EditPatientPage extends StatelessWidget {
  const EditPatientPage({super.key});

  @override
  Widget build(BuildContext context) {
    return const _PatientFormPage(title: 'Edit personal info', isEdit: true);
  }
}

class _PatientFormPage extends StatefulWidget {
  final String title;
  final bool isEdit;

  const _PatientFormPage({required this.title, required this.isEdit});

  @override
  State<_PatientFormPage> createState() => _PatientFormPageState();
}

class _PatientFormPageState extends State<_PatientFormPage> {
  final data = Get.find<DataController>();

  final emailC = TextEditingController();
  final nameC = TextEditingController();
  final phoneC = TextEditingController();
  final detailsC = TextEditingController();
  DateTime? birthdate;

  String? patientId;

  @override
  void initState() {
    super.initState();
    if (widget.isEdit) {
      final args = (Get.arguments as Map?) ?? {};
      patientId = args['patientId']?.toString();
      final p = patientId == null ? null : data.byId(patientId!);
      if (p != null) {
        emailC.text = p.email;
        nameC.text = p.name;
        phoneC.text = p.phone;
        detailsC.text = p.details;
        birthdate = p.birthdate;
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return AdminShell(
      title: widget.title,
      child: SectionContainer(
        title: widget.isEdit ? 'Update patient info' : 'Create new patient',
        subtitle: 'Email, name, birthdate, phone number, and details.',
        trailing: OutlineActionButton(
          text: 'Back',
          icon: Icons.arrow_back,
          onPressed: () => Get.back(),
        ),
        child: LayoutBuilder(
          builder: (context, constraints) {
            final wide = constraints.maxWidth >= 900;

            final fields = <Widget>[
              TextField(
                controller: emailC,
                decoration: const InputDecoration(labelText: 'Email'),
              ),
              TextField(
                controller: nameC,
                decoration: const InputDecoration(labelText: 'Name'),
              ),
              _BirthdateField(
                selected: birthdate,
                onPick: (d) => setState(() => birthdate = d),
              ),
              TextField(
                controller: phoneC,
                decoration: const InputDecoration(labelText: 'Phone number'),
              ),
              TextField(
                controller: detailsC,
                decoration: const InputDecoration(labelText: 'Details'),
                maxLines: 4,
              ),
            ];

            Widget actionRow() {
              return Row(
                children: [
                  Expanded(
                    child: OutlineActionButton(
                      text: 'Cancel',
                      onPressed: () => Get.back(),
                    ),
                  ),
                  const SizedBox(width: 10),
                  Expanded(
                    child: PrimaryButton(
                      text: widget.isEdit ? 'Save changes' : 'Add patient',
                      icon: widget.isEdit ? Icons.save : Icons.person_add_alt_1,
                      onPressed: _submit,
                    ),
                  ),
                ],
              );
            }

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
                      Expanded(child: fields[3]),
                    ],
                  ),
                  const SizedBox(height: 12),
                  fields[4],
                  const SizedBox(height: 14),
                  actionRow(), // ✅ always row
                ],
              );
            }

            // Mobile / small screens: keep buttons row, not stacked
            return Column(
              children: [
                for (int i = 0; i < fields.length; i++) ...[
                  fields[i],
                  if (i != fields.length - 1) const SizedBox(height: 12),
                ],
                const SizedBox(height: 14),
                actionRow(), // ✅ always row
              ],
            );
          },
        ),
      ),
    );
  }

  Future<void> _submit() async {
    final email = emailC.text.trim();
    final name = nameC.text.trim();
    final phone = phoneC.text.trim();
    final details = detailsC.text.trim();
    final b = birthdate;

    if (email.isEmpty || name.isEmpty || phone.isEmpty || b == null) {
      AppDialogs.warning(
        title: 'missing_fields'.tr,
        message: 'fill_all_fields'.tr,
      );
      return;
    }

    if (widget.isEdit) {
      if (patientId == null) return;
      await data.updatePatient(
        id: patientId!,
        email: email,
        name: name,
        birthdate: b,
        phone: phone,
        details: details,
      );
      Get.back();
      return;
    }

    await data.addPatient(
      email: email,
      name: name,
      birthdate: b,
      phone: phone,
      details: details,
    );
    Get.offAllNamed(Routes.patients);
  }
}

class _BirthdateField extends StatelessWidget {
  final DateTime? selected;
  final ValueChanged<DateTime> onPick;

  const _BirthdateField({required this.selected, required this.onPick});

  @override
  Widget build(BuildContext context) {
    final label = selected == null
        ? 'Select birthdate'
        : '${selected!.year}-${selected!.month.toString().padLeft(2, '0')}-${selected!.day.toString().padLeft(2, '0')}';

    return InkWell(
      onTap: () async {
        final now = DateTime.now();
        final picked = await showDatePicker(
          context: context,
          initialDate: selected ?? DateTime(now.year - 18, now.month, now.day),
          firstDate: DateTime(1900),
          lastDate: now,
          builder: (context, child) {
            return Theme(
              data: Theme.of(context).copyWith(
                colorScheme: ColorScheme.fromSeed(seedColor: AppColors.accent),
              ),
              child: child!,
            );
          },
        );
        if (picked != null) onPick(picked);
      },
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 14),
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(12),
          border: Border.all(color: AppColors.border),
        ),
        child: Row(
          children: [
            const Icon(Icons.cake_outlined, color: AppColors.muted),
            const SizedBox(width: 10),
            Expanded(
              child: Text(
                label,
                style: TextStyle(
                  color: selected == null ? AppColors.muted : AppColors.text,
                  fontWeight: FontWeight.w900,
                ),
              ),
            ),
            const Icon(Icons.calendar_month, color: AppColors.muted),
          ],
        ),
      ),
    );
  }
}
