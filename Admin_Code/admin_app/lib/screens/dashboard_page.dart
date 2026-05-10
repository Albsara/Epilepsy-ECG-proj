import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:flutter/material.dart';
import 'package:get/get.dart';
import '../controllers/dashboard_controller.dart';
import '../controllers/data_controller.dart';
import '../routes.dart';
import '../utils/app_colors.dart';
import '../widgets/admin_shell.dart';
import '../widgets/common_widgets.dart';

class DashboardPage extends StatelessWidget {
  DashboardPage({super.key});

  final dash = Get.find<DashboardController>();

  @override
  Widget build(BuildContext context) {
    final data = Get.find<DataController>();
    final fs = FirebaseFirestore.instance;

    final now = DateTime.now();
    final startOfDay = DateTime(now.year, now.month, now.day);
    final endOfDay = startOfDay.add(const Duration(days: 1));

    // Streams (keep them here; refresh is handled by rebuilding with a new Key)
    final caregiversStream = fs.collection('caregivers').snapshots();
    final alertsTotalStream = fs.collectionGroup('alerts').snapshots();
    final alertsTodayStream = fs
        .collectionGroup('alerts')
        .where('time', isGreaterThanOrEqualTo: Timestamp.fromDate(startOfDay))
        .where('time', isLessThan: Timestamp.fromDate(endOfDay))
        .snapshots();
    final latest3Stream = fs
        .collectionGroup('alerts')
        .orderBy('time', descending: true)
        .limit(3)
        .snapshots();

    return AdminShell(
      title: 'dashboard'.tr,
      headerActions: [
        PrimaryButton(
          text: 'add_patient'.tr,
          icon: Icons.person_add_alt_1,
          onPressed: () => Get.toNamed(Routes.addPatient),
        ),
      ],
      child: Obx(() {
        final tick = dash.refreshTick.value;

        final patientsCount = data.patients.length;

        return Column(
          key: ValueKey(
            tick,
          ), // IMPORTANT: forces StreamBuilders to re-subscribe
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            LayoutBuilder(
              builder: (context, constraints) {
                final isWide = constraints.maxWidth >= 900;

                final caregiversBox = StreamBuilder<QuerySnapshot>(
                  stream: caregiversStream,
                  builder: (context, snap) {
                    final caregiversCount = snap.hasData
                        ? snap.data!.docs.length
                        : 0;
                    return _StatBox(
                      label: 'caregivers'.tr,
                      value: '$caregiversCount',
                    );
                  },
                );
                final alertsTodayBox = StreamBuilder<QuerySnapshot>(
                  stream: alertsTodayStream,
                  builder: (context, snap) {
                    // ✅ PRINT errors to console
                    if (snap.hasError) {
                      debugPrint('alertsTodayStream ERROR: ${snap.error}');
                      debugPrint('alertsTodayStream STACK: ${snap.stackTrace}');
                    }

                    // Optional: show the error in the UI too
                    if (snap.hasError) {
                      return _StatBox(
                        label: 'seizure_alerts_today'.tr,
                        value: 'ERR',
                        accent: AppColors.alert,
                      );
                    }

                    final count = snap.hasData ? snap.data!.docs.length : 0;

                    return _StatBox(
                      label: 'seizure_alerts_today'.tr,
                      value: '$count',
                      accent: AppColors.alert,
                    );
                  },
                );

                final alertsTotalBox = StreamBuilder<QuerySnapshot>(
                  stream: alertsTotalStream,
                  builder: (context, snap) {
                    final count = snap.hasData ? snap.data!.docs.length : 0;
                    return _StatBox(
                      label: 'seizure_alerts_total'.tr,
                      value: '$count',
                    );
                  },
                );

                final statItems = [
                  _StatBox(label: 'patients'.tr, value: '$patientsCount'),
                  caregiversBox,
                  alertsTodayBox,
                  alertsTotalBox,
                ];

                if (isWide) {
                  return Row(
                    children: [
                      for (int i = 0; i < statItems.length; i++) ...[
                        Expanded(child: statItems[i]),
                        if (i != statItems.length - 1)
                          const SizedBox(width: 12),
                      ],
                    ],
                  );
                }

                return Column(
                  children: [
                    for (int i = 0; i < statItems.length; i++) ...[
                      statItems[i],
                      if (i != statItems.length - 1) const SizedBox(height: 12),
                    ],
                  ],
                );
              },
            ),
            const SizedBox(height: 14),
            SectionContainer(
              title: 'latest_alerts'.tr,
              subtitle: 'latest_alerts_desc'.tr,
              trailing: OutlineActionButton(
                text: 'view_patients'.tr,
                icon: Icons.people_alt_outlined,
                onPressed: () => Get.offAllNamed(Routes.patients),
              ),
              child: StreamBuilder<QuerySnapshot<Map<String, dynamic>>>(
                stream: latest3Stream,
                builder: (context, snap) {
                  if (snap.hasError) {
                    debugPrint(snap.error.toString());

                    return Text(
                      '${'error'.tr}: ${snap.error}',
                      style: const TextStyle(
                        color: AppColors.alert,
                        fontWeight: FontWeight.w800,
                      ),
                    );
                  }

                  if (snap.connectionState == ConnectionState.waiting) {
                    return Text(
                      'loading'.tr,
                      style: TextStyle(
                        color: AppColors.muted,
                        fontWeight: FontWeight.w700,
                      ),
                    );
                  }

                  final docs = snap.data?.docs ?? [];
                  if (docs.isEmpty) {
                    return Text(
                      'no_alerts'.tr,
                      style: TextStyle(
                        color: AppColors.muted,
                        fontWeight: FontWeight.w700,
                      ),
                    );
                  }

                  final alertItems = docs.map((d) {
                    // Path: patients/{id}/alerts/{alertId}
                    final parent = d.reference.parent.parent;
                    final patientId = parent?.id ?? '';

                    final dataMap = d.data();
                    final ts = dataMap['time'];

                    final bool badTime = ts is! Timestamp;

                    final DateTime time = badTime
                        ? DateTime.now()
                        : ts.toDate();

                    final hrRaw = dataMap['heartRate'];
                    final hrvRaw = dataMap['hrv'];

                    final hr = hrRaw is int
                        ? hrRaw
                        : int.tryParse('$hrRaw') ?? 0;
                    final hrv = hrvRaw is int
                        ? hrvRaw
                        : int.tryParse('$hrvRaw') ?? 0;

                    return {
                      'patientId': patientId,
                      'time': time,
                      'hr': hr,
                      'hrv': hrv,
                      'badTime': badTime,
                    };
                  }).toList();

                  return ListView.separated(
                    shrinkWrap: true,
                    physics: const NeverScrollableScrollPhysics(),
                    itemCount: alertItems.length,
                    separatorBuilder: (_, __) =>
                        const Divider(color: AppColors.border, height: 18),
                    itemBuilder: (context, i) {
                      final item = alertItems[i];
                      final patientId = item['patientId'] as String;
                      final time = item['time'] as DateTime;
                      final hr = item['hr'] as int;
                      final hrv = item['hrv'] as int;
                      final badTime = item['badTime'] as bool;

                      final p = data.byId(patientId);

                      return LayoutBuilder(
                        builder: (context, constraints) {
                          final isWide = constraints.maxWidth >= 700;

                          void goToDetail() {
                            if (patientId.isEmpty) return;
                            Get.toNamed(
                              Routes.patientDetail,
                              arguments: {'patientId': patientId},
                            );
                          }

                          final leftBar = Container(
                            width: 10,
                            height: 44,
                            decoration: BoxDecoration(
                              color: AppColors.alert,
                              borderRadius: BorderRadius.circular(8),
                            ),
                          );

                          final titleBlock = Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Text(
                                p == null
                                    ? '${'unknown_patient'.tr} $patientId'
                                    : p.name,
                                style: const TextStyle(
                                  fontWeight: FontWeight.w900,
                                  color: AppColors.text,
                                ),
                              ),
                              const SizedBox(height: 4),
                              Text(
                                badTime
                                    ? 'invalid_time'.tr
                                    : formatDateTimeAmPm(time),
                                style: TextStyle(
                                  fontSize: 12,
                                  fontWeight: FontWeight.w700,
                                  color: badTime
                                      ? AppColors.alert
                                      : AppColors.muted,
                                ),
                              ),
                            ],
                          );

                          final metrics = Column(
                            crossAxisAlignment: CrossAxisAlignment.end,
                            children: [
                              Text(
                                'HR: $hr',
                                style: const TextStyle(
                                  fontWeight: FontWeight.w900,
                                ),
                              ),
                              Text(
                                'HRV: $hrv',
                                style: const TextStyle(
                                  fontWeight: FontWeight.w900,
                                ),
                              ),
                            ],
                          );

                          final arrow = Container(
                            padding: const EdgeInsets.all(8),
                            decoration: BoxDecoration(
                              color: Colors.white,
                              borderRadius: BorderRadius.circular(10),
                              border: Border.all(color: AppColors.border),
                            ),
                            child: const Icon(
                              Icons.chevron_right,
                              color: AppColors.muted,
                            ),
                          );

                          return Material(
                            color: Colors.transparent,
                            child: InkWell(
                              onTap: goToDetail,
                              borderRadius: BorderRadius.circular(12),
                              child: Container(
                                padding: const EdgeInsets.all(12),
                                decoration: BoxDecoration(
                                  color: AppColors.alert.withOpacity(0.06),
                                  borderRadius: BorderRadius.circular(12),
                                  border: Border.all(
                                    color: AppColors.alert.withOpacity(0.22),
                                  ),
                                ),
                                child: isWide
                                    ? Row(
                                        children: [
                                          leftBar,
                                          const SizedBox(width: 12),
                                          Expanded(child: titleBlock),
                                          const SizedBox(width: 12),
                                          metrics,
                                          const SizedBox(width: 12),
                                          arrow,
                                        ],
                                      )
                                    : Column(
                                        crossAxisAlignment:
                                            CrossAxisAlignment.start,
                                        children: [
                                          Row(
                                            children: [
                                              leftBar,
                                              const SizedBox(width: 12),
                                              Expanded(child: titleBlock),
                                              const SizedBox(width: 10),
                                              arrow,
                                            ],
                                          ),
                                          const SizedBox(height: 10),
                                          Align(
                                            alignment: Alignment.centerRight,
                                            child: metrics,
                                          ),
                                        ],
                                      ),
                              ),
                            ),
                          );
                        },
                      );
                    },
                  );
                },
              ),
            ),
          ],
        );
      }),
    );
  }
}

class _StatBox extends StatelessWidget {
  final String label;
  final String value;
  final Color accent;

  const _StatBox({
    required this.label,
    required this.value,
    this.accent = AppColors.accent,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: AppColors.border),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.03),
            blurRadius: 16,
            offset: const Offset(0, 10),
          ),
        ],
      ),
      child: Row(
        children: [
          Container(
            width: 10,
            height: 48,
            decoration: BoxDecoration(
              color: accent,
              borderRadius: BorderRadius.circular(10),
            ),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  label,
                  style: const TextStyle(
                    fontSize: 12,
                    fontWeight: FontWeight.w800,
                    color: AppColors.muted,
                  ),
                ),
                const SizedBox(height: 6),
                Text(
                  value,
                  style: const TextStyle(
                    fontSize: 22,
                    fontWeight: FontWeight.w900,
                    color: AppColors.text,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
