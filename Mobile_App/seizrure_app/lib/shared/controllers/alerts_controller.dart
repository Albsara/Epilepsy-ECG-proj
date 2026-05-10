import 'package:get/get.dart';
import '../services/firestore_service.dart';
import '../services/notification_service.dart';
import '../models/alert_model.dart';
import 'package:intl/intl.dart';
import 'package:flutter/material.dart';
import '../utils/app_colors.dart';
import 'profile_controller.dart';
import 'dashboard_controller.dart';
import '../services/realtime_service.dart';
import 'dart:async';

class AlertsController extends GetxController {
  final FirestoreService _firestoreService = Get.find<FirestoreService>();
  final NotificationService _notificationService =
      Get.find<NotificationService>();
  final ProfileController _profileController = Get.find<ProfileController>();
  final DashboardController _dashboardController =
      Get.find<DashboardController>();
  final RealtimeService _realtimeService = Get.find<RealtimeService>();

  final historyList = <Map<String, String>>[].obs;
  final historyPatientName = ''.obs;
  final Set<String> _notifiedAlertIds = {};

  StreamSubscription? _alertsSub;
  StreamSubscription? _allAlertsSub;

  void listenToPatientAlerts(String patientId) {
    _alertsSub?.cancel();
    _alertsSub = _firestoreService.streamPatientAlerts(patientId).listen((
      alerts,
    ) {
      if (alerts.isNotEmpty) {
        final last = alerts.first;
        _dashboardController.hasActiveAlert.value = !last.is_handled;
        _dashboardController.seizureTime.value = DateFormat(
          'hh:mm a',
        ).format(last.time);
        _dashboardController.seizureStatusKey.value = last.is_handled
            ? 'Handled'
            : 'requires_review';

        for (var alert in alerts.take(10)) {
          _checkAndNotify(alert);
        }
      } else {
        _dashboardController.hasActiveAlert.value = false;
        _dashboardController.seizureTime.value = 'Unknown';
      }
      _updateHistoryList(alerts, patientId);
    });
  }

  void listenToAllAssignedAlerts(List<String> patientIds) {
    if (patientIds.isEmpty) return;
    _allAlertsSub?.cancel();
    _allAlertsSub = _firestoreService.streamAllAlerts().listen((alerts) {
      final filtered = alerts
          .where((a) => patientIds.contains(a.patientId))
          .toList();
      for (var alert in filtered.take(10)) {
        _checkAndNotify(alert);
      }
      _updateHistoryList(filtered, null);
    });
  }

  void _updateHistoryList(List<AlertModel> alerts, String? patientId) {
    historyList.value = alerts
        .map(
          (AlertModel a) => {
            'id': a.id,
            'patientId': a.patientId ?? patientId ?? '',
            'patientName':
                _profileController.patientNames[a.patientId ?? patientId] ?? '',
            'time': DateFormat('hh:mm a').format(a.time),
            'hr': a.heartRate.toString(),
            'hrv': a.hrv.toString(),
            'medication': a.medication,
            'symptoms': a.symptoms,
            'sleep': a.sleep,
            'stress': a.stress,
            'status': a.is_handled ? 'Handled'.tr : 'requires_review'.tr,
            'is_handled': a.is_handled.toString(),
          },
        )
        .toList();
  }

  void _checkAndNotify(AlertModel alert) {
    if (!alert.is_handled && !_notifiedAlertIds.contains(alert.id)) {
      if (DateTime.now().difference(alert.time).inMinutes < 10) {
        final pName =
            _profileController.patientNames[alert.patientId] ?? 'Patient';
        _notificationService.showNotification(
          title: 'seizure_alert_for'.trParams({'name': pName}),
          body: 'seek_help'.tr,
        );
      }
      _notifiedAlertIds.add(alert.id);
    }
  }

  Future<void> toggleAlertHandled(
    String patientId,
    String alertId,
    bool currentStatus,
  ) async {
    if (!currentStatus) {
      Get.defaultDialog(
        title: 'confirm'.tr,
        middleText: 'handle_seizure_confirmation'.tr,
        textCancel: 'cancel'.tr,
        textConfirm: 'confirm'.tr,
        confirmTextColor: Colors.white,
        buttonColor: AppColors.primaryGradientEnd,
        onConfirm: () async {
          Get.back();
          await _firestoreService.updateAlertHandled(patientId, alertId, true);
          // Reset RTDB alert status
          await _realtimeService.setAlertStatus(patientId, false);
        },
      );
    } else {
      await _firestoreService.updateAlertHandled(patientId, alertId, false);
    }
  }
}
