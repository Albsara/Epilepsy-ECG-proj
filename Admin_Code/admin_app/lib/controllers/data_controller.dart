import 'dart:async';
import 'package:get/get.dart';
import '../models/models.dart';
import '../services/firestore_service.dart';
import '../services/rtdb_service.dart';
import '../widgets/app_dialogs.dart';

class DataController extends GetxController {
  final FirestoreService _fs = FirestoreService();
  final RtdbService _rtdb = RtdbService();

  final RxList<Patient> patients = <Patient>[].obs;

  StreamSubscription? _patientsSub;

  // Per-patient subs for RTDB + subcollections
  final Map<String, StreamSubscription> _liveSubs = {};
  final Map<String, StreamSubscription> _caregiverSubs = {};
  final Map<String, StreamSubscription> _alertSubs = {};

  final RxList<Caregiver> allCaregivers = <Caregiver>[].obs;
  StreamSubscription? _allCaregiversSub;

  final RxList<SeizureAlert> globalAlerts = <SeizureAlert>[].obs;
  StreamSubscription? _globalAlertsSub;

  @override
  void onInit() {
    super.onInit();

    _allCaregiversSub = _fs.streamAllCaregivers().listen((list) {
      allCaregivers.assignAll(list);
    });

    _globalAlertsSub = _fs.streamAllAlertsGlobally().listen((list) {
      globalAlerts.assignAll(list);
    });

    _patientsSub = _fs.streamPatients().listen((list) {
      // Keep old instance if exists (so UI Rx references remain stable)
      final Map<String, Patient> old = {for (final p in patients) p.id: p};

      final merged = <Patient>[];
      for (final fresh in list) {
        final existing = old[fresh.id];
        if (existing != null) {
          existing.email = fresh.email;
          existing.name = fresh.name;
          existing.birthdate = fresh.birthdate;
          existing.phone = fresh.phone;
          existing.details = fresh.details;
          merged.add(existing);
        } else {
          merged.add(fresh);
        }
      }

      patients.assignAll(merged);

      // Ensure live streams attached
      for (final p in patients) {
        _ensureLiveStream(p);
      }

      // Cleanup removed
      final currentIds = patients.map((e) => e.id).toSet();
      final toRemove = _liveSubs.keys
          .where((id) => !currentIds.contains(id))
          .toList();
      for (final id in toRemove) {
        _liveSubs[id]?.cancel();
        _liveSubs.remove(id);
        _caregiverSubs[id]?.cancel();
        _caregiverSubs.remove(id);
        _alertSubs[id]?.cancel();
        _alertSubs.remove(id);
      }
    });
  }

  @override
  void onClose() {
    _patientsSub?.cancel();
    _allCaregiversSub?.cancel();
    _globalAlertsSub?.cancel();
    for (final s in _liveSubs.values) {
      s.cancel();
    }
    for (final s in _caregiverSubs.values) {
      s.cancel();
    }
    for (final s in _alertSubs.values) {
      s.cancel();
    }
    super.onClose();
  }

  Patient? byId(String id) {
    try {
      return patients.firstWhere((p) => p.id == id);
    } catch (_) {
      return null;
    }
  }

  void _ensureLiveStream(Patient p) {
    if (_liveSubs.containsKey(p.id)) return;
    _liveSubs[p.id] = _rtdb
        .streamLiveStatus(p.id, fallback: p.liveStatus.value)
        .listen((live) {
          p.liveStatus.value = live;
        });
  }

  // Call these when opening patient detail (so you stream caregivers/alerts only when needed)
  void attachDetailStreams(String patientId) {
    final p = byId(patientId);
    if (p == null) return;

    if (!_caregiverSubs.containsKey(patientId)) {
      _caregiverSubs[patientId] = _fs.streamCaregiversForPatient(patientId).listen((
        list,
      ) {
        p.caregivers.assignAll(list);
      });
    }

    if (!_alertSubs.containsKey(patientId)) {
      _alertSubs[patientId] = _fs.streamAlerts(patientId).listen((list) {
        p.alerts.assignAll(list);
      });
    }
  }

  // --- Patients (Firestore) ---
  Future<void> addPatient({
    required String email,
    required String name,
    required DateTime birthdate,
    required String phone,
    required String details,
  }) async {
    await _fs.addPatient(
      email: email,
      name: name,
      birthdate: birthdate,
      phone: phone,
      details: details,
    );
    AppDialogs.success(message: 'patient_created'.tr);
  }

  Future<void> updatePatient({
    required String id,
    required String email,
    required String name,
    required DateTime birthdate,
    required String phone,
    required String details,
  }) async {
    await _fs.updatePatient(
      id: id,
      email: email,
      name: name,
      birthdate: birthdate,
      phone: phone,
      details: details,
    );
    AppDialogs.success(message: 'patient_updated'.tr);
  }

  Future<void> deletePatient(String id) async {
    await _fs.deletePatient(id);
    AppDialogs.success(message: 'patient_removed'.tr);
  }

  // --- Caregivers ---
  Future<void> addCaregiver({
    required String name,
    required String email,
    required String phone,
    String? assignToPatientId,
  }) async {
    final newId = await _fs.addCaregiver(
      name: name,
      email: email,
      phone: phone,
    );
    if (assignToPatientId != null) {
      await _fs.assignCaregiver(patientId: assignToPatientId, caregiverId: newId);
    }
    AppDialogs.success(message: 'caregiver_created'.tr);
  }

  Future<void> updateCaregiver({
    required String caregiverId,
    required String name,
    required String email,
    required String phone,
  }) async {
    await _fs.updateCaregiver(
      caregiverId: caregiverId,
      name: name,
      email: email,
      phone: phone,
    );
    AppDialogs.success(message: 'caregiver_updated'.tr);
  }

  Future<void> deleteCaregiver(String caregiverId) async {
    await _fs.deleteCaregiver(caregiverId);
    AppDialogs.success(message: 'caregiver_removed'.tr);
  }

  Future<void> assignCaregiver(String patientId, String caregiverId) async {
    await _fs.assignCaregiver(patientId: patientId, caregiverId: caregiverId);
    AppDialogs.success(message: 'assigned_caregiver'.tr ?? 'Caregiver Assigned');
  }

  Future<void> unassignCaregiver(String patientId, String caregiverId) async {
    await _fs.unassignCaregiver(patientId: patientId, caregiverId: caregiverId);
  }

  // --- Alerts (Firestore subcollection) ---
  Future<void> addAlert({
    required String patientId,
    required DateTime time,
    required int heartRate,
    required int hrv,
  }) async {
    await _fs.addAlert(
      patientId: patientId,
      time: time,
      heartRate: heartRate,
      hrv: hrv,
    );
    AppDialogs.success(message: 'Alert added');
  }

  Future<void> handleAlert(SeizureAlert alert) async {
    await _fs.handleAlert(patientId: alert.patientId, alertId: alert.id);
  }

  // --- Dashboard helpers ---
  int caregiversCount() {
    return allCaregivers.length;
  }

  List<SeizureAlert> allAlerts() {
    final list = <SeizureAlert>[];
    for (final p in patients) {
      list.addAll(p.alerts);
    }
    list.sort((a, b) => b.time.compareTo(a.time));
    return list;
  }

  int alertsTodayCount(
    bool Function(DateTime, DateTime) isSameDay,
    DateTime now,
  ) {
    return allAlerts().where((a) => isSameDay(a.time, now)).length;
  }
}
