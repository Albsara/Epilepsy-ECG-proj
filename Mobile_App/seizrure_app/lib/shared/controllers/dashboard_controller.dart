import 'package:get/get.dart';
import '../services/realtime_service.dart';
import '../services/firestore_service.dart';
import '../models/live_data_model.dart';
import '../models/alert_model.dart';
import '../utils/ai_evaluator.dart';
import 'package:flutter/services.dart';
import 'dart:async';
import 'profile_controller.dart';

class DashboardController extends GetxController {
  final RealtimeService _realtimeService = Get.find<RealtimeService>();
  final FirestoreService _firestoreService = Get.find<FirestoreService>();
  final ProfileController _profileController = Get.find<ProfileController>();

  // Dashboard state
  final bpm = 0.obs;
  final hrv = 0.obs;
  final medication = 'yes'.obs;
  final symptoms = ''.obs;
  final sleep = 'good'.obs;
  final stress = 'low'.obs;

  final lastUpdated = 'Never'.obs;
  final isDataActive = false.obs;
  final lastBpmChangeTime = DateTime.now().obs;

  // Seizure State
  final hasActiveAlert = false.obs;
  final seizureTime = 'Unknown'.obs;
  final seizureStatusKey = 'just'.obs;

  final AiEvaluator _evaluator = AiEvaluator();
  final Map<String, DateTime> _lastAlertPushed = {};
  StreamSubscription? _liveSub;

  @override
  void onInit() {
    super.onInit();
    _loadModel();
  }

  Future<void> _loadModel() async {
    try {
      final String response = await rootBundle.loadString('assets/model.json');
      _evaluator.loadModel(response);
    } catch (e) {
      print('Error loading AI model: $e');
    }
  }

  void listenToLiveData(String patientId) {
    _liveSub?.cancel();
    _liveSub = _realtimeService.streamLiveData(patientId).listen((data) {
      if (data != null) {
        _updateVitals(data);
        _evaluateDanger(data, patientId);
      }
    });
  }

  void _updateVitals(LiveDataModel data) {
    if (data.hr != bpm.value) {
      lastBpmChangeTime.value = DateTime.now();
      isDataActive.value = true;
    } else if (DateTime.now().difference(lastBpmChangeTime.value).inSeconds >
        2) {
      isDataActive.value = false;
    }

    bpm.value = data.hr;
    hrv.value = data.hrv;
    medication.value = data.medication;
    symptoms.value = data.symptoms;
    sleep.value = data.sleep;
    stress.value = data.stress;
    lastUpdated.value = 'Just now';
  }

  Future<void> _evaluateDanger(LiveDataModel data, String pId) async {
    if (!_evaluator.isReady()) return;

    final isDanger = _evaluator.evaluate(data);
    if (isDanger && _profileController.isPatient.value) {
      final lastPushed = _lastAlertPushed[pId];
      final bool isWithinCoolingPeriod =
          lastPushed != null &&
          DateTime.now().difference(lastPushed).inMinutes < 5;

      if (!(isWithinCoolingPeriod && hasActiveAlert.value)) {
        _lastAlertPushed[pId] = DateTime.now();

        // 1. Set alert field to true in RTDB first
        await _realtimeService.setAlertStatus(pId, true);

        // 2. Add Firestore alert
        await _firestoreService.addSeizureAlert(
          pId,
          AlertModel(
            id: '',
            heartRate: data.hr,
            hrv: data.hrv,
            medication: data.medication,
            symptoms: data.symptoms,
            sleep: data.sleep,
            stress: data.stress,
            time: DateTime.now(),
            is_handled: false,
            createdAt: DateTime.now(),
          ),
        );
      }
    }
  }
}
