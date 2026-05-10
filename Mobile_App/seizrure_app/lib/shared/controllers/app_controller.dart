import 'package:flutter/material.dart';
import 'package:get/get.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:seizrure_app/shared/routes/app_routes.dart';
import '../services/auth_service.dart';
import '../services/firestore_service.dart';
import '../services/realtime_service.dart';
import '../services/notification_service.dart';
import '../models/alert_model.dart';
import '../models/caregiver_model.dart';
import '../models/patient_model.dart';
import 'auth_controller.dart';
import 'profile_controller.dart';
import 'dashboard_controller.dart';
import 'alerts_controller.dart';
import 'package:url_launcher/url_launcher.dart';

export 'profile_controller.dart';
export 'dashboard_controller.dart';
export 'alerts_controller.dart';
export '../models/caregiver_model.dart';
export '../models/patient_model.dart';
export '../models/alert_model.dart';

class AppBinding extends Bindings {
  @override
  void dependencies() {
    // Services
    Get.put(AuthService(), permanent: true);
    Get.put(FirestoreService(), permanent: true);
    Get.put(RealtimeService(), permanent: true);
    Get.put(NotificationService(), permanent: true);

    // Controllers
    Get.put(AuthController(), permanent: true);
    Get.put(ProfileController(), permanent: true);
    Get.put(DashboardController(), permanent: true);
    Get.put(AlertsController(), permanent: true);
    Get.put(AppController(), permanent: true);
  }
}

class AppController extends GetxController {
  final AuthController _authController = Get.find<AuthController>();
  final ProfileController profile = Get.find<ProfileController>();
  final DashboardController dashboard = Get.find<DashboardController>();
  final AlertsController alerts = Get.find<AlertsController>();

  // Global UI State
  final navIndex = 0.obs;

  // Shortcuts for UI (mapping to sub-controllers for compatibility)
  RxString get userName => profile.userName;
  RxString get userEmail => profile.userEmail;
  RxString get userPhone => profile.userPhone;
  RxString get userBirthDate => profile.userBirthDate;
  RxString get userGender => profile.userGender;

  RxInt get bpm => dashboard.bpm;
  RxInt get hrv => dashboard.hrv;
  RxBool get hasActiveAlert => dashboard.hasActiveAlert;
  RxString get seizureTime => dashboard.seizureTime;
  RxString get seizureStatusKey => dashboard.seizureStatusKey;

  RxBool get isPatient => profile.isPatient;
  RxList<CaregiverModel> get assignedCaregivers => profile.assignedCaregivers;
  RxList<AssignedPatientData> get assignedPatients => profile.assignedPatients;
  RxMap<String, String> get patientNames => profile.patientNames;
  RxString get caregiverName => profile.caregiverName;
  RxString get caregiverPhone => profile.caregiverPhone;

  RxList<Map<String, String>> get historyList => alerts.historyList;
  RxString get historyPatientName => alerts.historyPatientName;

  // Additional vitals for dashboard
  RxString get medication => dashboard.medication;
  RxString get symptoms => dashboard.symptoms;
  RxString get sleep => dashboard.sleep;
  RxString get stress => dashboard.stress;
  RxString get lastUpdated => dashboard.lastUpdated;
  RxBool get isDataActive => dashboard.isDataActive;
  RxString get statusKey => RxString('normal');

  final currentViewedPatientId = ''.obs;

  @override
  void onInit() {
    super.onInit();
    // Listen to current user changes to update profiles
    ever(_authController.firebaseUser, _onUserChanged);
  }

  void _onUserChanged(User? user) {
    if (user != null) {
      profile.loadProfile(user.uid);
    }
  }

  Future<void> refreshData() async {
    final user = _authController.firebaseUser.value;
    if (user != null) {
      profile.loadProfile(user.uid);
    }
    await Future.delayed(const Duration(milliseconds: 800));
  }

  // Navigation & Utility Methods
  void goNav(int index) => navIndex.value = index;

  void toggleLang() {
    final isEn = Get.locale?.languageCode == 'en';
    Get.updateLocale(Locale(isEn ? 'ar' : 'en'));
  }

  void loadPatientHistory(String patientId) {
    alerts.historyList.clear();
    alerts.historyPatientName.value =
        profile.patientNames[patientId] ?? 'Patient';
    dashboard.listenToLiveData(patientId);
    alerts.listenToPatientAlerts(patientId);
    Get.toNamed(Routes.patientDetails);
  }

  void viewAllAlerts(List<String> patientIds) {
    alerts.historyList.clear();
    alerts.listenToAllAssignedAlerts(patientIds);
    Get.toNamed(Routes.alertHistory);
  }

  Future<void> makePhoneCall(String phoneNumber) async {
    final sanitized = phoneNumber.replaceAll(RegExp(r'\s+'), '');
    final Uri launchUri = Uri.parse('tel:$sanitized');
    if (await canLaunchUrl(launchUri)) {
      await launchUrl(launchUri, mode: LaunchMode.externalApplication);
    }
  }

  Future<void> sendSms(String phoneNumber) async {
    final sanitized = phoneNumber.replaceAll(RegExp(r'\s+'), '');
    final Uri launchUri = Uri.parse('sms:$sanitized');
    if (await canLaunchUrl(launchUri)) {
      await launchUrl(launchUri, mode: LaunchMode.externalApplication);
    }
  }

  Future<void> saveProfile({
    required String name,
    required String phone,
    String? birthDate,
    String? gender,
  }) => profile.saveProfile(
    name: name,
    phone: phone,
    birthDate: birthDate,
    gender: gender,
  );

  Future<void> linkPatient(String patientId) => profile.linkPatient(patientId);
  Future<void> unlinkPatient(String patientId) =>
      profile.unlinkPatient(patientId);

  Future<void> toggleAlertHandled(
    String patientId,
    String alertId,
    bool currentStatus,
  ) => alerts.toggleAlertHandled(patientId, alertId, currentStatus);
}
