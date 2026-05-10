import 'package:get/get.dart';
import '../services/firestore_service.dart';
import '../models/patient_model.dart';
import '../models/caregiver_model.dart';
import '../services/realtime_service.dart';
import '../services/notification_service.dart';
import 'dashboard_controller.dart';
import 'alerts_controller.dart';
import 'dart:async';
import 'package:get_storage/get_storage.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:flutter/material.dart';
import 'auth_controller.dart';

class ProfileController extends GetxController {
  final FirestoreService _firestoreService = Get.find<FirestoreService>();
  final NotificationService _notificationService = Get.find<NotificationService>();
  final AuthController _authController = Get.find<AuthController>();

  // User Profiles
  final userName = ''.obs;
  final userEmail = ''.obs;
  final userPhone = ''.obs;
  final userBirthDate = ''.obs;
  final userGender = 'Male'.obs;

  // Roles & Relations
  final isPatient = true.obs;
  final assignedCaregivers = <CaregiverModel>[].obs;
  final assignedPatients = <AssignedPatientData>[].obs;
  final patientNames = <String, String>{}.obs;

  // primary caregiver info (backward compatibility)
  final caregiverName = ''.obs;
  final caregiverPhone = ''.obs;

  void loadProfile(String uid) {
    // Patient Stream
    _firestoreService.streamPatient(uid).listen((patient) async {
      if (patient != null) {
        isPatient.value = true;
        userName.value = patient.name;
        userEmail.value = patient.email;
        userPhone.value = patient.phone;
        userBirthDate.value = patient.birthdate != null
            ? "${patient.birthdate!.year}-${patient.birthdate!.month.toString().padLeft(2, '0')}-${patient.birthdate!.day.toString().padLeft(2, '0')}"
            : '';
        userGender.value = patient.details.isNotEmpty ? patient.details : 'Male';

        _loadAssignedCaregivers(patient.caregiverIds);
        
        // Start patient's own live data and alerts
        final dashboardController = Get.find<DashboardController>();
        final alertsController = Get.find<AlertsController>();
        dashboardController.listenToLiveData(uid);
        alertsController.listenToPatientAlerts(uid);
        
        final box = GetStorage();
        box.write('user_id', uid);
        box.write('user_role', 'patient');
        _updateFCMToken(uid, true);
      }
    });

    // Caregiver Stream
    _firestoreService.streamCaregiver(uid).listen((caregiver) {
      if (caregiver != null) {
        isPatient.value = false;
        userName.value = caregiver.name;
        userEmail.value = caregiver.email;
        userPhone.value = caregiver.phone;

        final box = GetStorage();
        box.write('user_id', uid);
        box.write('user_role', 'caregiver');
        box.write('assigned_patient_ids', caregiver.patientIds);
        _updateFCMToken(uid, false);

        _loadAssignedPatients(caregiver.patientIds);
      }
    });
  }

  @override
  void onClose() {
    _clearAssignedPatients();
    super.onClose();
  }

  void _loadAssignedCaregivers(List<String> cgIds) {
    assignedCaregivers.clear();
    for (var cgId in cgIds) {
      _firestoreService.streamCaregiver(cgId).listen((cg) {
        if (cg != null) {
          final index = assignedCaregivers.indexWhere((e) => e.authUid == cgId);
          if (index != -1) {
            assignedCaregivers[index] = cg;
          } else {
            assignedCaregivers.add(cg);
          }
          if (assignedCaregivers.isNotEmpty) {
            caregiverName.value = assignedCaregivers.first.name;
            caregiverPhone.value = assignedCaregivers.first.phone;
          }
        }
      });
    }
  }

  void _loadAssignedPatients(List<String> pIds) {
    _clearAssignedPatients();
    final realtimeService = Get.find<RealtimeService>();
    final alertsController = Get.find<AlertsController>();
    
    // Start listening to all alerts for these patients globally for notifications and the home list
    alertsController.listenToAllAssignedAlerts(pIds);

    for (var pId in pIds) {
      _firestoreService.streamPatient(pId).listen((p) {
        if (p != null) {
          patientNames[pId] = p.name;
          final existingIndex = assignedPatients.indexWhere((e) => e.patient.authUid == pId);
          if (existingIndex == -1) {
            final pData = AssignedPatientData(patient: p);
            
            // Start live data stream for this specific patient in the list
            pData.liveSub = realtimeService.streamLiveData(pId).listen((data) {
              if (data != null) {
                pData.hr.value = data.hr;
                pData.hrv.value = data.hrv;
                pData.isDanger.value = data.alert;
              }
            });
            
            assignedPatients.add(pData);
          } else {
            // Update existing patient data if needed
            // assignedPatients[existingIndex] = ...
          }
        }
      });
    }
  }

  void _clearAssignedPatients() {
    for (var p in assignedPatients) {
      p.cancel();
    }
    assignedPatients.clear();
    patientNames.clear();
  }

  Future<void> _updateFCMToken(String uid, bool isPat) async {
    final token = await _notificationService.getFCMToken();
    if (token != null) {
      if (isPat) {
        await _firestoreService.updatePatient(uid, {'fcmToken': token});
      } else {
        await _firestoreService.updateCaregiver(uid, {'fcmToken': token});
      }
    }
  }

  Future<void> saveProfile({
    required String name,
    required String phone,
    String? birthDate,
    String? gender,
  }) async {
    final user = _authController.firebaseUser.value;
    if (user == null) return;
    try {
      if (isPatient.value) {
        DateTime? parsedDate = birthDate != null ? DateTime.tryParse(birthDate) : null;
        await _firestoreService.updatePatient(user.uid, {
          'name': name,
          'phone': phone,
          if (parsedDate != null) 'birthdate': Timestamp.fromDate(parsedDate),
          if (gender != null) 'details': gender,
        });
      } else {
        await _firestoreService.updateCaregiver(user.uid, {'name': name, 'phone': phone});
      }
      Get.snackbar('success'.tr, 'profile_updated'.tr, backgroundColor: Colors.green, colorText: Colors.white);
    } catch (e) {
      Get.snackbar('error'.tr, e.toString(), backgroundColor: Colors.red, colorText: Colors.white);
    }
  }

  Future<void> linkPatient(String patientId) async {
    final caregiverId = _authController.firebaseUser.value?.uid;
    if (caregiverId == null) return;
    try {
      final patient = await _firestoreService.getPatient(patientId);
      if (patient == null) { Get.snackbar('Error', 'Patient not found'); return; }
      await _firestoreService.linkCaregiverToPatient(patientId, caregiverId);
      Get.snackbar('Success', 'Patient linked successfully');
    } catch (e) { Get.snackbar('Error', 'Failed to link patient: $e'); }
  }

  Future<void> unlinkPatient(String patientId) async {
    final caregiverId = _authController.firebaseUser.value?.uid;
    if (caregiverId == null) return;
    try {
      await _firestoreService.unlinkCaregiverFromPatient(patientId, caregiverId);
      Get.snackbar('Success', 'Patient unlinked successfully');
    } catch (e) { Get.snackbar('Error', 'Failed to unlink patient: $e'); }
  }
}

class AssignedPatientData {
  final PatientModel patient;
  final RxInt hr = 0.obs;
  final RxInt hrv = 0.obs;
  final RxBool isDanger = false.obs;
  StreamSubscription? liveSub;
  StreamSubscription? alertsSub;
  AssignedPatientData({required this.patient});
  void cancel() {
    liveSub?.cancel();
    alertsSub?.cancel();
  }
}
