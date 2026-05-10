import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:get/get.dart';
import 'package:get_storage/get_storage.dart';
import '../../models/caregiver_model.dart';
import '../../models/patient_model.dart';
import '../auth_controller.dart';
import '../../services/firestore_service.dart';
import '../../services/notification_service.dart';

mixin ProfileMixin on GetxController {
  final FirestoreService firestoreService = Get.find<FirestoreService>();
  final NotificationService notificationService =
      Get.find<NotificationService>();
  final AuthController authController = Get.find<AuthController>();

  // Role flag
  final isPatient = true.obs;

  // User Profiles
  final userName = ''.obs;
  final userEmail = ''.obs;
  final userPhone = ''.obs;
  final userBirthDate = ''.obs;
  final userGender = 'Male'.obs;

  // Caregiver Info
  final caregiverName = ''.obs;
  final caregiverPhone = ''.obs;
  final assignedCaregivers = <CaregiverModel>[].obs;

  final patientNames = <String, String>{}.obs;

  void updateProfileData(PatientModel patient) {
    isPatient.value = true;
    userName.value = patient.name;
    userEmail.value = patient.email;
    userPhone.value = patient.phone;
    userBirthDate.value = patient.birthdate != null
        ? "${patient.birthdate!.year}-${patient.birthdate!.month.toString().padLeft(2, '0')}-${patient.birthdate!.day.toString().padLeft(2, '0')}"
        : '';
    userGender.value = patient.details.isNotEmpty ? patient.details : 'Male';
  }
}
