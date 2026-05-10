import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/material.dart';
import 'package:get/get.dart';
import '../services/auth_service.dart';
import '../services/firestore_service.dart';
import '../routes/app_routes.dart';

class AuthController extends GetxController {
  final AuthService _authService = Get.find<AuthService>();
  final FirestoreService _firestoreService = Get.find<FirestoreService>();

  final Rxn<User> firebaseUser = Rxn<User>();

  @override
  void onInit() {
    super.onInit();
    firebaseUser.bindStream(_authService.authStateChanges);

    // Auto redirect logic
    ever(firebaseUser, _handleAuthChanged);
  }

  void _handleAuthChanged(User? user) async {
    if (user == null) {
      Get.offAllNamed(Routes.login);
    } else {
      // Logic to determine if user is patient or caregiver
      // For now, we'll check if they exist in patients or caregivers collection
      final patient = await _firestoreService.getPatient(user.uid);
      if (patient != null) {
        Get.offAllNamed(Routes.dashboard);
      } else {
        final caregiver = await _firestoreService.getCaregiver(user.uid);
        if (caregiver != null) {
          Get.offAllNamed(Routes.caregiverHome);
        } else {
          // New user or unknown role
          Get.offAllNamed(Routes.login);
          Get.defaultDialog(
            title: 'error'.tr,
            middleText: 'user_profile_not_found'.tr,
            textConfirm: 'ok'.tr,
            confirmTextColor: Colors.white,
            onConfirm: () => Get.back(),
          );
        }
      }
    }
  }

  Future<void> login(String email, String password) async {
    await _authService.login(email, password);
  }

  Future<void> logout() async {
    await _authService.logout();
  }
}
