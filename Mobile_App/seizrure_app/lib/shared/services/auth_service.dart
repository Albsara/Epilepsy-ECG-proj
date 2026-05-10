import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/material.dart';
import 'package:get/get.dart';

class AuthService extends GetxService {
  final FirebaseAuth _auth = FirebaseAuth.instance;
  
  // For Admin to create users without logging out current session
  // Usually this is done via Firebase Admin SDK or Cloud Functions, 
  // but if the spec suggests a secondary instance, we can potentially 
  // use a separate FirebaseApp instance if configured, 
  // or more commonly, a Cloud Function.
  // However, I'll implement standard login/logout first.

  User? get currentUser => _auth.currentUser;
  Stream<User?> get authStateChanges => _auth.authStateChanges();

  Future<UserCredential?> login(String email, String password) async {
    try {
      return await _auth.signInWithEmailAndPassword(email: email, password: password);
    } on FirebaseAuthException catch (e) {
      Get.defaultDialog(
        title: 'Login Error', // Maybe keep as is or map to key
        middleText: e.message ?? 'Unknown error occurred',
        textConfirm: 'ok'.tr,
        confirmTextColor: Colors.white,
        onConfirm: () => Get.back(),
      );
      return null;
    }
  }

  Future<void> logout() async {
    await _auth.signOut();
  }

  Future<void> resetPassword(String email) async {
    try {
      await _auth.sendPasswordResetEmail(email: email);
      Get.defaultDialog(
        title: 'success'.tr,
        middleText: 'reset_link_sent'.tr,
        textConfirm: 'ok'.tr,
        confirmTextColor: Colors.white,
        onConfirm: () => Get.back(),
      );
    } catch (e) {
      Get.defaultDialog(
        title: 'error'.tr,
        middleText: e.toString(),
        textConfirm: 'ok'.tr,
        confirmTextColor: Colors.white,
        onConfirm: () => Get.back(),
      );
    }
  }
}
