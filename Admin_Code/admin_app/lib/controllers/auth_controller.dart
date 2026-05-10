import 'dart:async';
import 'package:get/get.dart';
import '../routes.dart';
import '../services/auth_service.dart';
import '../widgets/app_dialogs.dart';

class AuthController extends GetxController {
  final AuthService _auth = AuthService();

  final RxBool isLoggedIn = false.obs;
  final RxString adminName = 'Admin'.obs;
  final RxString adminEmail = ''.obs;
  final RxString adminPhone = '+000 000 0000'.obs;

  StreamSubscription? _sub;

  @override
  void onInit() {
    super.onInit();
    _sub = _auth.authChanges().listen((user) {
      isLoggedIn.value = user != null;
      adminEmail.value = user?.email ?? '';
      if (user == null) {
        Get.offAllNamed(Routes.login);
      } else {
        Get.offAllNamed(Routes.dashboard);
      }
    });
  }

  @override
  void onClose() {
    _sub?.cancel();
    super.onClose();
  }

  Future<void> login(String email, String password) async {
    try {
      await _auth.signIn(email, password);
    } catch (e) {
      AppDialogs.error(message: e.toString());
    }
  }

  Future<void> logout() async {
    await _auth.signOut();
  }

  Future<void> sendResetLink(String email) async {
    if (email.trim().isEmpty) {
      AppDialogs.warning(
        title: 'missing_fields'.tr,
        message: 'Enter your email to reset password',
      );
      return;
    }
    try {
      await _auth.sendReset(email);
      AppDialogs.success(
        title: 'reset_link_sent'.tr,
        message: 'Check your email inbox',
      );
    } catch (e) {
      AppDialogs.error(message: e.toString());
    }
  }

  void updateProfile({
    required String name,
    required String email,
    required String phone,
  }) {
    // This is local UI info. If you want, store it in Firestore as admin profile.
    adminName.value = name.trim().isEmpty ? adminName.value : name.trim();
    adminPhone.value = phone.trim().isEmpty ? adminPhone.value : phone.trim();

    AppDialogs.success(message: 'personal_info_updated'.tr);
  }

  Future<void> changePassword(String newPassword) async {
    if (newPassword.trim().length < 6) {
      AppDialogs.warning(
        title: 'invalid_password'.tr,
        message: 'password_min_length'.tr ?? 'Password must be at least 6 characters',
      );
      return;
    }
    try {
      await _auth.updatePassword(newPassword.trim());
      AppDialogs.success(
        title: 'success'.tr,
        message: 'password_updated_success'.tr ?? 'Password updated successfully',
      );
    } catch (e) {
      AppDialogs.error(message: e.toString());
    }
  }
}
