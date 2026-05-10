import 'package:firebase_auth/firebase_auth.dart';
import 'package:get/get.dart';

import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';

class AuthService {
  final FirebaseAuth _auth = FirebaseAuth.instance;
  final FirebaseFirestore _db = FirebaseFirestore.instance;

  Stream<User?> authChanges() => _auth.authStateChanges();

  User? get currentUser => _auth.currentUser;

  /// Sign in ONLY if this email exists in Firestore collection: `admins`

  /// Throws FirebaseAuthException(code: 'not-admin') if not found.
  Future<void> signIn(String email, String password) async {
    final cleanEmail = email.trim().toLowerCase();
    final cleanPass = password.trim();

    if (cleanEmail.isEmpty || cleanPass.isEmpty) {
      throw FirebaseAuthException(
        code: 'missing-credentials',
        message: 'email_password_required'.tr,
      );
    }

    // 1) Check if email is allowed (exists in admins collection)
    final isAdmin = await _isEmailInAdmins(cleanEmail);
    if (!isAdmin) {
      throw FirebaseAuthException(
        code: 'not-admin',
        message: 'admin_access_denied'.tr,
      );
    }

    // 2) If allowed, proceed with Firebase Auth sign in
    await _auth.signInWithEmailAndPassword(
      email: cleanEmail,
      password: cleanPass,
    );
  }

  Future<bool> _isEmailInAdmins(String emailLower) async {
    // Query by email (recommended)
    final q = await _db
        .collection('admins')
        .where('email', isEqualTo: emailLower)
        .limit(1)
        .get();

    if (q.docs.isNotEmpty) return true;

    // Fallback: in case admins documents store email not lowercased
    final q2 = await _db
        .collection('admins')
        .where('email', isEqualTo: emailLower.toUpperCase())
        .limit(1)
        .get();

    return q2.docs.isNotEmpty;
  }

  Future<void> signOut() async {
    await _auth.signOut();
  }

  /// Only send reset if user email exists in `admins` too (optional but safer)
  Future<void> sendReset(String email) async {
    final cleanEmail = email.trim().toLowerCase();
    if (cleanEmail.isEmpty) return;

    final isAdmin = await _isEmailInAdmins(cleanEmail);
    if (!isAdmin) {
      throw FirebaseAuthException(
        code: 'not-admin',
        message: 'email_not_admin'.tr,
      );
    }

    await _auth.sendPasswordResetEmail(email: cleanEmail);
  }

  Future<void> updatePassword(String newPassword) async {
    final user = _auth.currentUser;
    if (user != null) {
      await user.updatePassword(newPassword);
    } else {
      throw FirebaseAuthException(
        code: 'no-user',
        message: 'No user signed in',
      );
    }
  }
}
