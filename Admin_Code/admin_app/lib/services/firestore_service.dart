import 'dart:async';

import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:firebase_core/firebase_core.dart';
import 'package:firebase_database/firebase_database.dart';
import 'package:get/get.dart';

import '../models/models.dart';

class FirestoreService {
  FirestoreService({FirebaseFirestore? firestore, FirebaseAuth? primaryAuth})
    : _db = firestore ?? FirebaseFirestore.instance,
      _primaryAuth = primaryAuth ?? FirebaseAuth.instance;

  final FirebaseFirestore _db;
  final FirebaseAuth _primaryAuth;

  // Use a secondary FirebaseApp + FirebaseAuth so creating users
  // does NOT sign out your admin session.
  FirebaseApp? _secondaryApp;
  FirebaseAuth? _secondaryAuth;

  CollectionReference<Map<String, dynamic>> get patientsCol =>
      _db.collection('patients');

  // =========================
  // INTERNAL: Secondary Auth
  // =========================
  Future<FirebaseAuth> _getSecondaryAuth() async {
    if (_secondaryAuth != null) return _secondaryAuth!;

    // reuse the same options as the default app
    final defaultApp = Firebase.app();
    final name = 'secondary-auth-${DateTime.now().millisecondsSinceEpoch}';

    _secondaryApp = await Firebase.initializeApp(
      name: name,
      options: defaultApp.options,
    );

    _secondaryAuth = FirebaseAuth.instanceFor(app: _secondaryApp!);
    return _secondaryAuth!;
  }

  /// Creates an Auth user (email/password) without affecting the current admin login.
  /// Returns created user's uid.
  Future<String> _createAuthUser({
    required String email,
    required String password,
  }) async {
    final auth = await _getSecondaryAuth();
    try {
      final cred = await auth.createUserWithEmailAndPassword(
        email: email.trim(),
        password: password,
      );
      final uid = cred.user?.uid;
      if (uid == null) {
        throw StateError('auth_user_creation_failed'.tr);
      }
      return uid;
    } on FirebaseAuthException catch (e) {
      // Common codes: email-already-in-use, invalid-email, weak-password, etc.
      throw FirebaseAuthException(code: e.code, message: e.message);
    }
  }

  Future<void> addMockAlertsForPatient({
    required String patientId,
    int count = 5,
  }) async {
    final alerts = alertsCol(patientId);

    final now = DateTime.now();

    // Create a batch so it’s fast
    final batch = FirebaseFirestore.instance.batch();

    for (int i = 0; i < count; i++) {
      final minutesAgo = 15 + (i * 60);
      final time = now.subtract(Duration(minutes: minutesAgo));

      final heartRate = 110 + (i % 25); // 110..134
      final hrv = 20 + (i % 30); // 20..49

      final docRef = alerts.doc(); // auto id
      batch.set(docRef, {
        'time': Timestamp.fromDate(time),
        'heartRate': heartRate,
        'hrv': hrv,
        'is_handled': i % 3 == 0,
        'createdAt': FieldValue.serverTimestamp(),
      });
    }

    await batch.commit();
  }

  // =========================
  // PATIENTS
  // =========================
  Stream<List<Patient>> streamPatients() {
    return patientsCol
        .orderBy('name')
        .snapshots()
        .map((snap) => snap.docs.map((d) => Patient.fromDoc(d)).toList());
  }

  /// Creates:
  /// 1) Firebase Auth user for patient (default password "123456")
  /// 2) Firestore patient document with docId == auth uid
  ///
  /// Returns patientId (uid).
  Future<String> addPatient({
    required String email,
    required String name,
    required DateTime birthdate,
    required String phone,
    required String details,
  }) async {
    final cleanEmail = email.trim();
    if (cleanEmail.isEmpty) {
      throw ArgumentError('patient_email_required'.tr);
    }

    // 1) Create Auth user first so we can use uid as patientId
    final uid = await _createAuthUser(email: cleanEmail, password: '123456');

    // 2) Create Firestore patient doc using uid as ID
    await patientsCol.doc(uid).set({
      'email': cleanEmail,
      'name': name.trim(),
      'birthdate': Timestamp.fromDate(birthdate),
      'phone': phone.trim(),
      'details': details.trim(),
      'authUid': uid,
      'caregiverIds': [],
      'createdAt': FieldValue.serverTimestamp(),
      'updatedAt': FieldValue.serverTimestamp(),
    });

    // 3) Initialize Realtime DB node (schema) for this patient
    //    live/{uid} => { hr, hrv, medication, symptoms, sleep, stress, updatedAt }
    await FirebaseDatabase.instance.ref('live/$uid').update({
      'hr': 0,
      'hrv': 0,
      'updatedAt': ServerValue.timestamp,
    });

    return uid;
  }

  Future<void> updatePatient({
    required String id,
    required String email,
    required String name,
    required DateTime birthdate,
    required String phone,
    required String details,
  }) async {
    // NOTE:
    // Updating the patient's FirebaseAuth email from the client is NOT supported
    // unless you sign in as that patient or use Admin SDK/Cloud Functions.
    // This updates Firestore only.
    await patientsCol.doc(id).update({
      'email': email.trim(),
      'name': name.trim(),
      'birthdate': Timestamp.fromDate(birthdate),
      'phone': phone.trim(),
      'details': details.trim(),
      'updatedAt': FieldValue.serverTimestamp(),
    });
  }

  Future<void> deletePatient(String id) async {
    // NOTE:
    // Deleting the patient's FirebaseAuth user from the client is NOT reliable
    // (you’d need to authenticate as that user or use Admin SDK).
    // This deletes Firestore doc only.
    //
    // In production, use a Cloud Function (Admin SDK) to:
    // - delete auth user
    // - cascade delete subcollections (caregivers/alerts)
    await patientsCol.doc(id).delete();
  }

  // =========================
  // CAREGIVERS
  // =========================
  CollectionReference<Map<String, dynamic>> get caregiversCol =>
      _db.collection('caregivers');

  Stream<List<Caregiver>> streamAllCaregivers() {
    return caregiversCol
        .orderBy('name')
        .snapshots()
        .map((snap) => snap.docs.map((d) => Caregiver.fromDoc(d)).toList());
  }

  Stream<List<Caregiver>> streamCaregiversForPatient(String patientId) {
    return caregiversCol
        .where('patientIds', arrayContains: patientId)
        .snapshots()
        .map((snap) {
           final list = snap.docs.map((d) => Caregiver.fromDoc(d)).toList();
           list.sort((a, b) => a.name.compareTo(b.name));
           return list;
        });
  }

  Future<String> addCaregiver({
    required String name,
    required String email,
    required String phone,
  }) async {
    final cleanEmail = email.trim();
    if (cleanEmail.isEmpty) {
      throw ArgumentError('caregiver_email_required'.tr);
    }

    final caregiverUid = await _createAuthUser(
      email: cleanEmail,
      password: '123456',
    );

    await caregiversCol.doc(caregiverUid).set({
      'name': name.trim(),
      'email': cleanEmail,
      'phone': phone.trim(),
      'authUid': caregiverUid,
      'patientIds': [],
      'createdAt': FieldValue.serverTimestamp(),
      'updatedAt': FieldValue.serverTimestamp(),
    });

    return caregiverUid;
  }

  Future<void> updateCaregiver({
    required String caregiverId,
    required String name,
    required String email,
    required String phone,
  }) async {
    await caregiversCol.doc(caregiverId).update({
      'name': name.trim(),
      'email': email.trim(),
      'phone': phone.trim(),
      'updatedAt': FieldValue.serverTimestamp(),
    });
  }

  Future<void> deleteCaregiver(String caregiverId) async {
    await caregiversCol.doc(caregiverId).delete();
  }

  // --- ASSIGNMENT ---
  Future<void> assignCaregiver({
    required String patientId,
    required String caregiverId,
  }) async {
    final batch = _db.batch();
    batch.update(patientsCol.doc(patientId), {
      'caregiverIds': FieldValue.arrayUnion([caregiverId])
    });
    batch.update(caregiversCol.doc(caregiverId), {
      'patientIds': FieldValue.arrayUnion([patientId])
    });
    await batch.commit();
  }

  Future<void> unassignCaregiver({
    required String patientId,
    required String caregiverId,
  }) async {
    final batch = _db.batch();
    batch.update(patientsCol.doc(patientId), {
      'caregiverIds': FieldValue.arrayRemove([caregiverId])
    });
    batch.update(caregiversCol.doc(caregiverId), {
      'patientIds': FieldValue.arrayRemove([patientId])
    });
    await batch.commit();
  }

  // =========================
  // ALERTS
  // =========================
  CollectionReference<Map<String, dynamic>> alertsCol(String patientId) =>
      patientsCol.doc(patientId).collection('alerts');

  Stream<List<SeizureAlert>> streamAllAlertsGlobally() {
    return _db.collectionGroup('alerts')
        .orderBy('time', descending: true)
        .snapshots()
        .map(
          (snap) => snap.docs.map((d) {
            final pId = d.reference.parent.parent?.id ?? 'unknown';
            return SeizureAlert.fromDoc(patientId: pId, doc: d);
          }).toList(),
        );
  }

  Stream<List<SeizureAlert>> streamAlerts(String patientId) {
    return alertsCol(patientId)
        .orderBy('time', descending: true)
        .snapshots()
        .map(
          (snap) => snap.docs
              .map((d) => SeizureAlert.fromDoc(patientId: patientId, doc: d))
              .toList(),
        );
  }

  Future<void> handleAlert({required String patientId, required String alertId}) async {
    await alertsCol(patientId).doc(alertId).update({'is_handled': true});
  }

  Future<void> addAlert({
    required String patientId,
    required DateTime time,
    required int heartRate,
    required int hrv,
  }) async {
    await alertsCol(patientId).add({
      'time': Timestamp.fromDate(time),
      'heartRate': heartRate,
      'hrv': hrv,
      'is_handled': false,
      'createdAt': FieldValue.serverTimestamp(),
    });
  }
}
