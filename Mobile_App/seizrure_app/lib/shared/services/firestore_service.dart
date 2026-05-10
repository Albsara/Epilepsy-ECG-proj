import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:get/get.dart';
import '../models/patient_model.dart';
import '../models/caregiver_model.dart';
import '../models/alert_model.dart';

class FirestoreService extends GetxService {
  final FirebaseFirestore _db = FirebaseFirestore.instance;

  // Patient methods
  Future<PatientModel?> getPatient(String uid) async {
    final doc = await _db.collection('patients').doc(uid).get();
    if (doc.exists) {
      return PatientModel.fromFirestore(doc);
    }
    return null;
  }

  Stream<PatientModel?> streamPatient(String uid) {
    return _db.collection('patients').doc(uid).snapshots().map((doc) =>
        doc.exists ? PatientModel.fromFirestore(doc) : null);
  }

  // Caregiver methods
  Future<CaregiverModel?> getCaregiver(String uid) async {
    final doc = await _db.collection('caregivers').doc(uid).get();
    if (doc.exists) {
      return CaregiverModel.fromFirestore(doc);
    }
    return null;
  }

  Stream<CaregiverModel?> streamCaregiver(String uid) {
    return _db.collection('caregivers').doc(uid).snapshots().map((doc) =>
        doc.exists ? CaregiverModel.fromFirestore(doc) : null);
  }

  // Alerts methods
  Stream<List<AlertModel>> streamPatientAlerts(String patientId) {
    return _db
        .collection('patients')
        .doc(patientId)
        .collection('alerts')
        .orderBy('time', descending: true)
        .snapshots()
        .map((snapshot) => snapshot.docs
            .map((doc) => AlertModel.fromFirestore(doc))
            .toList());
  }

  // Collection Group Query for Global Alerts
  Stream<List<AlertModel>> streamAllAlerts() {
    return _db
        .collectionGroup('alerts')
        .orderBy('time', descending: true)
        .snapshots()
        .map((snapshot) => snapshot.docs
            .map((doc) => AlertModel.fromFirestore(doc))
            .toList());
  }

  Future<void> addSeizureAlert(String patientId, AlertModel alert) async {
    await _db
        .collection('patients')
        .doc(patientId)
        .collection('alerts')
        .add(alert.toFirestore());
  }

  Future<void> updateAlertHandled(String patientId, String alertId, bool handled) async {
    await _db
        .collection('patients')
        .doc(patientId)
        .collection('alerts')
        .doc(alertId)
        .update({'is_handled': handled});
  }

  Future<void> updatePatient(String uid, Map<String, dynamic> data) async {
    await _db.collection('patients').doc(uid).update(data);
  }

  Future<void> updateCaregiver(String uid, Map<String, dynamic> data) async {
    await _db.collection('caregivers').doc(uid).update(data);
  }

  // Bidirectional link between Patient and Caregiver
  Future<void> linkCaregiverToPatient(String patientId, String caregiverId) async {
    final batch = _db.batch();
    
    batch.update(_db.collection('patients').doc(patientId), {
      'caregiverIds': FieldValue.arrayUnion([caregiverId])
    });

    batch.update(_db.collection('caregivers').doc(caregiverId), {
      'patientIds': FieldValue.arrayUnion([patientId])
    });

    await batch.commit();
  }

  Future<void> unlinkCaregiverFromPatient(String patientId, String caregiverId) async {
    final batch = _db.batch();
    
    batch.update(_db.collection('patients').doc(patientId), {
      'caregiverIds': FieldValue.arrayRemove([caregiverId])
    });

    batch.update(_db.collection('caregivers').doc(caregiverId), {
      'patientIds': FieldValue.arrayRemove([patientId])
    });

    await batch.commit();
  }
}
