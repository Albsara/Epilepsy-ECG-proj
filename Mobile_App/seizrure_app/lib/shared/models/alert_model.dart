import 'package:cloud_firestore/cloud_firestore.dart';

class AlertModel {
  final String id;
  final String? patientId;
  final num heartRate;
  final num hrv;
  final String medication;
  final String symptoms;
  final String sleep;
  final String stress;
  final DateTime time;
  final bool is_handled;
  final DateTime? createdAt;

  AlertModel({
    required this.id,
    this.patientId,
    required this.heartRate,
    required this.hrv,
    this.medication = 'yes',
    this.symptoms = 'no',
    this.sleep = 'good',
    this.stress = 'low',
    required this.time,
    this.is_handled = false,
    this.createdAt,
  });

  factory AlertModel.fromFirestore(DocumentSnapshot doc) {
    Map<String, dynamic> data = doc.data() as Map<String, dynamic>;
    String? pId;
    try {
      pId = doc.reference.parent.parent?.id;
    } catch (_) {}

    return AlertModel(
      id: doc.id,
      patientId: pId,
      heartRate: data['heartRate'] ?? 0,
      hrv: data['hrv'] ?? 0,
      medication: data['medication'] ?? 'yes',
      symptoms: data['symptoms'] ?? 'no',
      sleep: data['sleep'] ?? 'good',
      stress: data['stress'] ?? 'low',
      time: (data['time'] as Timestamp?)?.toDate() ?? DateTime.now(),
      is_handled: data['is_handled'] ?? false,
      createdAt: (data['createdAt'] as Timestamp?)?.toDate(),
    );
  }

  Map<String, dynamic> toFirestore() {
    return {
      'heartRate': heartRate,
      'hrv': hrv,
      'medication': medication,
      'symptoms': symptoms,
      'sleep': sleep,
      'stress': stress,
      'time': Timestamp.fromDate(time),
      'is_handled': is_handled,
      'createdAt': createdAt != null ? Timestamp.fromDate(createdAt!) : FieldValue.serverTimestamp(),
    };
  }
}

