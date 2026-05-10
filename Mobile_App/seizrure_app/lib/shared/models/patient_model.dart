import 'package:cloud_firestore/cloud_firestore.dart';

class PatientModel {
  final String authUid;
  final String name;
  final String email;
  final String phone;
  final DateTime? birthdate;
  final String details;
  final List<String> caregiverIds;
  final DateTime? createdAt;
  final DateTime? updatedAt;

  PatientModel({
    required this.authUid,
    required this.name,
    required this.email,
    required this.phone,
    this.birthdate,
    this.details = '',
    this.caregiverIds = const [],
    this.createdAt,
    this.updatedAt,
  });

  factory PatientModel.fromFirestore(DocumentSnapshot doc) {
    Map<String, dynamic> data = doc.data() as Map<String, dynamic>;
    return PatientModel(
      authUid: data['authUid'] ?? doc.id,
      name: data['name'] ?? '',
      email: data['email'] ?? '',
      phone: data['phone'] ?? '',
      birthdate: (data['birthdate'] as Timestamp?)?.toDate(),
      details: data['details'] ?? '',
      caregiverIds: List<String>.from(data['caregiverIds'] ?? []),
      createdAt: (data['createdAt'] as Timestamp?)?.toDate(),
      updatedAt: (data['updatedAt'] as Timestamp?)?.toDate(),
    );
  }

  Map<String, dynamic> toFirestore() {
    return {
      'authUid': authUid,
      'name': name,
      'email': email,
      'phone': phone,
      'birthdate': birthdate != null ? Timestamp.fromDate(birthdate!) : null,
      'details': details,
      'caregiverIds': caregiverIds,
      'createdAt': createdAt != null ? Timestamp.fromDate(createdAt!) : FieldValue.serverTimestamp(),
      'updatedAt': FieldValue.serverTimestamp(),
    };
  }
}
