import 'package:cloud_firestore/cloud_firestore.dart';

class CaregiverModel {
  final String authUid;
  final String name;
  final String email;
  final String phone;
  final List<String> patientIds;
  final DateTime? createdAt;
  final DateTime? updatedAt;

  CaregiverModel({
    required this.authUid,
    required this.name,
    required this.email,
    required this.phone,
    this.patientIds = const [],
    this.createdAt,
    this.updatedAt,
  });

  factory CaregiverModel.fromFirestore(DocumentSnapshot doc) {
    Map<String, dynamic> data = doc.data() as Map<String, dynamic>;
    return CaregiverModel(
      authUid: data['authUid'] ?? doc.id,
      name: data['name'] ?? '',
      email: data['email'] ?? '',
      phone: data['phone'] ?? '',
      patientIds: List<String>.from(data['patientIds'] ?? []),
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
      'patientIds': patientIds,
      'createdAt': createdAt != null ? Timestamp.fromDate(createdAt!) : FieldValue.serverTimestamp(),
      'updatedAt': FieldValue.serverTimestamp(),
    };
  }
}
