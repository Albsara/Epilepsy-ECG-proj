import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:get/get.dart';

enum SleepQuality { good, bad }

extension SleepQualityX on SleepQuality {
  String get label => this == SleepQuality.good ? 'Good' : 'Bad';

  static SleepQuality fromString(String? s) {
    if (s == 'bad') return SleepQuality.bad;
    return SleepQuality.good;
  }

  String toDb() => this == SleepQuality.bad ? 'bad' : 'good';
}

class Caregiver {
  final String id;
  String name;
  String email;
  String phone;
  List<String> patientIds;

  Caregiver({
    required this.id,
    required this.name,
    required this.email,
    required this.phone,
    this.patientIds = const [],
  });

  Map<String, dynamic> toMap() => {
    'name': name,
    'email': email,
    'phone': phone,
    'patientIds': patientIds,
  };

  static Caregiver fromDoc(DocumentSnapshot<Map<String, dynamic>> doc) {
    final d = doc.data() ?? {};
    return Caregiver(
      id: doc.id,
      name: (d['name'] ?? '').toString(),
      email: (d['email'] ?? '').toString(),
      phone: (d['phone'] ?? '').toString(),
      patientIds: List<String>.from(d['patientIds'] ?? []),
    );
  }
}

class SeizureAlert {
  final String id;
  final String patientId;
  final DateTime time;
  final int heartRate;
  final int hrv;
  final bool isHandled;
  final String medication;
  final String symptoms;
  final String sleep;
  final String stress;

  SeizureAlert({
    required this.id,
    required this.patientId,
    required this.time,
    required this.heartRate,
    required this.hrv,
    this.isHandled = false,
    this.medication = '—',
    this.symptoms = '—',
    this.sleep = '—',
    this.stress = '—',
  });

  Map<String, dynamic> toMap() => {
    'time': Timestamp.fromDate(time),
    'heartRate': heartRate,
    'hrv': hrv,
    'is_handled': isHandled,
    'medication': medication,
    'symptoms': symptoms,
    'sleep': sleep,
    'stress': stress,
  };

  static SeizureAlert fromDoc({
    required String patientId,
    required DocumentSnapshot<Map<String, dynamic>> doc,
  }) {
    final d = doc.data() ?? {};
    final ts = d['time'];
    return SeizureAlert(
      id: doc.id,
      patientId: patientId,
      time: (ts is Timestamp) ? ts.toDate() : DateTime.now(),
      heartRate: (d['heartRate'] ?? 0) is int
          ? d['heartRate']
          : int.tryParse('${d['heartRate']}') ?? 0,
      hrv: (d['hrv'] ?? 0) is int ? d['hrv'] : int.tryParse('${d['hrv']}') ?? 0,
      isHandled: d['is_handled'] == true,
      medication: (d['medication'] ?? '—').toString(),
      symptoms: (d['symptoms'] ?? '—').toString(),
      sleep: (d['sleep'] ?? '—').toString(),
      stress: (d['stress'] ?? '—').toString(),
    );
  }
}

class LiveStatus {
  int hr;
  int hrv;
  String medication;
  String symptoms;
  SleepQuality sleep;
  String stress;

  LiveStatus({
    required this.hr,
    required this.hrv,
    required this.medication,
    required this.symptoms,
    required this.sleep,
    required this.stress,
  });

  LiveStatus copyWith({
    int? hr,
    int? hrv,
    String? medication,
    String? symptoms,
    SleepQuality? sleep,
    String? stress,
  }) {
    return LiveStatus(
      hr: hr ?? this.hr,
      hrv: hrv ?? this.hrv,
      medication: medication ?? this.medication,
      symptoms: symptoms ?? this.symptoms,
      sleep: sleep ?? this.sleep,
      stress: stress ?? this.stress,
    );
  }
}

class Patient {
  final String id;
  String email;
  String name;
  DateTime birthdate;
  String phone;
  String details;
  List<String> caregiverIds;

  // Loaded on demand / via streams
  final RxList<Caregiver> caregivers = <Caregiver>[].obs;
  final RxList<SeizureAlert> alerts = <SeizureAlert>[].obs;

  // Live status comes from RTDB
  final Rx<LiveStatus> liveStatus;

  Patient({
    required this.id,
    required this.email,
    required this.name,
    required this.birthdate,
    required this.phone,
    required this.details,
    required LiveStatus liveStatus,
    this.caregiverIds = const [],
  }) : liveStatus = liveStatus.obs;

  Map<String, dynamic> toMap() => {
    'email': email,
    'name': name,
    'birthdate': Timestamp.fromDate(birthdate),
    'phone': phone,
    'details': details,
    'caregiverIds': caregiverIds,
  };

  static Patient fromDoc(DocumentSnapshot<Map<String, dynamic>> doc) {
    final d = doc.data() ?? {};
    final ts = d['birthdate'];
    return Patient(
      id: doc.id,
      email: (d['email'] ?? '').toString(),
      name: (d['name'] ?? '').toString(),
      birthdate: (ts is Timestamp) ? ts.toDate() : DateTime(2000, 1, 1),
      phone: (d['phone'] ?? '').toString(),
      details: (d['details'] ?? '').toString(),
      caregiverIds: List<String>.from(d['caregiverIds'] ?? []),
      liveStatus: LiveStatus(
        hr: 0,
        hrv: 0,
        medication: '—',
        symptoms: '—',
        sleep: SleepQuality.good,
        stress: '—',
      ),
    );
  }
}
