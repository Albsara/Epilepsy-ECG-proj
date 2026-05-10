class LiveDataModel {
  final int hr;
  final int hrv;
  final String medication;
  final String symptoms;
  final String sleep;
  final String stress;
  final bool alert;
  final DateTime? updatedAt;

  LiveDataModel({
    required this.hr,
    required this.hrv,
    this.medication = 'yes',
    this.symptoms = '',
    this.sleep = 'good',
    this.stress = 'low',
    this.alert = false,
    this.updatedAt,
  });

  factory LiveDataModel.fromMap(Map<dynamic, dynamic> data) {
    return LiveDataModel(
      hr: data['hr'] ?? 0,
      hrv: data['hrv'] ?? 0,
      medication: data['medication'] ?? 'yes',
      symptoms: data['symptoms'] ?? '',
      sleep: data['sleep'] ?? 'good',
      stress: data['stress'] ?? 'low',
      alert: data['alert'] ?? false,
      updatedAt: data['updatedAt'] != null 
          ? DateTime.fromMillisecondsSinceEpoch(data['updatedAt'] as int) 
          : null,
    );
  }

  Map<String, dynamic> toMap() {
    return {
      'hr': hr,
      'hrv': hrv,
      'medication': medication,
      'symptoms': symptoms,
      'sleep': sleep,
      'stress': stress,
      'alert': alert,
      'updatedAt': DateTime.now().millisecondsSinceEpoch,
    };
  }
}
