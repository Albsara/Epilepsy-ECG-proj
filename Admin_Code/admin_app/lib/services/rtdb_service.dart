import 'dart:async';
import 'package:firebase_database/firebase_database.dart';
import '../models/models.dart';

class RtdbService {
  final FirebaseDatabase _rtdb = FirebaseDatabase.instance;

  DatabaseReference liveRef(String patientId) => _rtdb.ref('live/$patientId');

  Stream<LiveStatus> streamLiveStatus(
    String patientId, {
    LiveStatus? fallback,
  }) {
    final base =
        fallback ??
        LiveStatus(
          hr: 0,
          hrv: 0,
          medication: '—',
          symptoms: '—',
          sleep: SleepQuality.good,
          stress: '—',
        );

    return liveRef(patientId).onValue.map((event) {
      final val = event.snapshot.value;
      if (val is! Map) return base;

      int toInt(dynamic x) {
        if (x is int) return x;
        if (x is double) return x.toInt();
        return int.tryParse('$x') ?? 0;
      }

      final hr = toInt(val['hr']);
      final hrv = toInt(val['hrv']);

      final medication = (val['medication'] ?? base.medication).toString();
      final symptoms = (val['symptoms'] ?? base.symptoms).toString();
      final stress = (val['stress'] ?? base.stress).toString();
      final sleepStr = val['sleep']?.toString();
      final sleep = SleepQualityX.fromString(sleepStr);

      return base.copyWith(
        hr: hr,
        hrv: hrv,
        medication: medication,
        symptoms: symptoms,
        stress: stress,
        sleep: sleep,
      );
    });
  }

  Future<void> setHrHrv(
    String patientId, {
    required int hr,
    required int hrv,
  }) async {
    await liveRef(patientId).update({'hr': hr, 'hrv': hrv});
  }
}
