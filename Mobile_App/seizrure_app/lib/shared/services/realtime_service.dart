import 'package:firebase_database/firebase_database.dart';
import 'package:get/get.dart';
import '../models/live_data_model.dart';

class RealtimeService extends GetxService {
  final FirebaseDatabase _db = FirebaseDatabase.instance;

  Stream<LiveDataModel?> streamLiveData(String patientId) {
    return _db.ref('live/$patientId').onValue.map((event) {
      final data = event.snapshot.value as Map?;
      if (data != null) {
        return LiveDataModel.fromMap(data);
      }
      return null;
    });
  }

  Future<void> updateLiveData(String patientId, LiveDataModel liveData) async {
    await _db.ref('live/$patientId').set(liveData.toMap());
  }

  Future<void> setAlertStatus(String patientId, bool isDanger) async {
    await _db.ref('live/$patientId/alert').set(isDanger);
  }
}
