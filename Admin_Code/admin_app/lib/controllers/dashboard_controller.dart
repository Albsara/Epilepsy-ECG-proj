import 'package:get/get.dart';
import 'data_controller.dart';

class DashboardController extends GetxController {
  // change this value to force StreamBuilder + UI rebuild
  final RxInt refreshTick = 0.obs;

  /// Call this from UI (Refresh button).
  Future<void> refreshDashboard() async {
    // Force rebuild of Dashboard widgets that depend on refreshTick
    refreshTick.value++;

    // Also refresh patients list UI (GetX)
    final data = Get.find<DataController>();
    data.patients.refresh();
  }
}
