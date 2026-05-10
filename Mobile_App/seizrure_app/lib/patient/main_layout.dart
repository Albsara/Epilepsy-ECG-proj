import 'package:flutter/material.dart';
import 'package:get/get.dart';
import 'package:seizrure_app/shared/controllers/app_controller.dart';
import '../shared/widgets/bottom_nav.dart';
import 'dashboard/dashboard_content.dart';
import 'seizure/seizure_content.dart';
import 'settings/settings_content.dart';
import 'status/daily_status_view.dart';
import '../shared/utils/app_colors.dart';

class MainLayout extends GetView<AppController> {
  const MainLayout({super.key});

  @override
  Widget build(BuildContext context) {
    final mq = MediaQuery.of(context);
    final w = mq.size.width;

    return Scaffold(
      backgroundColor: AppColors.background,
      body: SafeArea(
        child: Padding(
          padding: EdgeInsets.symmetric(horizontal: w * 0.065),
          child: Obx(
            () => IndexedStack(
              index: controller.navIndex.value,
              children: [
                const DashboardContent(),
                SeizureContent(),
                DailyStatusView(),
                SettingsContent(),
              ],
            ),
          ),
        ),
      ),
      bottomNavigationBar: Padding(
        padding: EdgeInsets.only(left: w * 0.065, right: w * 0.065, bottom: 18),
        child: const BottomNavBar(radius: 18),
      ),
    );
  }
}
