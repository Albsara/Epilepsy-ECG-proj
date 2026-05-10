import 'package:flutter/material.dart';
import 'package:flutter/foundation.dart';
import 'package:get/get.dart';
import 'package:seizrure_app/shared/controllers/app_controller.dart';
import 'package:seizrure_app/shared/utils/app_colors.dart';
import '../../shared/widgets/common_widgets.dart';
import '../../shared/routes/app_routes.dart';
import 'dashboard_widgets.dart';

class DashboardContent extends GetView<AppController> {
  const DashboardContent({super.key});

  @override
  Widget build(BuildContext context) {
    // keep indicator/nav in sync if user navigates back - Logic handled by MainLayout now mostly
    WidgetsBinding.instance.addPostFrameCallback((_) {
      //  if (controller.navIndex.value != 0) controller.navIndex.value = 0;
    });

    final mq = MediaQuery.of(context);
    final h = mq.size.height;
    final cardRadius = 18.0;

    return RefreshIndicator(
      onRefresh: controller.refreshData,
      color: AppColors.primaryGradientEnd,
      child: SingleChildScrollView(
        physics: const AlwaysScrollableScrollPhysics(),
        child: Column(
          children: [
            SizedBox(height: h * 0.015),
            CustomAppBar(
              title: 'dashboard',
              showRefresh: kIsWeb,
              onRefresh: controller.refreshData,
            ),
            SizedBox(height: h * 0.018),
            Column(
              children: [
                StatusCard(radius: cardRadius),
                SizedBox(height: h * 0.018),
                EcgCard(radius: cardRadius),
                SizedBox(height: h * 0.014),
                MetricsCard(radius: cardRadius),
                SizedBox(height: h * 0.018),
                GradientButton(
                  text: 'view_details'.tr,
                  height: 48,
                  radius: 14,
                  colors: const [Color(0xFF76CDE6), Color(0xFF5BA7D9)],
                  onTap: () => Get.toNamed(Routes.alertHistory),
                ),
                SizedBox(height: h * 0.02),
              ],
            ),
          ],
        ),
      ),
    );
  }
}
