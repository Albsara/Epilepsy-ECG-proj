import 'package:flutter/material.dart';
import 'package:get/get.dart';
import 'package:seizrure_app/shared/controllers/app_controller.dart';
import '../../shared/utils/app_colors.dart';
import '../../shared/widgets/common_widgets.dart';
import '../../shared/widgets/seizure_alert_tile.dart';

class AlertsHistoryView extends GetView<AppController> {
  const AlertsHistoryView({super.key});

  @override
  Widget build(BuildContext context) {
    final w = MediaQuery.of(context).size.width;

    return Scaffold(
      backgroundColor: AppColors.background,
      body: SafeArea(
        child: Padding(
          padding: EdgeInsets.symmetric(horizontal: w * 0.05),
          child: Column(
            children: [
              const SizedBox(height: 10),
              const CustomAppBar(
                title: 'seizure_history',
                showBackButton: true,
              ),
              const SizedBox(height: 10),
              Expanded(
                child: RefreshIndicator(
                  onRefresh: controller.refreshData,
                  color: AppColors.primaryGradientEnd,
                  child: Obx(() {
                    if (controller.historyList.isEmpty) {
                      return SingleChildScrollView(
                        physics: const AlwaysScrollableScrollPhysics(),
                        child: SizedBox(
                          height: MediaQuery.of(context).size.height * 0.6,
                          child: Center(
                            child: Text(
                              'no_alerts'.tr,
                              style: const TextStyle(
                                color: AppColors.textGrey,
                                fontSize: 16,
                              ),
                            ),
                          ),
                        ),
                      );
                    }

                    return ListView.builder(
                      padding: const EdgeInsets.symmetric(vertical: 16),
                      clipBehavior: Clip.none,
                      itemCount: controller.historyList.length,
                      itemBuilder: (ctx, i) {
                        final item = controller.historyList[i];
                        return SeizureAlertTile(
                          item: item,
                          isCaregiver: !controller.isPatient.value,
                          onHandled: () {
                            controller.toggleAlertHandled(
                              item['patientId']!,
                              item['id']!,
                              item['is_handled'] == 'true',
                            );
                          },
                        );
                      },
                    );
                  }),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
