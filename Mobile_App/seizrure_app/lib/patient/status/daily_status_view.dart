import 'package:flutter/material.dart';
import 'package:get/get.dart';
import '../../shared/widgets/common_widgets.dart';
import '../../shared/utils/app_colors.dart';
import 'package:seizrure_app/shared/controllers/auth_controller.dart';
import 'package:seizrure_app/shared/models/live_data_model.dart';
import 'package:seizrure_app/shared/services/realtime_service.dart';
import 'package:seizrure_app/shared/controllers/app_controller.dart';

class DailyStatusController extends GetxController {
  final RealtimeService _realtimeService = Get.find<RealtimeService>();
  final AuthController _authController = Get.find<AuthController>();
  final AppController _appController = Get.find<AppController>();

  final RxBool hasSymptoms = false.obs;
  final RxBool isSleepGood = true.obs;
  final RxBool tookMedication = true.obs;
  final RxBool isStressNormal = true.obs;

  @override
  void onInit() {
    super.onInit();
    // Initialize from current app state
    hasSymptoms.value = _appController.symptoms.value == 'yes';
    isSleepGood.value = _appController.sleep.value == 'good';
    tookMedication.value = _appController.medication.value == 'yes';
    isStressNormal.value = _appController.stress.value == 'low';
  }

  Future<void> saveStatus() async {
    final user = _authController.firebaseUser.value;
    if (user == null) return;

    final liveData = LiveDataModel(
      hr: _appController.bpm.value,
      hrv: _appController.hrv.value,
      medication: tookMedication.value ? 'yes' : 'no',
      symptoms: hasSymptoms.value ? 'yes' : 'no',
      sleep: isSleepGood.value ? 'good' : 'bad',
      stress: isStressNormal.value ? 'low' : 'high',
    );

    try {
      await _realtimeService.updateLiveData(user.uid, liveData);
      
      Get.defaultDialog(
        title: 'success'.tr,
        middleText: 'status_saved'.tr,
        textConfirm: 'ok'.tr,
        confirmTextColor: Colors.white,
        buttonColor: Colors.green,
        onConfirm: () => Get.back(),
      );
    } catch (e) {
      Get.defaultDialog(
        title: 'error'.tr,
        middleText: 'failed_to_save_status'.trParams({'error': e.toString()}),
        textConfirm: 'ok'.tr,
        confirmTextColor: Colors.white,
        buttonColor: Colors.red,
        onConfirm: () => Get.back(),
      );
    }
  }
}

class DailyStatusView extends StatelessWidget {
  const DailyStatusView({super.key});

  @override
  Widget build(BuildContext context) {
    // Ensure controller is created
    final controller = Get.put(DailyStatusController());
    final h = MediaQuery.of(context).size.height;

    return Column(
      children: [
        SizedBox(height: h * 0.015), // Consistent spacing
        const CustomAppBar(
          title: 'current_status',
          showLangToggle:
              true, // Uses global controller.toggleLang via CustomAppBar
        ),

        SizedBox(height: h * 0.018),
        Expanded(
          child: ListView(
            // padding handled by MainLayout?
            // MainLayout has Padding(horizontal: w * 0.065, child: Obx(... IndexedStack ...))
            // So we don't need horizontal padding here if we want to align with others.
            // However, the cards have shadows and might need some margin or the parent padding is enough.
            // Previous DailyStatusView had padding horizontal 24.
            // MainLayout padding is w * 0.065 approx 24-25.
            // So let's remove horizontal padding from ListView to avoid double padding.
            clipBehavior: Clip.none, // Allow shadows to paint outside
            children: [
              const SizedBox(height: 10),

              _buildCard(
                context,
                title: 'symptoms',
                icon: Icons.sick_outlined,
                child: _buildToggleGroup(
                  option1: 'yes',
                  option2: 'no',
                  groupValue: controller.hasSymptoms,
                  val1: true,
                  val2: false,
                  color1: const Color(0xFFEF4444), // Red
                  color2: const Color(0xFF22C55E), // Green
                ),
              ),

              _buildCard(
                context,
                title: 'sleep',
                icon: Icons.bedtime_outlined,
                child: _buildToggleGroup(
                  option1: 'good',
                  option2: 'bad',
                  groupValue: controller.isSleepGood,
                  val1: true,
                  val2: false,
                  color1: const Color(0xFF22C55E), // Green
                  color2: const Color(0xFFEF4444), // Red
                ),
              ),

              _buildCard(
                context,
                title: 'medication',
                icon: Icons.medication_outlined,
                child: _buildToggleGroup(
                  option1: 'yes',
                  option2: 'no',
                  groupValue: controller.tookMedication,
                  val1: true,
                  val2: false,
                  color1: const Color(0xFF22C55E), // Green
                  color2: const Color(0xFFEF4444), // Red
                ),
              ),

              _buildCard(
                context,
                title: 'stress',
                icon: Icons.bolt_outlined,
                child: _buildToggleGroup(
                  option1: 'normal_stress',
                  option2: 'high_stress',
                  groupValue: controller.isStressNormal,
                  val1: true,
                  val2: false,
                  color1: const Color(0xFF22C55E), // Green
                  color2: const Color(0xFFEF4444), // Red
                ),
              ),

              SizedBox(height: h * 0.04),

              GradientButton(
                text: 'save'.tr,
                height: 54,
                radius: 16,
                colors: const [
                  AppColors.primaryGradientStart,
                  AppColors.primaryGradientEnd,
                ],
                onTap: controller.saveStatus,
              ),
              const SizedBox(height: 40),
            ],
          ),
        ),
      ],
    );
  }

  Widget _buildCard(
    BuildContext context, {
    required String title,
    required IconData icon,
    required Widget child,
  }) {
    return Container(
      margin: const EdgeInsets.only(bottom: 20),
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(20),
        boxShadow: [
          BoxShadow(
            color: const Color(0xFF64748B).withOpacity(0.08),
            blurRadius: 24,
            offset: const Offset(0, 8),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                padding: const EdgeInsets.all(10),
                decoration: BoxDecoration(
                  color: const Color(0xFFF1F5F9),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: Icon(
                  icon,
                  color: AppColors.primaryGradientStart,
                  size: 22,
                ),
              ),
              const SizedBox(width: 14),
              Expanded(
                child: Text(
                  title.tr,
                  style: const TextStyle(
                    fontSize: 17,
                    fontWeight: FontWeight.w700,
                    color: Color(0xFF1E293B),
                    letterSpacing: -0.5,
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 20),
          child,
        ],
      ),
    );
  }

  Widget _buildToggleGroup({
    required String option1,
    required String option2,
    required RxBool groupValue,
    required bool val1, // Value for option1
    required bool val2, // Value for option2
    required Color color1, // Active color for option1
    required Color color2, // Active color for option2
  }) {
    return Container(
      padding: const EdgeInsets.all(4),
      decoration: BoxDecoration(
        color: const Color(0xFFF8FAFC),
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: const Color(0xFFE2E8F0)),
      ),
      child: Obx(() {
        final current = groupValue.value;
        return Row(
          children: [
            Expanded(
              child: _AnimatedOptionButton(
                text: option1,
                isSelected: current == val1,
                activeColor: color1,
                onTap: () => groupValue.value = val1,
              ),
            ),
            const SizedBox(width: 4),
            Expanded(
              child: _AnimatedOptionButton(
                text: option2,
                isSelected: current == val2,
                activeColor: color2,
                onTap: () => groupValue.value = val2,
              ),
            ),
          ],
        );
      }),
    );
  }
}

class _AnimatedOptionButton extends StatelessWidget {
  const _AnimatedOptionButton({
    required this.text,
    required this.isSelected,
    required this.activeColor,
    required this.onTap,
  });

  final String text;
  final bool isSelected;
  final Color activeColor;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 250),
        curve: Curves.easeOutCubic,
        padding: const EdgeInsets.symmetric(vertical: 12),
        decoration: BoxDecoration(
          color: isSelected ? Colors.white : Colors.transparent,
          borderRadius: BorderRadius.circular(10),
          boxShadow: isSelected
              ? [
                  BoxShadow(
                    color: activeColor.withOpacity(0.15),
                    blurRadius: 8,
                    offset: const Offset(0, 4),
                  ),
                ]
              : [],
          border: isSelected
              ? Border.all(color: activeColor.withOpacity(0.2), width: 1)
              : Border.all(color: Colors.transparent, width: 1),
        ),
        alignment: Alignment.center,
        child: Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            // Optional: Add indicator dot
            AnimatedContainer(
              duration: const Duration(milliseconds: 200),
              width: 8,
              height: 8,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                color: isSelected ? activeColor : const Color(0xFF94A3B8),
              ),
            ),
            const SizedBox(width: 8),
            Text(
              text.tr,
              style: TextStyle(
                fontSize: 14,
                fontWeight: isSelected ? FontWeight.w700 : FontWeight.w600,
                color: isSelected
                    ? const Color(0xFF1E293B)
                    : const Color(0xFF64748B),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
