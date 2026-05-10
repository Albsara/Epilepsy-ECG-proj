import 'package:flutter/material.dart';
import 'package:get/get.dart';
import 'package:seizrure_app/shared/utils/app_colors.dart';
import 'package:seizrure_app/shared/controllers/app_controller.dart';

class BottomNavBar extends GetView<AppController> {
  const BottomNavBar({super.key, required this.radius});
  final double radius;

  @override
  Widget build(BuildContext context) {
    return Container(
      height: 72,
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(radius),
        boxShadow: const [
          BoxShadow(
            color: Color(0x12000000),
            blurRadius: 18,
            offset: Offset(0, 10),
          ),
        ],
      ),
      child: Obx(() {
        final idx = controller.navIndex.value;

        return Row(
          mainAxisAlignment: MainAxisAlignment.spaceAround,
          children: [
            NavItem(
              icon: Icons.home_filled,
              active: idx == 0,
              onTap: () => controller.goNav(0),
            ),
            NavItem(
              icon: Icons.show_chart,
              active: idx == 1,
              onTap: () => controller.goNav(1),
            ),
            NavItem(
              icon: Icons.checklist_rtl_rounded,
              active: idx == 2,
              onTap: () => controller.goNav(2),
            ),
            NavItem(
              icon: Icons.settings_outlined,
              active: idx == 3,
              onTap: () => controller.goNav(3),
            ),
          ],
        );
      }),
    );
  }
}

class NavItem extends StatelessWidget {
  const NavItem({
    super.key,
    required this.icon,
    required this.active,
    required this.onTap,
  });

  final IconData icon;
  final bool active;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        width: 44,
        height: 44,
        decoration: BoxDecoration(
          shape: BoxShape.circle,
          gradient: active
              ? const LinearGradient(
                  begin: Alignment.topLeft,
                  end: Alignment.bottomRight,
                  colors: [
                    AppColors.primaryGradientStart,
                    AppColors.primaryGradientEnd,
                  ],
                )
              : null,
          color: active ? null : Colors.transparent,
        ),
        child: Icon(
          icon,
          size: 22,
          color: active ? Colors.white : const Color(0xFF93A0AE),
        ),
      ),
    );
  }
}
