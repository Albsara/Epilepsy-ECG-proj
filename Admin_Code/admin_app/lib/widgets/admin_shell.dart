import 'package:flutter/material.dart';
import 'package:get/get.dart';
import '../controllers/auth_controller.dart';
import '../routes.dart';
import '../utils/app_colors.dart';
import 'common_widgets.dart';

class AdminShell extends StatelessWidget {
  final String title;
  final Widget child;
  final List<Widget>? headerActions;

  const AdminShell({
    super.key,
    required this.title,
    required this.child,
    this.headerActions,
  });

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(
      builder: (context, constraints) {
        final isDesktop = constraints.maxWidth >= 900;

        final header = Container(
          width: double.infinity,
          padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
          decoration: const BoxDecoration(
            color: Colors.white,
            border: Border(bottom: BorderSide(color: AppColors.border)),
          ),
          child: Row(
            children: [
              if (!isDesktop)
                Builder(
                  builder: (ctx) => IconButton(
                    onPressed: () => Scaffold.of(ctx).openDrawer(),
                    icon: const Icon(Icons.menu, color: AppColors.text),
                    tooltip: 'menu'.tr,
                    padding: EdgeInsets.zero,
                    constraints: const BoxConstraints(),
                  ),
                ),
              if (!isDesktop) const SizedBox(width: 12),
              Expanded(
                child: Text(
                  title,
                  style: const TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.w900,
                    color: AppColors.text,
                    overflow: TextOverflow.ellipsis,
                  ),
                ),
              ),
              const SizedBox(width: 8),
              if (headerActions != null)
                Row(
                  mainAxisSize: MainAxisSize.min,
                  children: headerActions!,
                ),
              const SizedBox(width: 8),
              const LanguageToggleButton(),
            ],
          ),
        );

        final nav = _SideNav(isDesktop: isDesktop);

        if (isDesktop) {
          return Scaffold(
            body: Row(
              children: [
                SizedBox(width: 260, child: nav),
                Expanded(
                  child: Column(
                    children: [
                      header,
                      Expanded(
                        child: SingleChildScrollView(
                          padding: responsivePadding(context),
                          child: child,
                        ),
                      ),
                    ],
                  ),
                ),
              ],
            ),
          );
        }

        // ✅ MOBILE/TABLET
        return Scaffold(
          drawer: Drawer(
            backgroundColor: Colors.white,
            child: SafeArea(child: nav),
          ),
          body: SafeArea(
            // Add extra breathing room so content isn't tight to notch/status bar
            child: Padding(
              padding: const EdgeInsets.only(top: 6), // <--- extra space
              child: Column(
                children: [
                  header,
                  Expanded(
                    child: SingleChildScrollView(
                      // add a little extra padding for mobile content
                      padding: responsivePadding(
                        context,
                      ).copyWith(top: responsivePadding(context).top + 6),
                      child: child,
                    ),
                  ),
                ],
              ),
            ),
          ),
        );
      },
    );
  }
}

class _SideNav extends StatelessWidget {
  final bool isDesktop;
  const _SideNav({required this.isDesktop});

  Widget _navItem({
    required IconData icon,
    required String label,
    required VoidCallback onTap,
    bool active = false,
  }) {
    return InkWell(
      onTap: onTap,
      child: Container(
        margin: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 12),
        decoration: BoxDecoration(
          color: active ? AppColors.accent.withOpacity(0.10) : Colors.white,
          borderRadius: BorderRadius.circular(12),
          border: Border.all(
            color: active
                ? AppColors.accent.withOpacity(0.35)
                : AppColors.border,
          ),
        ),
        child: Row(
          children: [
            Icon(
              icon,
              size: 20,
              color: active ? AppColors.accent : AppColors.muted,
            ),
            const SizedBox(width: 10),
            Expanded(
              child: Text(
                label,
                style: TextStyle(
                  color: AppColors.text,
                  fontWeight: active ? FontWeight.w900 : FontWeight.w700,
                ),
              ),
            ),
            const Icon(Icons.chevron_right, color: AppColors.border),
          ],
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final auth = Get.find<AuthController>();
    final route = Get.currentRoute;

    return Container(
      decoration: const BoxDecoration(
        color: Colors.white,
        border: Border(right: BorderSide(color: AppColors.border)),
      ),
      child: Column(
        children: [
          Container(
            width: double.infinity,
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 18),
            decoration: const BoxDecoration(
              color: Colors.white,
              border: Border(bottom: BorderSide(color: AppColors.border)),
            ),
            child: Row(
              children: [
                Container(
                  width: 10,
                  height: 44,
                  decoration: BoxDecoration(
                    color: AppColors.accent,
                    borderRadius: BorderRadius.circular(10),
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Obx(
                        () => Container(
                          padding: const EdgeInsets.symmetric(
                            horizontal: 10,
                            vertical: 8,
                          ),
                          decoration: BoxDecoration(
                            color: AppColors.bgSoft,
                            borderRadius: BorderRadius.circular(999),
                            border: Border.all(color: AppColors.border),
                          ),
                          child: Row(
                            children: [
                              const Icon(
                                Icons.verified_user,
                                size: 18,
                                color: AppColors.muted,
                              ),
                              const SizedBox(width: 8),
                              Text(
                                auth.adminEmail.value.isEmpty
                                    ? '—'
                                    : auth.adminEmail.value,
                                style: const TextStyle(
                                  color: AppColors.muted,
                                  fontSize: 12,
                                  fontWeight: FontWeight.w700,
                                ),
                              ),
                            ],
                          ),
                        ),
                      ),
                      const SizedBox(height: 6),
                      Obx(
                        () => Text(
                          auth.adminName.value,
                          style: const TextStyle(
                            fontSize: 12,
                            fontWeight: FontWeight.w700,
                            color: AppColors.muted,
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(height: 8),
          _navItem(
            icon: Icons.dashboard_outlined,
            label: 'dashboard'.tr,
            active: route == Routes.dashboard,
            onTap: () => Get.offAllNamed(Routes.dashboard),
          ),
          _navItem(
            icon: Icons.people_alt_outlined,
            label: 'patients'.tr,
            active: route == Routes.patients || route.startsWith('/patients'),
            onTap: () => Get.offAllNamed(Routes.patients),
          ),
          _navItem(
            icon: Icons.support_agent,
            label: 'caregivers'.tr,
            active: route == Routes.caregivers,
            onTap: () => Get.offAllNamed(Routes.caregivers),
          ),
          _navItem(
            icon: Icons.notifications_active_outlined,
            label: 'alerts_history'.tr ?? 'Alerts History',
            active: route == Routes.alertsHistory,
            onTap: () => Get.offAllNamed(Routes.alertsHistory),
          ),
          _navItem(
            icon: Icons.person_outline,
            label: 'personal_info'.tr,
            active: route == Routes.profile,
            onTap: () => Get.toNamed(Routes.profile),
          ),
          const Spacer(),
          Container(
            margin: const EdgeInsets.all(12),
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: AppColors.alert.withOpacity(0.08),
              borderRadius: BorderRadius.circular(12),
              border: Border.all(color: AppColors.alert.withOpacity(0.25)),
            ),
            child: Row(
              children: [
                const Icon(Icons.logout, color: AppColors.alert),
                const SizedBox(width: 10),
                Expanded(
                  child: TextButton(
                    onPressed: () => auth.logout(),
                    style: TextButton.styleFrom(
                      foregroundColor: AppColors.alert,
                      padding: const EdgeInsets.symmetric(vertical: 12),
                      textStyle: const TextStyle(fontWeight: FontWeight.w900),
                    ),
                    child: Align(
                      alignment: Alignment.centerLeft,
                      child: Text('logout'.tr),
                    ),
                  ),
                ),
              ],
            ),
          ),
          if (!isDesktop) const SizedBox(height: 8),
        ],
      ),
    );
  }
}
