import 'package:firebase_core/firebase_core.dart';
import 'package:firebase_messaging/firebase_messaging.dart';
import 'package:flutter/material.dart';
import 'package:flutter/foundation.dart';
import 'package:get/get.dart';
import 'firebase_options.dart';

import 'shared/routes/app_routes.dart';
import 'shared/utils/translations.dart';
import 'package:seizrure_app/shared/controllers/app_controller.dart';
import 'patient/main_layout.dart';
import 'patient/profile/profile_view.dart';
import 'patient/history/alerts_history_view.dart';
import 'patient/status/daily_status_view.dart';

import 'shared/auth/login_view.dart';
import 'shared/auth/forgot_password_view.dart';
import 'caregiver/caregiver_home_view.dart';
import 'caregiver/caregiver_profile_view.dart';
import 'caregiver/patient_details_view.dart';

import 'package:get_storage/get_storage.dart';
import 'shared/services/background_monitor_service.dart';

@pragma('vm:entry-point')
Future<void> _firebaseMessagingBackgroundHandler(RemoteMessage message) async {
  await Firebase.initializeApp(options: DefaultFirebaseOptions.currentPlatform);
  print("Background Message ID: ${message.messageId}");
}

void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  // Initialize storage and firebase with error handling and don't let it block app start indefinitely
  try {
    await GetStorage.init();
    await Firebase.initializeApp(
      options: DefaultFirebaseOptions.currentPlatform,
    ).timeout(const Duration(seconds: 5));
    print('Main: Firebase Initialized');
  } catch (e) {
    print('Main: Initialization Error: $e');
  }

  // Set up background service AFTER app starts to avoid blocking the main thread during boot
  if (!kIsWeb) {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      BackgroundMonitorService.initializeService();
    });
    FirebaseMessaging.onBackgroundMessage(_firebaseMessagingBackgroundHandler);
  }

  runApp(const App());
}

class App extends StatelessWidget {
  const App({super.key});

  @override
  Widget build(BuildContext context) {
    return GetMaterialApp(
      debugShowCheckedModeBanner: false,
      initialBinding: AppBinding(),
      translations: AppTranslations(),
      locale: const Locale('en'),
      fallbackLocale: const Locale('en'),
      initialRoute: Routes.login,
      getPages: [
        GetPage(
          name: Routes.login,
          page: () => const LoginView(),
          binding: AppBinding(),
        ),
        GetPage(
          name: Routes.forgotPassword,
          page: () => const ForgotPasswordView(),
          binding: AppBinding(),
        ),
        GetPage(
          name: Routes.dashboard,
          page: () => const MainLayout(),
          binding: AppBinding(),
        ),
        GetPage(
          name: Routes.profile,
          page: () => const ProfileView(),
          binding: AppBinding(),
        ),
        GetPage(
          name: Routes.alertHistory,
          page: () => const AlertsHistoryView(),
          binding: AppBinding(),
        ),
        GetPage(
          name: Routes.dailyStatus,
          page: () => const DailyStatusView(),
          binding: AppBinding(),
        ),
        GetPage(
          name: Routes.caregiverHome,
          page: () => const CaregiverHomeView(),
          binding: AppBinding(),
        ),
        GetPage(
          name: Routes.caregiverProfile,
          page: () => const CaregiverProfileView(),
          binding: AppBinding(),
        ),
        GetPage(
          name: Routes.patientDetails,
          page: () => const PatientDetailsView(),
          binding: AppBinding(),
        ),
      ],
    );
  }
}
