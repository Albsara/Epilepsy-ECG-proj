import 'package:admin_app/controllers/auth_controller.dart';
import 'package:admin_app/controllers/dashboard_controller.dart';
import 'package:admin_app/controllers/data_controller.dart';
import 'package:admin_app/firebase_options.dart';
import 'package:admin_app/routes.dart';
import 'package:admin_app/utils/app_colors.dart';
import 'package:admin_app/screens/login_page.dart';
import 'package:admin_app/screens/forgot_password_page.dart';
import 'package:admin_app/screens/dashboard_page.dart';
import 'package:admin_app/screens/patients_page.dart';
import 'package:admin_app/screens/patient_form_page.dart';
import 'package:admin_app/screens/patient_detail_page.dart';
import 'package:admin_app/screens/admin_profile_page.dart';
import 'package:admin_app/screens/caregivers_page.dart';
import 'package:admin_app/screens/alerts_history_page.dart';
import 'package:firebase_core/firebase_core.dart';
import 'package:firebase_database/firebase_database.dart';
import 'package:admin_app/utils/messages.dart';
import 'package:flutter_localizations/flutter_localizations.dart';

import 'package:flutter/material.dart';
import 'package:get/get.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await Firebase.initializeApp(options: DefaultFirebaseOptions.currentPlatform);

  runApp(const SeizureAdminApp());
}

class SeizureAdminApp extends StatelessWidget {
  const SeizureAdminApp({super.key});

  @override
  Widget build(BuildContext context) {
    // Controllers
    Get.put(AuthController(), permanent: true);
    Get.put(DataController(), permanent: true);
    Get.put(DashboardController(), permanent: true);

    return GetMaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Seizure Admin',
      translations: Messages(),
      locale: Get.deviceLocale,
      fallbackLocale: const Locale('en', 'US'),
      localizationsDelegates: const [
        GlobalMaterialLocalizations.delegate,
        GlobalWidgetsLocalizations.delegate,
        GlobalCupertinoLocalizations.delegate,
      ],
      supportedLocales: const [Locale('en', 'US'), Locale('ar', 'SA')],
      theme: ThemeData(
        scaffoldBackgroundColor: AppColors.bgSoft,
        fontFamily: 'Roboto',
        textTheme: const TextTheme(
          bodyMedium: TextStyle(color: AppColors.text),
        ),
        inputDecorationTheme: InputDecorationTheme(
          filled: true,
          fillColor: Colors.white,
          contentPadding: const EdgeInsets.symmetric(
            horizontal: 12,
            vertical: 12,
          ),
          border: OutlineInputBorder(
            borderSide: const BorderSide(color: AppColors.border),
            borderRadius: BorderRadius.circular(12),
          ),
          enabledBorder: OutlineInputBorder(
            borderSide: const BorderSide(color: AppColors.border),
            borderRadius: BorderRadius.circular(12),
          ),
          focusedBorder: OutlineInputBorder(
            borderSide: const BorderSide(color: AppColors.accent),
            borderRadius: BorderRadius.circular(12),
          ),
        ),
      ),
      initialRoute: Routes.login,
      getPages: [
        GetPage(name: Routes.login, page: () => const LoginPage()),
        GetPage(name: Routes.forgot, page: () => const ForgotPasswordPage()),
        GetPage(
          name: Routes.dashboard,
          page: () => DashboardPage(),
          middlewares: [AuthGuard()],
        ),
        GetPage(
          name: Routes.patients,
          page: () => const PatientsPage(),
          middlewares: [AuthGuard()],
        ),
        GetPage(
          name: Routes.caregivers,
          page: () => const CaregiversPage(),
          middlewares: [AuthGuard()],
        ),
        GetPage(
          name: Routes.alertsHistory,
          page: () => const AlertsHistoryPage(),
          middlewares: [AuthGuard()],
        ),
        GetPage(
          name: Routes.addPatient,
          page: () => const AddPatientPage(),
          middlewares: [AuthGuard()],
        ),
        GetPage(
          name: Routes.patientDetail,
          page: () => const PatientDetailPage(),
          middlewares: [AuthGuard()],
        ),
        GetPage(
          name: Routes.editPatient,
          page: () => const EditPatientPage(),
          middlewares: [AuthGuard()],
        ),
        GetPage(
          name: Routes.profile,
          page: () => const AdminProfilePage(),
          middlewares: [AuthGuard()],
        ),
      ],
    );
  }
}

class RtdbService {
  final FirebaseDatabase _rtdb = FirebaseDatabase.instance;

  DatabaseReference liveRef(String patientId) => _rtdb.ref('live/$patientId');

  Future<void> initRealtimeDbSchema({
    required List<String> patientIds,
    int schemaVersion = 1,
    int defaultHr = 0,
    int defaultHrv = 0,
    String defaultMedication = '—',
    String defaultSymptoms = '—',
    String defaultSleep = 'good', // 'good' or 'bad'
    String defaultStress = '—',
  }) async {
    // Meta info (optional)
    await _rtdb.ref('meta').update({
      'schemaVersion': schemaVersion,
      'updatedAt': ServerValue.timestamp,
    });

    // Build multi-location update (fast + atomic-ish)
    final Map<String, Object?> updates = {};

    for (final pid in patientIds) {
      final basePath = 'live/$pid';

      // Only set defaults. If you want to preserve existing values,
      // use update (this does).
      updates['$basePath/hr'] = defaultHr;
      updates['$basePath/hrv'] = defaultHrv;
      updates['$basePath/medication'] = defaultMedication;
      updates['$basePath/symptoms'] = defaultSymptoms;
      updates['$basePath/sleep'] = defaultSleep;
      updates['$basePath/stress'] = defaultStress;
      updates['$basePath/updatedAt'] = ServerValue.timestamp;
    }

    await _rtdb.ref().update(updates);
  }

  /// Initialize one patient node (convenience helper).
  Future<void> initLiveNodeForPatient({
    required String patientId,
    int defaultHr = 0,
    int defaultHrv = 0,
    String defaultMedication = '—',
    String defaultSymptoms = '—',
    String defaultSleep = 'good',
    String defaultStress = '—',
  }) async {
    await _rtdb.ref('live/$patientId').update({
      'hr': defaultHr,
      'hrv': defaultHrv,
      'updatedAt': ServerValue.timestamp,
    });
  }
}
