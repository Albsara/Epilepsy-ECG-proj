import 'dart:async';
import 'dart:ui';
import 'package:firebase_core/firebase_core.dart';
import 'package:firebase_database/firebase_database.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:flutter/material.dart';
import 'package:flutter_background_service/flutter_background_service.dart';
import 'package:flutter_background_service_android/flutter_background_service_android.dart';
import 'package:flutter_background_service_platform_interface/flutter_background_service_platform_interface.dart';
import 'package:flutter_local_notifications/flutter_local_notifications.dart';
import 'package:flutter/services.dart';
import 'package:get_storage/get_storage.dart';

import '../utils/ai_evaluator.dart';
import '../models/live_data_model.dart';
import '../../../firebase_options.dart';

@pragma('vm:entry-point')
void onStart(ServiceInstance service) async {
  // 1. Initialize Bindings First - MUST be the very first thing
  DartPluginRegistrant.ensureInitialized();
  WidgetsFlutterBinding.ensureInitialized();

  print('Background Service: onStart called');

  if (service is AndroidServiceInstance) {
    // CRITICAL: Signal to Android immediately that we are starting as foreground
    // This prevents the "DidNotStartInTimeException"
    // On some devices, calling this twice or too early/late causes issues.
    // If isForegroundMode was set to true in configure, this might be redundant but safe.
    service.setAsForegroundService();
  }
  await GetStorage.init();

  // 1. Initialize Firebase - Await it properly now that foreground signal is sent
  try {
    await Firebase.initializeApp(
      options: DefaultFirebaseOptions.currentPlatform,
    ).timeout(const Duration(seconds: 10));
    print('Background Service: Firebase initialized');
  } catch (e) {
    print('Background Service: Firebase initialization error: $e');
    // If Firebase fails to initialize, we can't really proceed with monitoring
    if (service is AndroidServiceInstance) {
      service.setForegroundNotificationInfo(
        title: 'Seizure Monitor Error',
        content: 'Failed to connect to database.',
      );
    }
    Future.delayed(const Duration(seconds: 5), () => service.stopSelf());
    return;
  }

  // 2. Load preferences
  final box = GetStorage();
  final String? uid = box.read('user_id');
  final String? role = box.read('user_role');

  if (uid == null || role == null) {
    print('Background Service: User not logged in. Stopping.');
    if (service is AndroidServiceInstance) {
      service.setForegroundNotificationInfo(
        title: 'Seizure Monitor',
        content: 'Service is stopping (not logged in)',
      );
    }
    // Give it a moment to register the foreground status before stopping
    Future.delayed(const Duration(seconds: 2), () {
      service.stopSelf();
    });
    return;
  }

  final aiEvaluator = AiEvaluator();
  try {
    print('Background Service: Loading AI model...');
    final String response = await rootBundle.loadString('assets/model.json');
    aiEvaluator.loadModel(response);
    print('Background Service: AI Model loaded');
  } catch (e) {
    print('Background Service: AI Model load error: $e');
  }

  final firestore = FirebaseFirestore.instance;
  final rtdb = FirebaseDatabase.instance.ref();
  final notifications = FlutterLocalNotificationsPlugin();
  final Set<String> notifiedIds = {};

  if (role == 'patient') {
    DateTime? lastAlertTime;
    bool isLastUnhandled = false;

    // Listen to own alerts to track handled status and timing for throttling
    firestore
        .collection('patients')
        .doc(uid)
        .collection('alerts')
        .orderBy('time', descending: true)
        .limit(1)
        .snapshots()
        .listen((snap) {
          if (snap.docs.isNotEmpty) {
            final data = snap.docs.first.data();
            isLastUnhandled = data['is_handled'] == false;
            lastAlertTime = (data['time'] as Timestamp?)?.toDate();
          }
        });

    rtdb.child('live_data').child(uid).onValue.listen((event) {
      if (event.snapshot.value != null) {
        final data = Map<String, dynamic>.from(event.snapshot.value as Map);
        final liveModel = LiveDataModel(
          hr: data['hr'] ?? 0,
          hrv: data['hrv'] ?? 0,
          medication: data['medication'] ?? 'yes',
          symptoms: data['symptoms'] ?? 'no',
          sleep: data['sleep'] ?? 'good',
          stress: data['stress'] ?? 'low',
        );

        if (aiEvaluator.isReady() && aiEvaluator.evaluate(liveModel)) {
          // Check if we should throttle
          final bool withinCoolingPeriod =
              lastAlertTime != null &&
              DateTime.now().difference(lastAlertTime!).inMinutes < 5;

          if (!(withinCoolingPeriod && isLastUnhandled)) {
            _handleBackgroundSeizure(uid, liveModel, firestore);
            // Update local tracking to prevent immediate repeat before stream updates
            lastAlertTime = DateTime.now();
            isLastUnhandled = true;
          }
        }
      }
    });
  } else {
    firestore
        .collectionGroup('alerts')
        .where('is_handled', isEqualTo: false)
        .snapshots()
        .listen((snapshot) {
          final List<String> patientIds = List<String>.from(
            box.read('assigned_patient_ids') ?? [],
          );
          for (var doc in snapshot.docs) {
            final alertId = doc.id;
            final pId = doc.reference.parent.parent?.id;

            if (pId != null &&
                patientIds.contains(pId) &&
                !notifiedIds.contains(alertId)) {
              final alertTime = (doc.data()['time'] as Timestamp).toDate();
              if (DateTime.now().difference(alertTime).inMinutes.abs() < 10) {
                _showLocalNotification(
                  notifications,
                  'Emergency!',
                  'A seizure has been detected.',
                );
              }
              notifiedIds.add(alertId);
            }
          }
        });
  }

  service.on('stopService').listen((event) {
    service.stopSelf();
  });
}

void _showLocalNotification(
  FlutterLocalNotificationsPlugin plugin,
  String title,
  String body,
) {
  plugin.show(
    id: DateTime.now().millisecondsSinceEpoch ~/ 1000,
    title: title,
    body: body,
    notificationDetails: const NotificationDetails(
      android: AndroidNotificationDetails(
        'seizure_alerts_channel',
        'Emergency Alerts',
        importance: Importance.max,
        priority: Priority.high,
        icon: '@mipmap/ic_launcher',
      ),
    ),
  );
}

void _handleBackgroundSeizure(
  String pId,
  LiveDataModel data,
  FirebaseFirestore firestore,
) async {
  // 1. Set alert field to true in RTDB
  // Using 'live_data/$pId' to match the listener path in onStart
  await FirebaseDatabase.instance.ref().child('live_data').child(pId).child('alert').set(true);

  // 2. Add Firestore alert
  await firestore.collection('patients').doc(pId).collection('alerts').add({
    'heartRate': data.hr,
    'hrv': data.hrv,
    'medication': data.medication,
    'symptoms': data.symptoms,
    'sleep': data.sleep,
    'stress': data.stress,
    'time': FieldValue.serverTimestamp(),
    'is_handled': false,
    'createdAt': FieldValue.serverTimestamp(),
  });
}

@pragma('vm:entry-point')
class BackgroundMonitorService {
  static const String notificationChannelId = 'background_monitor_channel';
  static const int notificationId = 888;

  static Future<void> initializeService() async {
    final service = FlutterBackgroundService();

    // Create notification channel for Android
    const AndroidNotificationChannel channel = AndroidNotificationChannel(
      notificationChannelId,
      'Seizure Monitoring Service',
      description: 'Running health monitoring in background',
      importance: Importance.max, // Increased importance
    );

    final FlutterLocalNotificationsPlugin flutterLocalNotificationsPlugin =
        FlutterLocalNotificationsPlugin();

    await flutterLocalNotificationsPlugin
        .resolvePlatformSpecificImplementation<
          AndroidFlutterLocalNotificationsPlugin
        >()
        ?.createNotificationChannel(channel);

    await service.configure(
      androidConfiguration: AndroidConfiguration(
        onStart: onStart,
        autoStart: true,
        isForegroundMode: true,
        notificationChannelId: notificationChannelId,
        initialNotificationTitle: 'Seizure Monitor Active',
        initialNotificationContent: 'Monitoring vital signs...',
        foregroundServiceNotificationId: notificationId,
      ),
      iosConfiguration: IosConfiguration(
        autoStart: true,
        onForeground: onStart,
        onBackground: (ServiceInstance service) => Future.value(true),
      ),
    );
  }
}
