import 'package:flutter/material.dart';
import 'package:get/get.dart';
import 'controllers/auth_controller.dart';

class Routes {
  static const login = '/login';
  static const forgot = '/forgot';
  static const dashboard = '/dashboard';
  static const patients = '/patients';
  static const caregivers = '/caregivers';
  static const alertsHistory = '/alerts';
  static const addPatient = '/patients/add';
  static const patientDetail = '/patients/detail';
  static const editPatient = '/patients/edit';
  static const profile = '/profile';
}

class AuthGuard extends GetMiddleware {
  final AuthController auth = Get.find<AuthController>();

  @override
  RouteSettings? redirect(String? route) {
    if (!auth.isLoggedIn.value &&
        route != Routes.login &&
        route != Routes.forgot) {
      return const RouteSettings(name: Routes.login);
    }
    return null;
  }
}
