import 'package:flutter/material.dart';

class AppColors {
  static const white = Colors.white;
  static const bgSoft = Color(0xFFF6FAFB);

  static const text = Color(0xFF1E2A2F);
  static const muted = Color(0xFF6B7B84);

  static const accent = Color(0xFF7ABCC8); // main accent
  static const alert = Color(0xFFF0767B); // alerts
  static const button = Color(0xFF7BC0DE); // buttons
  static const border = Color(0xFFE8EEF2);
}

String formatDateTimeAmPm(DateTime dt) {
  const months = [
    'Jan',
    'Feb',
    'Mar',
    'Apr',
    'May',
    'Jun',
    'Jul',
    'Aug',
    'Sep',
    'Oct',
    'Nov',
    'Dec',
  ];
  final m = months[dt.month - 1];
  final day = dt.day.toString().padLeft(2, '0');
  final year = dt.year.toString();

  int hour = dt.hour;
  final ampm = hour >= 12 ? 'PM' : 'AM';
  hour = hour % 12;
  if (hour == 0) hour = 12;

  final hh = hour.toString().padLeft(2, '0');
  final mm = dt.minute.toString().padLeft(2, '0');

  return '$m $day, $year $hh:$mm $ampm';
}

EdgeInsets responsivePadding(BuildContext context) {
  final w = MediaQuery.of(context).size.width;
  if (w >= 1200) return const EdgeInsets.all(24);
  if (w >= 900) return const EdgeInsets.all(20);
  if (w >= 600) return const EdgeInsets.all(16);
  return const EdgeInsets.all(12);
}

bool isSameDay(DateTime a, DateTime b) =>
    a.year == b.year && a.month == b.month && a.day == b.day;
