import 'dart:math' as math;
import 'dart:async';
import 'package:flutter/material.dart';
import 'package:flutter/scheduler.dart';
import 'package:get/get.dart';
import 'package:fl_chart/fl_chart.dart';
import 'package:seizrure_app/shared/controllers/app_controller.dart';
import 'package:seizrure_app/shared/routes/app_routes.dart';
import '../../shared/widgets/ecg_painter.dart';

class StatusCard extends GetView<AppController> {
  const StatusCard({super.key, required this.radius});
  final double radius;

  @override
  Widget build(BuildContext context) {
    return Obx(() {
      final bool isDanger = controller.hasActiveAlert.value;

      return GestureDetector(
        onTap: isDanger ? () => Get.toNamed(Routes.alertHistory) : null,
        child: Container(
          height: 92,
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(radius),
            gradient: LinearGradient(
              begin: Alignment.centerLeft,
              end: Alignment.centerRight,
              colors: isDanger
                  ? [const Color(0xFFE57373), const Color(0xFFD32F2F)]
                  : [const Color(0xFF7FE3D7), const Color(0xFF5FB6BE)],
            ),
            boxShadow: [
              BoxShadow(
                color: isDanger
                    ? const Color(0x44D32F2F)
                    : const Color(0x22000000),
                blurRadius: 16,
                offset: const Offset(0, 10),
              ),
            ],
          ),
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 14),
            child: Row(
              children: [
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        isDanger
                            ? 'seizure_detected'.tr
                            : controller.statusKey.value.tr,
                        style: const TextStyle(
                          color: Colors.white,
                          fontSize: 18,
                          fontWeight: FontWeight.w700,
                          letterSpacing: 0.2,
                        ),
                      ),
                      const SizedBox(height: 10),
                      Text(
                        isDanger
                            ? 'seek_help'.tr.replaceAll('\n', ' ')
                            : 'last_updated'.trParams({
                                'time': controller.lastUpdated.value,
                              }),
                        style: const TextStyle(
                          color: Color(0xCCFFFFFF),
                          fontSize: 12,
                          fontWeight: FontWeight.w500,
                        ),
                      ),
                    ],
                  ),
                ),
                Container(
                  width: 34,
                  height: 34,
                  decoration: BoxDecoration(
                    borderRadius: BorderRadius.circular(10),
                    color: const Color(0x26FFFFFF),
                  ),
                  child: Icon(
                    isDanger ? Icons.warning_amber_rounded : Icons.sync_alt,
                    size: 18,
                    color: Colors.white,
                  ),
                ),
              ],
            ),
          ),
        ),
      );
    });
  }
}

class EcgCard extends StatefulWidget {
  const EcgCard({super.key, required this.radius});
  final double radius;

  @override
  State<EcgCard> createState() => _EcgCardState();
}

class _EcgCardState extends State<EcgCard> {
  final AppController _appController = Get.find<AppController>();
  final List<FlSpot> _spots = [];
  double _xValue = 0;
  Timer? _timer;

  @override
  void initState() {
    super.initState();
    // Pre-fill with baseline
    for (int i = 0; i < 60; i++) {
      _spots.add(FlSpot(_xValue, 70));
      _xValue += 1;
    }

    _timer = Timer.periodic(const Duration(milliseconds: 200), (timer) {
      if (_appController.isDataActive.value) {
        setState(() {
          _spots.add(FlSpot(_xValue, _appController.bpm.value.toDouble()));
          _xValue += 1;
          if (_spots.length > 60) {
            _spots.removeAt(0);
          }
        });
      }
    });
  }

  @override
  void dispose() {
    _timer?.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(widget.radius),
        boxShadow: const [
          BoxShadow(
            color: Color(0x14000000),
            blurRadius: 18,
            offset: Offset(0, 10),
          ),
        ],
      ),
      child: Padding(
        padding: const EdgeInsets.fromLTRB(14, 12, 14, 14),
        child: Column(
          children: [
            Row(
              children: [
                Text(
                  'heart_rate_trend'.tr,
                  style: const TextStyle(
                    fontSize: 12,
                    fontWeight: FontWeight.w700,
                    color: Color(0xFF2A2E35),
                  ),
                ),
                const Spacer(),
                const Icon(
                  Icons.show_chart_rounded,
                  size: 14,
                  color: Color(0xFF3AA6D1),
                ),
              ],
            ),
            const SizedBox(height: 16),
            Container(
              height: 130,
              width: double.infinity,
              padding: const EdgeInsets.only(right: 16, top: 12, bottom: 8),
              decoration: BoxDecoration(
                color: const Color(0xFFF7FAFC),
                borderRadius: BorderRadius.circular(14),
                border: Border.all(color: const Color(0xFFE6EDF4)),
              ),
              child: ClipRRect(
                borderRadius: BorderRadius.circular(14),
                child: LineChart(
                  LineChartData(
                    gridData: FlGridData(
                      show: true,
                      drawVerticalLine: true,
                      horizontalInterval: 30,
                      verticalInterval: 10,
                      getDrawingHorizontalLine: (value) => FlLine(
                        color: const Color(0xFFE6EDF4),
                        strokeWidth: 1,
                      ),
                      getDrawingVerticalLine: (value) => FlLine(
                        color: const Color(0xFFE6EDF4),
                        strokeWidth: 1,
                      ),
                    ),
                    titlesData: const FlTitlesData(show: false),
                    borderData: FlBorderData(show: false),
                    minX: _spots.isNotEmpty ? _spots.first.x : 0,
                    maxX: _spots.isNotEmpty ? _spots.last.x : 0,
                    minY: 40,
                    maxY: 180,
                    lineBarsData: [
                      LineChartBarData(
                        spots: _spots,
                        isCurved: true,
                        color: const Color(0xFF3AA6D1),
                        barWidth: 3,
                        isStrokeCapRound: true,
                        dotData: const FlDotData(show: false),
                        belowBarData: BarAreaData(
                          show: true,
                          color: const Color(0xFF3AA6D1).withOpacity(0.15),
                        ),
                      ),
                    ],
                  ),
                  duration: const Duration(milliseconds: 0),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class MetricsCard extends GetView<AppController> {
  const MetricsCard({super.key, required this.radius});
  final double radius;

  @override
  Widget build(BuildContext context) {
    return Container(
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
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 14),
        child: Obx(() {
          return Row(
            children: [
              Expanded(
                child: Metric(
                  title: 'heart_rate'.tr,
                  value: '${controller.bpm.value}',
                  unit: 'unit_bpm'.tr,
                ),
              ),
              const DividerV(),
              Expanded(
                child: Metric(
                  title: 'heart_rate_variability'.tr,
                  value: '${controller.hrv.value}',
                  unit: 'unit_ms'.tr,
                ),
              ),
            ],
          );
        }),
      ),
    );
  }
}

class DividerV extends StatelessWidget {
  const DividerV({super.key});

  @override
  Widget build(BuildContext context) {
    return Container(
      width: 1,
      height: 40,
      margin: const EdgeInsets.symmetric(horizontal: 8),
      color: const Color(0xFFEAF0F6),
    );
  }
}

class Metric extends StatelessWidget {
  const Metric({
    super.key,
    required this.title,
    required this.value,
    required this.unit,
  });
  final String title;
  final String value;
  final String unit;

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Text(
          title,
          style: const TextStyle(
            fontSize: 10,
            fontWeight: FontWeight.w800,
            color: Color(0xFF7C8A99),
            letterSpacing: 0.2,
          ),
        ),
        const SizedBox(height: 8),
        Row(
          mainAxisAlignment: MainAxisAlignment.center,
          crossAxisAlignment: CrossAxisAlignment.end,
          children: [
            Text(
              value,
              style: const TextStyle(
                fontSize: 20,
                fontWeight: FontWeight.w900,
                color: Color(0xFF2A2E35),
              ),
            ),
            const SizedBox(width: 4),
            Padding(
              padding: const EdgeInsets.only(bottom: 3),
              child: Text(
                unit,
                style: const TextStyle(
                  fontSize: 10,
                  fontWeight: FontWeight.w700,
                  color: Color(0xFF8A98A8),
                ),
              ),
            ),
          ],
        ),
      ],
    );
  }
}
