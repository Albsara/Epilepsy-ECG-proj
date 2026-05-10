import 'dart:math' as math;
import 'package:flutter/material.dart';

class EcgGridPainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    final bg = Paint()..color = const Color(0xFFF7FAFC);
    canvas.drawRect(Offset.zero & size, bg);

    final major = Paint()
      ..color = const Color(0xFFE2EAF2)
      ..strokeWidth = 1;

    final minor = Paint()
      ..color = const Color(0xFFF0F4F8)
      ..strokeWidth = 1;

    const minorStep = 18.0;
    const majorEvery = 3;

    for (double x = 0; x <= size.width; x += minorStep) {
      final isMajor = ((x / minorStep).round() % majorEvery == 0);
      canvas.drawLine(
        Offset(x, 0),
        Offset(x, size.height),
        isMajor ? major : minor,
      );
    }

    for (double y = 0; y <= size.height; y += minorStep) {
      final isMajor = ((y / minorStep).round() % majorEvery == 0);
      canvas.drawLine(
        Offset(0, y),
        Offset(size.width, y),
        isMajor ? major : minor,
      );
    }
  }

  @override
  bool shouldRepaint(CustomPainter oldDelegate) => false;
}

class EcgWavePainter extends CustomPainter {
  EcgWavePainter({
    required this.waveColor,
    required this.bpm,
    required this.time,
    this.hrv = 0,
    this.verticalOffset = 0,
  });

  final Color waveColor;
  final int bpm;
  final double time; // Time in seconds for continuous scrolling
  final int hrv;
  final double verticalOffset;

  @override
  void paint(Canvas canvas, Size size) {
    if (size.width == 0 || size.height == 0) return;

    final paint = Paint()
      ..color = waveColor
      ..strokeWidth = 2.5
      ..style = PaintingStyle.stroke
      ..strokeCap = StrokeCap.round
      ..strokeJoin = StrokeJoin.round;

    final path = Path();
    final centerY = (size.height / 2) + verticalOffset;

    // Standard ECG sweep speed: 150 pixels per second
    const double pixelsPerSecond = 150.0;
    
    // The length of one full ECG beat in pixels
    // Time per beat (s) = 60 / BPM
    final double beatPeriod = 60 / math.max(bpm, 30);
    final double beatWidth = beatPeriod * pixelsPerSecond;
    
    // Continuous offset based on time
    final double timeOffset = time * pixelsPerSecond;

    // Total pixels to draw
    const double step = 2.0;
    bool started = false;

    for (double x = 0; x <= size.width; x += step) {
      // Calculate local position within a beat cycle
      final double totalX = x + timeOffset;
      
      // Calculate which beat we are currently drawing to apply consistent HRV jitter per beat
      final int beatIndex = (totalX / beatWidth).floor();
      
      // Seeded random based on beat index to ensure the jitter is stable as the wave scrolls
      final double jitter = hrv > 0 
          ? (math.Random(beatIndex).nextDouble() - 0.5) * (hrv / 1000.0) * pixelsPerSecond 
          : 0.0;
      
      final double localX = (totalX + jitter) % beatWidth;
      
      double yOffset = 0;
      final double p = localX / beatWidth;

      // ECG Component simulation (simplified P-QRS-T)
      if (p > 0.1 && p < 0.18) {
        // P wave (Small bump)
        yOffset = math.sin((p - 0.1) / 0.08 * math.pi) * 4;
      } else if (p >= 0.2 && p < 0.22) {
        // Q (Small dip)
        yOffset = -((p - 0.2) / 0.02) * 5;
      } else if (p >= 0.22 && p < 0.26) {
        // R (Sharp spike)
        final double rP = (p - 0.22) / 0.04;
        if (rP < 0.5) {
          yOffset = -5 + (rP / 0.5) * -35;
        } else {
          yOffset = -40 + ((rP - 0.5) / 0.5) * 45;
        }
      } else if (p >= 0.26 && p < 0.28) {
        // S (dip below baseline)
        yOffset = 5 - ((p - 0.26) / 0.02) * 5;
      } else if (p > 0.45 && p < 0.6) {
        // T wave (Medium bump)
        yOffset = math.sin((p - 0.45) / 0.15 * math.pi) * 8;
      }

      final double finalY = centerY + yOffset;

      if (!started) {
        path.moveTo(x, finalY);
        started = true;
      } else {
        path.lineTo(x, finalY);
      }
    }

    // Add a bit of glow
    canvas.drawPath(
      path,
      paint..maskFilter = MaskFilter.blur(BlurStyle.solid, 0.5),
    );
  }

  @override
  bool shouldRepaint(covariant EcgWavePainter oldDelegate) =>
      oldDelegate.time != time || 
      oldDelegate.bpm != bpm || 
      oldDelegate.hrv != hrv ||
      oldDelegate.verticalOffset != verticalOffset;
}
