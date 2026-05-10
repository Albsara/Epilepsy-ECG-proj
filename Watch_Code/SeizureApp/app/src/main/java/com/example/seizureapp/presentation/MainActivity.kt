package com.example.seizureapp.presentation

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.lifecycle.lifecycleScope
import androidx.wear.compose.material.*
import androidx.wear.tooling.preview.devices.WearDevices
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlin.random.Random


// Activity الرئيسية للتطبيق وتشغيل الواجهة
class MainActivity : ComponentActivity() {

    // حالة تشغيل أو إيقاف القياس
    private val isMeasuringState = mutableStateOf(false)

    // قيمة نبض القلب BPM المعروضة على الشاشة
    private val bpmText = mutableStateOf("--")

    // قيمة HRV المعروضة على الشاشة
    private val hrvText = mutableStateOf("--")

    // النص الذي يوضح حالة التطبيق (Relax / Stress / Stopped)
    private val statusText = mutableStateOf("Relax Mode")

    // تحديد هل الوضع الحالي Stress أو Relax
    private val isStressMode = mutableStateOf(false)

    // مهمة تشغيل التحديث المستمر في الخلفية
    private var simulationJob: Job? = null


    // يتم استدعاؤها عند بدء التطبيق
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // ربط واجهة Compose مع الـ Activity
        setContent {
            WearApp(
                bpm = bpmText.value, // إرسال BPM للواجهة
                hrv = hrvText.value, // إرسال HRV للواجهة
                status = statusText.value, // إرسال الحالة
                isMeasuring = isMeasuringState.value, // حالة التشغيل
                isStressMode = isStressMode.value, // وضع Stress أو Relax

                onStart = { startMeasuring() }, // استدعاء تشغيل القياس
                onStop = { stopMeasuring() }, // استدعاء إيقاف القياس
                onRelax = { setRelax() }, // تغيير إلى وضع Relax
                onStress = { setStress() } // تغيير إلى وضع Stress
            )
        }
    }


    // تغيير الوضع إلى Relax (راحة)
    private fun setRelax() {
        isStressMode.value = false // تعطيل وضع التوتر
        statusText.value = "Relax Mode" // تحديث النص
    }


    // تغيير الوضع إلى Stress (توتر)
    private fun setStress() {
        isStressMode.value = true // تفعيل وضع التوتر
        statusText.value = "Stress Mode" // تحديث النص
    }


    // بدء محاكاة قياس نبض القلب و HRV
    private fun startMeasuring() {
        if (isMeasuringState.value) return // منع التشغيل إذا كان شغال بالفعل

        isMeasuringState.value = true // تفعيل حالة القياس

        simulationJob?.cancel() // إيقاف أي مهمة سابقة

        // تشغيل Coroutine في الخلفية
        simulationJob = lifecycleScope.launch {

            delay(1000) // تأخير بسيط لمحاكاة بدء القياس

            // حلقة تحديث مستمرة طالما القياس شغال
            while (isActive && isMeasuringState.value) {

                val bpm: Int // متغير نبض القلب
                val hrv: Int // متغير HRV

                // تحديد القيم حسب الوضع
                if (isStressMode.value) {
                    bpm = Random.nextInt(95, 115) // نبض مرتفع في التوتر
                    hrv = Random.nextInt(10, 30) // HRV منخفض في التوتر
                } else {
                    bpm = Random.nextInt(60, 75) // نبض طبيعي في الراحة
                    hrv = Random.nextInt(50, 90) // HRV أعلى في الراحة
                }

                bpmText.value = bpm.toString() // تحديث BPM في الواجهة
                hrvText.value = hrv.toString() // تحديث HRV في الواجهة

                delay(1000) // انتظار ثانية قبل التحديث القادم
            }
        }
    }


    // إيقاف عملية القياس
    private fun stopMeasuring() {
        isMeasuringState.value = false // إيقاف الحالة
        statusText.value = "Stopped" // تحديث النص
        bpmText.value = "--" // إعادة BPM للقيمة الافتراضية
        hrvText.value = "--" // إعادة HRV للقيمة الافتراضية
        simulationJob?.cancel() // إيقاف المهمة الخلفية
    }
}


// واجهة المستخدم الرئيسية
@Composable
fun WearApp(
    bpm: String, // قيمة نبض القلب القادمة من MainActivity
    hrv: String, // قيمة HRV القادمة من MainActivity
    status: String, // حالة النظام
    isMeasuring: Boolean, // هل القياس شغال
    isStressMode: Boolean, // هل الوضع Stress

    onStart: () -> Unit, // تشغيل القياس
    onStop: () -> Unit, // إيقاف القياس
    onRelax: () -> Unit, // تفعيل وضع الراحة
    onStress: () -> Unit // تفعيل وضع التوتر
) {

    MaterialTheme {

        Column(
            modifier = Modifier
                .fillMaxSize() // ملء الشاشة
                .background(MaterialTheme.colors.background) // خلفية
                .padding(8.dp), // مسافة داخلية
            horizontalAlignment = Alignment.CenterHorizontally, // توسيط أفقي
            verticalArrangement = Arrangement.Center // توسيط عمودي
        ) {

            Spacer(Modifier.height(6.dp)) // مسافة بين العناصر

            // صف يحتوي BPM و HRV بجانب بعض
            Row(
                horizontalArrangement = Arrangement.SpaceEvenly,
                modifier = Modifier.fillMaxWidth()
            ) {

                // عرض BPM
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    Text("BPM", style = MaterialTheme.typography.caption2)
                    Text(
                        bpm,
                        style = MaterialTheme.typography.title2,
                        fontWeight = FontWeight.Bold
                    )
                }

                // عرض HRV
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    Text("HRV", style = MaterialTheme.typography.caption2)
                    Text(
                        hrv,
                        style = MaterialTheme.typography.title3
                    )
                }
            }

            Spacer(Modifier.height(6.dp)) // مسافة

            // عرض حالة التطبيق
            Text(
                status,
                style = MaterialTheme.typography.caption2,
                textAlign = TextAlign.Center
            )

            Spacer(Modifier.height(6.dp)) // مسافة

            // أزرار تغيير الوضع
            Row(
                horizontalArrangement = Arrangement.spacedBy(4.dp)
            ) {

                Chip(
                    onClick = onRelax, // استدعاء وضع الراحة
                    label = { Text("Relax") },
                    modifier = Modifier.size(70.dp, 28.dp)
                )

                Chip(
                    onClick = onStress, // استدعاء وضع التوتر
                    label = { Text("Stress") },
                    modifier = Modifier.size(75.dp, 28.dp)
                )
            }

            Spacer(Modifier.height(6.dp)) // مسافة

            // زر تشغيل أو إيقاف القياس
            Chip(
                onClick = if (isMeasuring) onStop else onStart, // تبديل بين Start و Stop
                label = {
                    Text(
                        if (isMeasuring) "Stop" else "Start", // نص الزر حسب الحالة
                        textAlign = TextAlign.Center
                    )
                },
                modifier = Modifier.size(70.dp, 36.dp)
            )
        }
    }
}