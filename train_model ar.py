import numpy as np # مكتبة العمليات الحسابية والمصفوفات
import pandas as pd # مكتبة تحليل البيانات والتعامل مع ملفات الإكسل
import tensorflow as tf # المكتبة الأساسية لبناء الشبكات العصبية (الذكاء الاصطناعي)
from sklearn.model_selection import train_test_split # أداة لتقسيم البيانات لتدريب واختبار
from sklearn.ensemble import RandomForestClassifier # خوارزمية "الغابة العشوائية" (النموذج المعلم)
from sklearn.metrics import accuracy_score # أداة لقياس دقة النموذج

# ---------------------------
# (A) تحميل البيانات من ملف إكسل
# ---------------------------
excel_path = "synthetic_seizure_dataset_20k.xlsx"  # تحديد مسار ملف البيانات
df = pd.read_excel(excel_path, sheet_name="encoded") # قراءة ورقة العمل المسميات "encoded"

# التأكد من أن جميع الأعمدة المطلوبة موجودة في الملف
expected_cols = ["HR", "HRV", "Medication", "Symptoms", "Sleep", "Stress", "risk"]
missing = [c for c in expected_cols if c not in df.columns]
if missing:
    raise ValueError(f"هناك أعمدة مفقودة في ملف الإكسل: {missing}")

# استخراج المدخلات (X) وهي الحساسات، والنتيجة (y) وهي مستوى الخطر
X = df[["HR", "HRV", "Medication", "Symptoms", "Sleep", "Stress"]].astype(np.float32).values
y = df["risk"].astype(int).values

# تقسيم البيانات: 80% للتدريب و 20% لاختبار جودة النموذج بعد التدريب
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("حجم البيانات المحملة:", df.shape)
print("توزيع حالات الخطر:\n", pd.Series(y).value_counts())

# ---------------------------
# (B) تدريب نموذج "الغابة العشوائية")
# ---------------------------
rf = RandomForestClassifier(
    n_estimators=300, # بناء 300 شجرة قرار لضمان دقة عالية
    random_state=42, # تثبيت العشوائية لضمان نفس النتائج في كل مرة
    class_weight="balanced", # موازنة البيانات إذا كانت حالات التشنج قليلة
    n_jobs=-1 # استخدام جميع أنوية المعالج لتسريع التدريب
)
rf.fit(X_train, y_train) # بدء عملية التعلم من بيانات التدريب

rf_val_pred = rf.predict(X_val) # تجربة النموذج على بيانات الاختبار
print("دقة النموذج :", accuracy_score(y_val, rf_val_pred)) #91

# استخراج "الاحتمالات" بدلاً من (0 أو 1) لتعليمها للنموذج الصغير (Soft Labels)
y_train_soft = rf.predict_proba(X_train).astype(np.float32) 
y_val_soft   = rf.predict_proba(X_val).astype(np.float32)

# ---------------------------
# (C) بناء وتدريب الشبكة العصبية 
# ---------------------------
student = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)), # استقبال 6 مدخلات (بيانات الحساسات)
    tf.keras.layers.Dense(32, activation="relu"), # طبقة مخفية أولى بها 32 عصب لمعالجة العلاقات
    tf.keras.layers.Dense(16, activation="relu"), # طبقة مخفية ثانية بها 16 عصب لتقليل حجم النموذج
    tf.keras.layers.Dense(2, activation="softmax"), # طبقة المخرج: تعطي احتمالين (خطر أو أمان)
])

# تجهيز النموذج الطالب ليتعلم كيف يقلد احتمالات النموذج المعلم
student.compile(
    optimizer="adam", # خوارزمية ذكية لتعديل الأوزان بسرعة
    loss=tf.keras.losses.KLDivergence(), # دالة خسارة تقيس مدى تشابه احتمالات الطالب مع المعلم
    metrics=["accuracy"] # عرض الدقة أثناء التدريب
)

# بدء تدريب النموذج الطالب لمدة 10 دورات (Epochs)
student.fit(
    X_train, y_train_soft,
    validation_data=(X_val, y_val_soft),
    epochs=10,
    batch_size=256 # معالجة 256 عينة في كل خطوة لتسريع العملية
)

# قياس دقة النموذج الطالب النهائي مقارنة بالنتائج الحقيقية
student_val_pred = np.argmax(student.predict(X_val), axis=1)
print("دقة النموذج الطالب النهائي:", accuracy_score(y_val, student_val_pred))

# ---------------------------
# (D) تحويل النموذج لصيغة TFLite (للعمل على الجوال)
# ---------------------------
converter = tf.lite.TFLiteConverter.from_keras_model(student) # تجهيز المحول
converter.optimizations = [tf.lite.Optimize.DEFAULT] # ضغط النموذج ليكون حجمه صغيراً جداً
tflite_model = converter.convert() # تنفيذ عملية التحويل

# حفظ النموذج المحول في ملف بصيغة .tflite
with open("seizure_model.tflite", "wb") as f:
    f.write(tflite_model)

print("تم حفظ النموذج بنجاح: seizure_model.tflite")

# ---------------------------
# (E) دالة تجريبية لمحاكاة إدخال بيانات من تطبيق جوال
# ---------------------------
def encode_app_input(HR, HRV, Medication, Symptoms, Sleep, Stress):
    # تحويل الكلمات النصية إلى أرقام (0 أو 1) لأن النموذج لا يفهم إلا الأرقام
    med = 1.0 if Medication.lower() == "yes" else 0.0
    sym = 1.0 if Symptoms.lower() == "yes" else 0.0
    slp = 1.0 if Sleep.lower() == "good" else 0.0
    strs = 1.0 if Stress.lower() == "high" else 0.0
    return np.array([[HR, HRV, med, sym, slp, strs]], dtype=np.float32)

# تجربة حقيقية لبيانات شخص (نبض 92، توتر عالٍ، نوم سيء)
sample = encode_app_input(92, 28, "no", "yes", "bad", "high")

pred_probs = student.predict(sample) # طلب التوقع من النموذج
print("نسب احتمالات التوقع [أمان، خطر]:", pred_probs)
print("التصنيف النهائي (0=أمان، 1=خطر):", int(np.argmax(pred_probs, axis=1)[0]))