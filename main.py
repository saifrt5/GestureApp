import cv2
import mediapipe as mp
import time
import math
import os
import pygame

# ======================
# الإعدادات العامة والمسارات
# ======================
# تصحيح المشكلة: os.path.dirname يتم تنفيذه الآن بشكل صحيح في Python
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# تهيئة الأصوات
try:
    # تهيئة مكبر الصوت بتردد قياسي
    pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
    print("✅ Pygame mixer initialized successfully.")
except Exception as e:
    print(f"❌ Failed to initialize pygame mixer: {e}")

# تعريف ملفات الصوت (يجب أن تكون هذه الملفات موجودة في نفس مجلد ملف البايثون)
# ملاحظة: يمكنك تغيير المسارات لتشير إلى ملفات صوتية موجودة لديك
scroll_sound_path = os.path.join(BASE_DIR, "scroll.wav")
camera_sound_path = os.path.join(BASE_DIR, "camera.wav")
playpause_sound_path = os.path.join(BASE_DIR, "playpause.wav")


def play_sound(path):
    """تشغيل ملف صوتي والتحقق من وجوده."""
    try:
        if os.path.exists(path):
            # التأكد من عدم تشغيل ملفات صوتية كثيرة في نفس الوقت
            if not pygame.mixer.get_busy():
                snd = pygame.mixer.Sound(path)
                snd.set_volume(0.8)  # خفض الصوت قليلاً
                snd.play()
                print(f"🔊 Playing sound: {os.path.basename(path)}")
        else:
            # يمكنك إزالة هذا السطر بعد التأكد من وضع الملفات الصوتية
            print(f"⚠️ Sound file not found: {path}") 
    except Exception as e:
        print(f"❌ Sound playback error for {path}: {e}")

# ======================
# Mediapipe إعدادات وحالة
# ======================
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# عتبة المسافة (Distance Thresholds) بالبكسل - تعتمد على دقة الكاميرا
THUMB_INDEX_PINCH_THRESHOLD = 40  # للمسك (لقطة/كاميرا)
LEFT_HAND_OPEN_THRESHOLD = 90  # لليد اليسرى المفتوحة (تشغيل/إيقاف)
RIGHT_HAND_SCROLL_THRESHOLD = 100 # لليد اليمنى المفتوحة (تمرير)

# فترات التهدئة (Cooldown) لمنع التكرار السريع
THUMB_COOLDOWN = 2.0  # ثواني
INDEX_COOLDOWN = 1.5  # ثواني

last_thumb_time = 0.0
last_index_time = 0.0
is_playing = False


# ======================
# الدالة الرئيسية
# ======================
# ابدأ تشغيل الكاميرا (0 عادة هو الكاميرا الافتراضية)
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=2
) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("❌ لم يتم التقاط الإطار من الكاميرا.")
            break

        # عكس الصورة (للمرآة) وتحويلها إلى RGB لـ Mediapipe
        image = cv2.flip(image, 1)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        # النص الذي سيظهر على الشاشة
        status_text = "Idle"
        
        # ------------------------------------
        # تحليل النتائج
        # ------------------------------------
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, hand_handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = hand_handedness.classification[0].label  # "Left" أو "Right"
                
                # رسم المعالم على الشاشة
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # الحصول على إحداثيات الإبهام والسبابة
                h, w, _ = image.shape
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_TIP]

                # تحويل الإحداثيات من 0-1 إلى بكسل
                thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
                index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)

                # حساب المسافة بين الإبهام والسبابة
                distance = math.hypot(index_x - thumb_x, index_y - thumb_y)

                # -------------------------------
                # 1️⃣ الإبهام والسبابة (القرص) = لقطة (كاميرا)
                # -------------------------------
                current_time = time.time()
                if distance < THUMB_INDEX_PINCH_THRESHOLD and (current_time - last_thumb_time > THUMB_COOLDOWN):
                    last_thumb_time = current_time
                    play_sound(camera_sound_path)
                    status_text = "📸 Captured!"

                # -------------------------------
                # 2️⃣ اليد اليسرى (فتح اليد) = Play / Pause
                # -------------------------------
                if label == "Left":
                    if distance > LEFT_HAND_OPEN_THRESHOLD and (current_time - last_index_time > INDEX_COOLDOWN):
                        last_index_time = current_time
                        global is_playing
                        is_playing = not is_playing
                        play_sound(playpause_sound_path)
                        status_text = "▶️ Playing" if is_playing else "⏸️ Paused"

                # -------------------------------
                # 3️⃣ اليد اليمنى (فتح اليد) = تمرير
                # -------------------------------
                elif label == "Right":
                    # هنا يجب أن نحدد مؤشرًا لفتح اليد، المسافة بين الإبهام والسبابة طريقة جيدة.
                    if distance > RIGHT_HAND_SCROLL_THRESHOLD and (current_time - last_index_time > INDEX_COOLDOWN):
                        last_index_time = current_time
                        play_sound(scroll_sound_path)
                        status_text = "🖱️ Scrolling!"

        # ------------------------------------
        # عرض النتائج
        # ------------------------------------
        # عرض حالة التشغيل على الشاشة
        cv2.putText(image, status_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        # عرض حالة تشغيل/إيقاف مؤقت
        play_status = "Playing" if is_playing else "Paused"
        cv2.putText(image, f"Status: {play_status}", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)


        cv2.imshow("Gesture Control App", image)
        
        # اضغط Esc للخروج
        if cv2.waitKey(5) & 0xFF == 27:
            break

# ------------------------------------
# تنظيف
# ------------------------------------
cap.release()
cv2.destroyAllWindows()
print("👋 البرنامج أغلق بنجاح.")
