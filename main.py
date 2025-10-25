import cv2
import mediapipe as mp
import time
import math
import os
import pygame

# ======================
# الإعدادات العامة
# ======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# تهيئة الأصوات
try:
    pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
    print("✅ Pygame mixer initialized successfully.")
except Exception as e:
    print(f"❌ Failed to initialize pygame mixer: {e}")

# ملفات الصوت (يفضل WAV)
scroll_sound = os.path.join(BASE_DIR, "scroll.wav")
camera_sound = os.path.join(BASE_DIR, "camera.wav")
playpause_sound = os.path.join(BASE_DIR, "playpause.wav")

def play_sound(path):
    try:
        if os.path.exists(path):
            snd = pygame.mixer.Sound(path)
            snd.set_volume(1.0)
            snd.play()
            print(f"🔊 Playing sound: {os.path.basename(path)}")
        else:
            print(f"⚠️ Sound file not found: {path}")
    except Exception as e:
        print(f"❌ Sound playback error for {path}: {e}")

# ======================
# Mediapipe إعدادات
# ======================
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Cooldown في الثواني
THUMB_COOLDOWN = 2
INDEX_COOLDOWN = 1.5

last_thumb_time = 0
last_index_time = 0

# حالة التشغيل
is_playing = False

# ======================
# الدالة الرئيسية
# ======================
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

        # عكس الصورة لتكون مثل المرآة
        image = cv2.flip(image, 1)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, hand_handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = hand_handedness.classification[0].label  # "Left" أو "Right"
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # الحصول على إحداثيات الإبهام والسبابة
                h, w, _ = image.shape
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_TIP]

                thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
                index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)

                # حساب المسافة بين الإبهام والسبابة
                distance = math.hypot(index_x - thumb_x, index_y - thumb_y)

                # -------------------------------
                # 1️⃣ الإبهام = لقطة (كاميرا)
                # -------------------------------
                current_time = time.time()
                if distance < 40 and (current_time - last_thumb_time > THUMB_COOLDOWN):
                    last_thumb_time = current_time
                    play_sound(camera_sound)
                    print("📸 لقطة بالكاميرا!")

                # -------------------------------
                # 2️⃣ اليد اليسرى = Play / Pause
                # -------------------------------
                if label == "Left":
                    if distance > 90 and (current_time - last_index_time > INDEX_COOLDOWN):
                        last_index_time = current_time
                        is_playing = not is_playing
                        play_sound(playpause_sound)
                        print("▶️ تشغيل" if is_playing else "⏸️ إيقاف مؤقت")

                # -------------------------------
                # 3️⃣ حركة السبابة = تمرير
                # -------------------------------
                if label == "Right" and (current_time - last_index_time > INDEX_COOLDOWN):
                    if distance > 100:
                        last_index_time = current_time
                        play_sound(scroll_sound)
                        print("🖱️ تمرير!")

        cv2.imshow("GestureApp", image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
