import cv2
import mediapipe as mp # for hand tracking
import time # framerate checking

mp_hands = mp.solutions.hands
# if true it will continuously look for movement (slow), false it will only when it detects movement
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2)
mp_draw = mp.solutions.drawing_utils

# get webcam
cap = cv2.VideoCapture("http://192.168.101.102:4747/video")

# for fps
current_time = 0
previous_time = 0

if not cap.isOpened():
    print("Error: Couldn't open webcam")
    exit()


while True:
    success, img = cap.read()

    if not success:
        print("Error: Couldn't read frame")
        break

    # img = cv2.resize(img, (640, 480))  # Adjust as needed

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands.process(img_rgb)

    hand_landmarks = results.multi_hand_landmarks

    if hand_landmarks:
        # for each hand in hand landmarks
        for hand in hand_landmarks:
            for id, landmark in enumerate(hand.landmark):
                # print(id, landmark)
                height, width, channels = img.shape
                cx, cy = int(landmark.x * width), int(landmark.y * height)

                cv2.putText(img, str(id), (cx, cy), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 3)
            
            mp_draw.draw_landmarks(img, hand, mp_hands.HAND_CONNECTIONS)

    # FPS calculation every 5 frames
    if cv2.getTickCount() % 5 == 0:
        current_time = time.time()
        fps = 1 / (current_time - previous_time)
        previous_time = current_time
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 0, 255), 3)

    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break


cap.release()
cv2.destroyAllWindows()