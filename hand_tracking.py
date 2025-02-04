from typing import List
import cv2
import mediapipe as mp  # for hand tracking
import time  # framerate checking


class HandDetectorFindModel:
    def __init__(self, id, x_pos, y_pos):
        self.id = id
        self.x_pos = x_pos
        self.y_pos = y_pos


class HandDetector:
    def __init__(
        self,
        mode=False,
        max_hands=2,
        model_complexity=1,
        detection_con=0.5,
        track_con=0.5,
    ):
        self.mode = mode
        self.max_hands = max_hands
        self.model_complexity = model_complexity
        self.detection_con = detection_con
        self.track_con = track_con

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.max_hands,
            model_complexity=self.model_complexity,
            min_detection_confidence=self.detection_con,
            min_tracking_confidence=self.track_con,
        )
        self.mp_draw = mp.solutions.drawing_utils

        # for fps
        self.current_time = 0
        self.previous_time = 0

    def find_hands(self, img, draw=True):
        # img = cv2.resize(img, (640, 480))  # Adjust as needed
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.processed_hand_results = self.hands.process(img_rgb)
        hand_landmarks = self.processed_hand_results.multi_hand_landmarks

        if hand_landmarks:
            # for each hand in hand landmarks
            for hand in hand_landmarks:
                for id, landmark in enumerate(hand.landmark):
                    if draw:
                        # print(id, landmark)
                        height, width, channels = img.shape
                        cx, cy = int(landmark.x * width), int(landmark.y * height)

                        cv2.putText(
                            img,
                            str(id),
                            (cx, cy),
                            cv2.FONT_HERSHEY_COMPLEX,
                            0.5,
                            (255, 0, 255),
                            1,
                        )

                if draw:
                    self.mp_draw.draw_landmarks(
                        img, hand, self.mp_hands.HAND_CONNECTIONS
                    )

        return img

    def find_position(self, img, hand_number: int = 0) -> List[HandDetectorFindModel]:
        # list of landmarks
        results: List[HandDetectorFindModel] = []

        if (
            self.processed_hand_results
            and self.processed_hand_results.multi_hand_landmarks
        ):
            hand = self.processed_hand_results.multi_hand_landmarks[hand_number]

            for id, landmark in enumerate(hand.landmark):
                height, width, channels = img.shape
                cx, cy = int(landmark.x * width), int(landmark.y * height)
                # results.append([id, cx, cy])
                results.append(HandDetectorFindModel(id, cx, cy))

        return results

    def add_fps_to_output(self, img):
        # FPS calculation every 5 frames
        if cv2.getTickCount() % 5 == 0:
            self.current_time = time.time()
            fps = 1 / (self.current_time - self.previous_time)
            self.previous_time = self.current_time
            cv2.putText(
                img,
                str(int(fps)),
                (10, 70),
                cv2.FONT_HERSHEY_COMPLEX,
                3,
                (255, 0, 255),
                3,
            )


def main():
    detector = HandDetector()

    # get webcam
    cap = cv2.VideoCapture("http://192.168.101.102:4747/video")

    if not cap.isOpened():
        print("Error: Couldn't open webcam")
        exit()

    while True:
        success, img = cap.read()

        if not success:
            print("Error: Couldn't read frame")
            break

        img = detector.find_hands(img, True)

        detector.add_fps_to_output(img)

        cv2.imshow("Image", img)

        if cv2.waitKey(1) & 0xFF == ord("q"):  # Press 'q' to quit
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
