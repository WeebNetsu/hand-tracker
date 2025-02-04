import cv2
import hand_tracking as hdm
import math
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import numpy as np


devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
# volume.GetMute()
# volume.GetMasterVolumeLevel()
# print(volume.GetVolumeRange())
# volume.SetMasterVolumeLevel(-40.0, None)

volume_range = volume.GetVolumeRange()
max_volume = volume_range[1]
min_volume = volume_range[0]

min_finger_distance = 10
max_finger_distance = 90

# get webcam
cap = cv2.VideoCapture("http://192.168.101.102:4747/video")

if not cap.isOpened():
    print("Error: Couldn't open webcam")
    exit()

detector = hdm.HandDetector(detection_con=0.7)

while True:
    success, img = cap.read()

    if not success:
        print("Error: Couldn't read frame")
        break

    img = detector.find_hands(img, True)

    landmarks_list = detector.find_position(img)
    if len(landmarks_list) != 0:
        # landmark 4 is thumb
        thumb_point = next(
            (landmark for landmark in landmarks_list if landmark.id == 4), None
        )
        # landmark 8 is index finger
        index_point = next(
            (landmark for landmark in landmarks_list if landmark.id == 8), None
        )

        if index_point and thumb_point:
            center_x, center_y = (thumb_point.x_pos + index_point.x_pos) // 2, (
                thumb_point.y_pos + index_point.y_pos
            ) // 2

            cv2.circle(
                img, (thumb_point.x_pos, thumb_point.y_pos), 10, (255, 0, 0), cv2.FILLED
            )
            cv2.circle(
                img, (index_point.x_pos, index_point.y_pos), 10, (255, 0, 0), cv2.FILLED
            )
            cv2.line(
                img,
                (thumb_point.x_pos, thumb_point.y_pos),
                (index_point.x_pos, index_point.y_pos),
                (0, 255, 0),
                3,
            )
            cv2.circle(img, (center_x, center_y), 10, (0, 0, 255), cv2.FILLED)

            length = math.hypot(
                index_point.x_pos - thumb_point.x_pos,
                index_point.y_pos - thumb_point.y_pos,
            )

            # this wil normalise the values, since distance and the volume values are not the same at all
            vol = np.interp(
                length,
                [min_finger_distance, max_finger_distance],
                [min_volume, max_volume],
            )

            volume.SetMasterVolumeLevel(vol, None)
            # print(length)

            if length <= min_finger_distance:
                cv2.circle(img, (center_x, center_y), 10, (0, 255, 0), cv2.FILLED)

    detector.add_fps_to_output(img)

    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord("q"):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()


cap.release()
cv2.destroyAllWindows()
