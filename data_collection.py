import csv
import os
import mediapipe as mp
import cv2 as cv
import numpy as np

from config import gesture_labels, MediaPipeConfigs, CSV_PATH
from utils import mediapipe_process, extract_keypoint_values, calc_bounding_rect, draw_bounding_rect


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


def setup_data_collection():
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, mode='w') as file:
            writer = csv.writer(file)
            header = [f"{i}_{axis}" for i in range(21) for axis in ['x', 'y', 'z']] + ['label']
            writer.writerow(header)

'''
def setup_folders():
    for gesture in gesture_labels:
        gesture_path = os.path.join(DATA_PATH, gesture)
        os.makedirs(gesture_path, exist_ok=True)

        try:
            existing_folders = os.listdir(gesture_path)
            if existing_folders:
                dirmax = np.max(np.array(existing_folders).astype(int))
            else:
                dirmax = 0
        except ValueError:
            dirmax = 0

        for sequence in range(1, n_sequences + 1):
            sequence_path = os.path.join(gesture_path, str(dirmax + sequence))
            os.makedirs(sequence_path, exist_ok=True)

'''


def collect_keypoint_values_to_csv():
    dc_flag = False
    cap = None

    gesture_names = list(gesture_labels.keys())
    current_gesture_index = 0

    try:
        config = MediaPipeConfigs()
        cap = cv.VideoCapture(config.device)
        cap.set(cv.CAP_PROP_FRAME_WIDTH, config.width)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, config.height)

        with mp_hands.Hands(
                model_complexity=0,
                max_num_hands=2,
                min_detection_confidence=config.min_detection_confidence,
                min_tracking_confidence=config.min_tracking_confidence) as hands:

            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    continue

                image, results = mediapipe_process(image, hands)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            image,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style())

                        # Pre-processing
                        landmarks = np.array([[landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks.landmark])
                        brect = calc_bounding_rect(landmarks)
                        image = draw_bounding_rect(image, brect)

                        # Collect key points
                        key_points = extract_keypoint_values(hand_landmarks)
                        flat_key_points = np.array(key_points).flatten().tolist()

                        if dc_flag:
                            with open(CSV_PATH, mode='a', newline='') as file:
                                writer = csv.writer(file)
                                current_gesture_label = gesture_names[current_gesture_index]
                                writer.writerow(flat_key_points + [current_gesture_label])


                key = cv.waitKey(1) & 0xFF

                if key == ord('s'):
                    dc_flag = not dc_flag

                elif ord('1') <= key <= ord('9'):
                    selected_index = int(9 - abs(ord('9') - key)) - 1
                    if 0 <= selected_index < len(gesture_labels):
                        current_gesture_index = selected_index

                elif key == ord('q'):
                    break

                image = cv.flip(image, 1)
                status_text = f"Gesture: {gesture_names[current_gesture_index]} | Collecting: {'ON' if dc_flag else 'OFF'}"
                cv.putText(image, status_text, (10, 30), cv.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0) if dc_flag else (0, 0, 255), 2)

                cv.imshow('MediaPipe Hands', image)

    finally:
        cap.release()
        cv.destroyAllWindows()


if __name__ == '__main__':
    setup_data_collection()
    collect_keypoint_values_to_csv()
