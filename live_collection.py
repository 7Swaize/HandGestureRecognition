import copy
import torch
import time
import cv2 as cv
import mediapipe as mp
import numpy as np

from GNN_model import GCN
from config import MediaPipeConfigs, gesture_labels, MODEL_PATH
from utils import mediapipe_process, calc_bounding_rect, draw_bounding_rect, create_graph_without_label, extract_keypoint_values, calc_fps

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


def main():
    try:
        config = MediaPipeConfigs()
        cap = cv.VideoCapture(config.device)
        cap.set(cv.CAP_PROP_FRAME_WIDTH, config.width)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, config.height)

        model = GCN(out_channels=len(gesture_labels))
        model.load_state_dict(torch.load(MODEL_PATH))
        model.eval()

        prev_time = time.time()

        with mp_hands.Hands(
                model_complexity=0,
                max_num_hands=2,
                min_detection_confidence=config.min_detection_confidence,
                min_tracking_confidence=config.min_tracking_confidence) as hands:
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    print("Error: Could not read frame.")
                    continue

                image, results = mediapipe_process(image, hands)
                debug_image = copy.deepcopy(image)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            debug_image,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style())

                        # Pre-processing
                        brect = calc_bounding_rect(np.array([[landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks.landmark]))
                        debug_image = draw_bounding_rect(debug_image, brect)

                        # Model
                        predicted_class, probabilities = model.predict(create_graph_without_label(extract_keypoint_values(hand_landmarks)))
                        predicted_class_text = f'Predicted Class: {predicted_class}'


                debug_image = cv.flip(debug_image, 1)
                fps, prev_time = calc_fps(prev_time)

                try:
                    cv.putText(debug_image, predicted_class_text, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0),2)
                except NameError:
                    pass

                cv.putText(debug_image, f'FPS: {int(fps)}', (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                cv.imshow('MediaPipe Hands', debug_image)

                if cv.waitKey(1) & 0xFF == ord('q'):
                    break
    finally:
        cap.release()
        cv.destroyAllWindows()


if __name__ == '__main__':
    main()
