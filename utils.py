import time

import cv2 as cv
import mediapipe as mp
import numpy as np
import torch
import pandas as pd

from torch_geometric.data import Data
from config import gesture_labels, MediaPipeConfigs


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

edges = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20)
]
edges = torch.tensor(edges, dtype=torch.long).t().contiguous()

def mediapipe_process(image, model):
    image.flags.writeable = False
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    results = model.process(image)
    image.flags.writeable = True
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

    return image, results

def extract_keypoint_values(hand_landmarks):
    key_points = (np.array([[pos.x, pos.y, pos.z] for pos in hand_landmarks.landmark], dtype=np.float32).flatten()
                  if hand_landmarks
                  else np.zeros(21 * 3)
    )

    return key_points

def normalize_landmarks(landmarks):
    landmarks = landmarks.reshape(21, 3)
    x, y, w, h = calc_bounding_rect(landmarks)
    image_height, image_width = MediaPipeConfigs.height, MediaPipeConfigs.width

    center_x, center_y = x + w / 2, y + h / 2
    normalized_landmarks = []

    for landmark in landmarks:
        norm_x = (landmark[0] * image_width - center_x) / w
        norm_y = (landmark[1] * image_height - center_y) / h
        norm_z = landmark[2]  # Z value remains unchanged
        normalized_landmarks.append([norm_x, norm_y, norm_z])

    normalized_landmarks = np.array(normalized_landmarks)
    norm = np.linalg.norm(normalized_landmarks, axis=1)
    normalized_landmarks /= norm[:, np.newaxis]

    return normalized_landmarks

def calc_bounding_rect(landmarks):
    image_height, image_width = MediaPipeConfigs.height, MediaPipeConfigs.width
    landmark_array = []

    for landmark in landmarks:
        landmark_x = min(int(landmark[0] * image_width), image_width - 1)
        landmark_y = min(int(landmark[1] * image_height), image_height - 1)
        landmark_array.append([landmark_x, landmark_y])

    landmark_array = np.array(landmark_array, dtype=np.int32)
    x, y, w, h = cv.boundingRect(landmark_array)

    return x, y, w, h

def draw_bounding_rect(image, brect):
    x, y, w, h = brect
    cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return image

def create_graph_with_label(landmarks, label):
    landmarks = normalize_landmarks(landmarks)
    x = torch.tensor(landmarks, dtype=torch.float)

    return Data(x=x, edge_index=edges, y=torch.tensor([label], dtype=torch.long))

def create_graph_without_label(landmarks):
    landmarks = normalize_landmarks(landmarks)
    graph = Data(x=torch.tensor(landmarks, dtype=torch.float), edge_index=edges)

    return graph


def load_graph_data_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    data_list = []

    for i in range(len(df)):
        row = df.iloc[i]
        landmarks = row[:63].values.reshape(21, 3).astype('float32')
        label = gesture_labels[row[-1]]

        graph = create_graph_with_label(landmarks, label)
        data_list.append(graph)

    return data_list

def clear_csv_truncate(file_path):
    with open(file_path, "w") as file:
        file.truncate(0)

def calc_fps(prev_time):
    current_time = time.time()
    fps = 1.0 / (current_time - prev_time) if current_time != prev_time else 0.0
    return fps, current_time
