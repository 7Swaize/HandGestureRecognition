import os

# Configuration class for MediaPipe settings
class MediaPipeConfigs:
    device = 0
    width = 960
    height = 540
    use_static_image_mode = False
    min_detection_confidence = 0.7
    min_tracking_confidence = 0.5

# Data collection settings
CSV_PATH = os.path.join('MP_Data', 'Data.csv')
MODEL_PATH = os.path.join('Model', 'gesture_model.pth')

gesture_labels = {
    'hand_closed': 0,
    'hand_three': 1,
    'hand_open': 2,
    'hand_zero': 3
}
