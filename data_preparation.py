import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_spectrogram_data(images_path, genres, img_size=(128, 128)):
    image_data = []
    labels = []
    for genre in genres:
        print(f"Loading {genre} spectrogram images...")
        genre_path = os.path.join(images_path, genre)
        for img_file in os.listdir(genre_path):
            if img_file.endswith('.png'):
                img_path = os.path.join(genre_path, img_file)
                img = cv2.imread(img_path)
                img = cv2.resize(img, img_size)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                image_data.append(img)
                labels.append(genre)

    # Convert data to numpy arrays and normalize
    X = np.array(image_data) / 255.0
    y = np.array(labels)

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    return X, y_encoded, label_encoder

def prepare_train_test_split(X, y, test_size=0.2):
    return train_test_split(X, y, test_size=test_size, random_state=42)
