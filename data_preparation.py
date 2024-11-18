import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from PIL import Image

def load_spectrogram_data(images_path, genres, img_size=(128, 128)):
    image_data = []
    labels = []
    for genre in genres:
        print(f"Loading {genre} spectrogram images...")
        genre_path = os.path.join(images_path, genre)
        for img_file in os.listdir(genre_path):
            if img_file.endswith('.png'):
                img_path = os.path.join(genre_path, img_file)
                img = Image.open(img_path).convert('RGB')  # Convert to RGB format
                img = img.resize(img_size)
                image_data.append(np.array(img))
                labels.append(genre)

    # Convert data to numpy arrays and normalize
    X = np.array(image_data, dtype=np.float32) / 255.0
    y = np.array(labels)

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    return X, y_encoded, label_encoder

def prepare_train_test_split(X, y, test_size=0.2):
    return train_test_split(X, y, test_size=test_size, random_state=42)