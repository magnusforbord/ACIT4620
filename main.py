import os
from data_preparation import load_spectrogram_data, prepare_train_test_split
from model import create_cnn_model
from utils import plot_training_history
from tensorflow.keras.utils import to_categorical

# Set paths
images_path = os.path.join(os.getcwd(), 'Data', 'images_original')

# Genres
genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()

# Load and prepare data
X, y, label_encoder = load_spectrogram_data(images_path, genres)
X_train, X_test, y_train, y_test = prepare_train_test_split(X, y)

# Convert labels to categorical (one-hot encoding)
y_train_cat = to_categorical(y_train, num_classes=len(genres))
y_test_cat = to_categorical(y_test, num_classes=len(genres))

# Create model
input_shape = (128, 128, 3)
model = create_cnn_model(input_shape, num_classes=len(genres))

# Train the model
history = model.fit(
    X_train, y_train_cat,
    epochs=30,
    batch_size=32,
    validation_data=(X_test, y_test_cat)
)

# Save the model
model.save('music_genre_cnn_model.keras')

# Plot training history
plot_training_history(history)
