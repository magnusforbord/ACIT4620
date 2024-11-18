import os
import numpy as np
import random
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from data_preparation import load_spectrogram_data, prepare_train_test_split
from model import create_cnn_model
from sklearn.utils import class_weight

# Set paths and genres
images_path = os.path.join(os.getcwd(), 'Data', 'images_original')
genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()

# Load and prepare data
X, y, label_encoder = load_spectrogram_data(images_path, genres)
X_train, X_test, y_train, y_test = prepare_train_test_split(X, y)

# Convert labels to categorical
y_train_cat = to_categorical(y_train, num_classes=len(genres)).astype(np.float32)
y_test_cat = to_categorical(y_test, num_classes=len(genres)).astype(np.float32)

# Data Augmentation with more aggressive parameters
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
datagen.fit(X_train)

# Create model
input_shape = (128, 128, 3)
model = create_cnn_model(input_shape, num_classes=len(genres))

# Early Stopping with increased patience
early_stopping = EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True)

# Convert y_train to an integer array to avoid errors with class_weight
y_train_int = y_train.astype(int)

# Train the model with class weights
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_int),
    y=y_train_int
)

# Ensure X_train and X_test are of type float32
X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)

# Train the model
train_generator = datagen.flow(X_train, y_train_cat, batch_size=64)
validation_data = (X_test, y_test_cat)

history = model.fit(
    train_generator,
    steps_per_epoch=len(X_train) // 16,
    epochs=50,
    validation_data=validation_data,
    callbacks=[early_stopping],
)

# Print data shapes
print("X_train shape:", X_train.shape)
print("y_train_cat shape:", y_train_cat.shape)
print("X_test shape:", X_test.shape)
print("y_test_cat shape:", y_test_cat.shape)

# Function to count occurrences of each class
def count_classes(y, label_encoder):
    class_counts = {}
    for i in range(len(label_encoder.classes_)):
        class_counts[label_encoder.classes_[i]] = np.sum(y == i)
    return class_counts

# Print class distribution
print("Training Set Class Distribution:", count_classes(y_train, label_encoder))
print("Validation Set Class Distribution:", count_classes(y_test, label_encoder))

# Function to plot sample images
def plot_sample_images(X, y, label_encoder, set_name="Training"):
    plt.figure(figsize=(15, 15))
    indices = random.sample(range(X.shape[0]), 25)  # Plot 25 random samples
    for i, idx in enumerate(indices):
        plt.subplot(5, 5, i+1)
        plt.imshow(X[idx])
        plt.axis('off')
        label_index = np.argmax(y[idx])  # y is one-hot encoded
        label = label_encoder.classes_[label_index]
        plt.title(label, fontsize=8)
    plt.suptitle(f"Sample Images from {set_name} Set", fontsize=16)
    plt.show()

# Visualize raw images
def visualize_raw_images(X, num_images=5):
    plt.figure(figsize=(10, 10))
    indices = random.sample(range(len(X)), num_images)
    for i, idx in enumerate(indices):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(X[idx])
        plt.axis('off')
    plt.show()

# Visualize raw images after loading
visualize_raw_images(X)

# Plot samples from training and validation sets
plot_sample_images(X_train, y_train_cat, label_encoder, set_name="Training")
plot_sample_images(X_test, y_test_cat, label_encoder, set_name="Validation")

# Save the model
model.save('music_genre_cnn_model.keras')

# Plot training history
def plot_training_history(history):
    plt.figure(figsize=(12, 4))
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

plot_training_history(history)
