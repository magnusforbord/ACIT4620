import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC

# Set the correct dataset path
dataset_path = os.path.join(os.getcwd(), 'Data', 'genres_original')

# Initialize lists to hold features and labels
features_list = []
labels_list = []

# Define a function to extract features from a file
def extract_features(file_name):
    try:
        # Load the audio file (30 seconds)
        audio, sample_rate = librosa.load(file_name, duration=30)
        
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        return mfccs_mean
    except Exception as e:
        print(f"Error encountered while parsing file: {file_name}")
        return None

# List of genres
genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()

# Loop through each genre
for genre in genres:
    print(f"Processing {genre} files...")
    genre_path = os.path.join(dataset_path, genre)
    for filename in os.listdir(genre_path):
        if filename.endswith('.wav'):
            file_path = os.path.join(genre_path, filename)
            # Extract features
            data = extract_features(file_path)
            if data is not None:
                features_list.append(data)
                labels_list.append(genre)

# Create a DataFrame with extracted features and corresponding labels
features_df = pd.DataFrame(features_list)
features_df['label'] = labels_list

# Encode the labels
label_encoder = LabelEncoder()
features_df['label_encoded'] = label_encoder.fit_transform(features_df['label'])

# Separate features and labels
X = features_df.iloc[:, :-2].values  # Features
y = features_df['label_encoded'].values  # Encoded labels

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Create and train the SVM classifier
svm = SVC(kernel='linear', C=1.0, random_state=42)
svm.fit(X_train, y_train)

# Predict the test set and evaluate the model
y_pred = svm.predict(X_test)

# Classification report
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
print(report)

# Plot the confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Genre')
plt.ylabel('True Genre')
plt.show()
