import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import utils

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

import matplotlib.pyplot as plt


# Function to load and preprocess images
def load_and_preprocess_images(folder_path):
    images = []
    labels = []

    # Emotion labels corresponding to folder names
    emotion_labels = {'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'sad': 4, 'surprise': 5, 'neutral': 6}

    for emotion_folder in os.listdir(folder_path):
        emotion_path = os.path.join(folder_path, emotion_folder)
        emotion_label = emotion_labels[emotion_folder]

        for image_file in os.listdir(emotion_path):
            image_path = os.path.join(emotion_path, image_file)
            # Read and resize the image
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (48, 48))
            # Normalize pixel values
            img = img.astype('float32') / 255.0

            images.append(img)
            labels.append(emotion_label)

    return np.array(images), to_categorical(labels, num_classes=7)

# Load and preprocess training images
X_train, y_train = load_and_preprocess_images('./train')

# Load and preprocess testing images
X_test, y_test = load_and_preprocess_images('./test')

# Split the dataset into training and testing sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Verify the shapes of the sets
print("Training Set - Images:", X_train.shape, "Labels:", y_train.shape)
print("Validation Set - Images:", X_val.shape, "Labels:", y_val.shape)
print("Testing Set - Images:", X_test.shape, "Labels:", y_test.shape)



# Define the CNN model
model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Display the model summary
model.summary()

# Train the model
history = model.fit(X_train, y_train, epochs=15, batch_size=64, validation_data=(X_val, y_val))
# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc}")

# Visualize training history (optional)


plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# Save the model to an HDF5 file
model.save('your_model.h5')