import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Constants
DATASET_PATH = 'brain_tumor_dataset'
IMG_SIZE = 128

# Load and preprocess data
def load_data(dataset_path):
    data = []
    labels = []
    for label, category in enumerate(["no", "yes"]):  # 0 for no tumor, 1 for tumor
        folder = os.path.join(dataset_path, category)
        for img_name in os.listdir(folder):
            img_path = os.path.join(folder, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            data.append(img)
            labels.append(label)
    return np.array(data), np.array(labels)

# Load data
X, y = load_data(DATASET_PATH)
X = X / 255.0  # Normalize
X = X.reshape(-1, IMG_SIZE, IMG_SIZE, 1)  # Add channel dimension
y = to_categorical(y, 2)  # One-hot encode labels (0: no, 1: yes)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Image augmentation
datagen = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
                             width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15, horizontal_flip=True)
datagen.fit(X_train)

# Build CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')  # 2 classes: yes or no
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Display model architecture
model.summary()

# Train model
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = model.fit(datagen.flow(X_train, y_train, batch_size=32),
                    epochs=20,
                    validation_data=(X_test, y_test),
                    callbacks=[early_stop])

# Save model
model.save("brain_tumor_model.h5")

print(" Model saved as brain_tumor_model.h5")

test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")

